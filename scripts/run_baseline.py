"""
scripts/run_baseline.py

FASE 3B — Baseline Científico: TF-IDF + Logistic Regression

Experimentos executados:
  [A] In-domain stratified split — por dataset
  [B] Cross-domain: treina em X, testa em Y
  [C] LODO: Leave-One-Domain-Out
  [D] Ablation: original vs anonimizado
  [E] Ablation: configurações do TF-IDF

Outputs:
  experiments/baseline/
    results_all.json          → todas as métricas
    results_summary.csv       → tabela para TCC
    figures/                  → todos os gráficos

Execução:
    python scripts/run_baseline.py

NOTA METODOLÓGICA:
  O TF-IDF vectorizer é ajustado APENAS no conjunto de treino.
  No cross-domain, o vocabulário do dataset de teste nunca
  contamina o vocabulário aprendido. Isso é crítico para
  evitar data leakage e garantir validade dos experimentos.
"""

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_calibration_metrics,
    compute_confusion_matrix_stats,
    generalization_gap,
    format_metrics_table_row,
)
from src.evaluation.calibration import PlattScaler, compare_calibration
from src.data.splitters import (
    stratified_split,
    cross_domain_split,
    leave_one_domain_out,
    CROSS_DOMAIN_EXPERIMENTS,
    LODO_EXPERIMENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Estilo visual consistente com o EDA ──────────────────────────────────
PALETTE = {"fake": "#E74C3C", "real": "#2ECC71", "neutral": "#3498DB",
           "calib": "#9B59B6", "gap": "#E67E22"}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})

SEED = 42
np.random.seed(SEED)

# ── Diretórios ─────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data/processed/unified"
OUT_DIR    = ROOT / "experiments/baseline"
FIG_DIR    = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ════════════════════════════════════════════════════════════════════════════

def load_all_datasets(use_anonymized: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Carrega todos os datasets processados.

    use_anonymized: se True, usa text_anonymized como input do modelo.
                    Se False, usa text_cleaned.
    """
    suffix = "_anon" if use_anonymized else ""
    datasets = {}

    dataset_names = ["kaggle_fakenews", "liar", "gossipcop", "politifact"]

    for name in dataset_names:
        path = DATA_DIR / f"{name}{suffix}.parquet"
        if not path.exists():
            # Fallback: tentar sem sufixo
            path = DATA_DIR / f"{name}.parquet"
            if not path.exists():
                logger.warning(f"Dataset não encontrado: {name}{suffix}.parquet — pulando")
                continue

        df = pd.read_parquet(path)

        # Garantir coluna 'input_text' — fallback gracioso
        if use_anonymized and "text_anonymized" in df.columns:
            text_col = "text_anonymized"
        elif "text_cleaned" in df.columns:
            text_col = "text_cleaned"
        elif "input_text" in df.columns:
            text_col = "input_text"
        else:
            text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[2])
            logger.warning(f"[{name}] Usando coluna fallback: {text_col}")
        df["input_text"] = df[text_col].fillna("").astype(str)

        # Remover textos vazios e labels nulos
        mask = (df["input_text"].str.split().str.len() >= 3) & df["label"].notna()
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        if n_before - len(df) > 0:
            logger.info(f"[{name}] Removidos {n_before - len(df)} textos inválidos")

        df["label"] = df["label"].astype(int)
        datasets[name] = df
        logger.info(
            f"Carregado [{name}]: {len(df):,} artigos | "
            f"fake={int((df['label']==0).sum()):,} | "
            f"real={int((df['label']==1).sum()):,}"
        )

    return datasets


# ════════════════════════════════════════════════════════════════════════════
# MODELOS — TF-IDF + LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def build_tfidf_pipeline(
    ngram_range:  Tuple[int, int] = (1, 2),
    max_features: int             = 100_000,
    min_df:       int             = 3,
    max_df:       float           = 0.90,
    class_weight: Optional[str]   = "balanced",
    C:            float           = 1.0,
) -> Pipeline:
    """
    Constrói pipeline TF-IDF + Logistic Regression.

    Decisões de design:
      ngram_range=(1,2): unigramas + bigramas capturam frases como
                         "breaking news", "fake report", "sources say"
                         que são discriminativas mas invisíveis para unigramas.

      max_df=0.90: remove termos em >90% dos documentos (stop words implícitas).
                   Mais robusto que lista fixa de stop words para múltiplos domínios.

      min_df=3: remove termos raros (noise, typos, nomes únicos).
                Valor baixo para preservar vocabulário de domínios menores (PolitiFact).

      class_weight="balanced": compensa desbalanceamento de classes.
                               OBRIGATÓRIO para PolitiFact (83% fake) e GossipCop (24% fake).

      C=1.0: regularização L2 padrão. Valores menores (0.1, 0.01) = mais regularização
              = mais generalização. Testar como hiperparâmetro no ablation.
    """
    # Calcular class_weight baseado no dataset seria ideal, mas
    # "balanced" é suficiente e independe dos dados de treino no momento de construção
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,      # TF = 1 + log(TF): reduz impacto de termos muito frequentes
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",  # Mínimo 2 caracteres
    )
    lr = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("lr", lr)])


# ════════════════════════════════════════════════════════════════════════════
# RUNNER DE EXPERIMENTO
# ════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    model:         Pipeline,
    X_train:       pd.Series,
    y_train:       np.ndarray,
    X_val:         pd.Series,
    y_val:         np.ndarray,
    X_test:        pd.Series,
    y_test:        np.ndarray,
    experiment_id: str,
    calibrate:     bool = True,
) -> Dict[str, Any]:
    """
    Executa um experimento completo: treino → avaliação → calibração.

    Retorna dict com todas as métricas para o dataset de teste.
    """
    t0 = time.time()

    # ── Treino ─────────────────────────────────────────────────────────
    model.fit(X_train, y_train)

    # ── Probabilidades brutas (não calibradas) ─────────────────────────
    y_prob_val_raw  = model.predict_proba(X_val)[:, 1]
    y_prob_test_raw = model.predict_proba(X_test)[:, 1]
    y_pred_test     = model.predict(X_test)

    # ── Métricas no conjunto de teste (sem calibração) ─────────────────
    metrics_raw = compute_classification_metrics(
        y_test, y_pred_test, y_prob_test_raw, dataset_name=experiment_id
    )

    # ── Calibração Platt Scaling (ajustada no VAL, aplicada no TEST) ───
    calibration_result = None
    y_prob_test_cal    = y_prob_test_raw

    if calibrate and len(np.unique(y_val)) > 1:
        try:
            scaler = PlattScaler()
            scaler.fit(y_val, y_prob_val_raw)
            y_prob_test_cal = scaler.transform(y_prob_test_raw)

            calibration_result = compare_calibration(
                y_test, y_prob_test_raw, y_prob_test_cal
            )

            cal_metrics = compute_calibration_metrics(y_test, y_prob_test_cal)
            metrics_raw.update({
                "ece":           cal_metrics["ece"],
                "brier":         cal_metrics["brier"],
                "overconf_rate": cal_metrics["overconf_rate"],
                "bin_data":      cal_metrics["bin_data"],
                "calibration":   calibration_result,
            })
        except Exception as e:
            logger.warning(f"Calibração falhou para {experiment_id}: {e}")

    # ── Calibração ECE sem Platt (para comparação) ─────────────────────
    else:
        try:
            cal_metrics = compute_calibration_metrics(y_test, y_prob_test_raw)
            metrics_raw.update({
                "ece":   cal_metrics["ece"],
                "brier": cal_metrics["brier"],
                "bin_data": cal_metrics["bin_data"],
            })
        except Exception:
            pass

    # ── Confusion matrix ────────────────────────────────────────────────
    cm_stats = compute_confusion_matrix_stats(y_test, y_pred_test)
    metrics_raw.update(cm_stats)

    metrics_raw["elapsed_s"]      = round(time.time() - t0, 2)
    metrics_raw["y_prob_test"]    = y_prob_test_cal.tolist()
    metrics_raw["y_pred_test"]    = y_pred_test.tolist()
    metrics_raw["y_true_test"]    = y_test.tolist()

    logger.info(
        f"  [{experiment_id}] "
        f"Macro F1={metrics_raw.get('macro_f1', 0):.4f} | "
        f"MCC={metrics_raw.get('mcc', 0):.4f} | "
        f"AUC={metrics_raw.get('roc_auc', float('nan')):.4f} | "
        f"ECE={metrics_raw.get('ece', float('nan')):.4f}"
    )

    return metrics_raw


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENTOS
# ════════════════════════════════════════════════════════════════════════════

def run_indomain_experiments(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    [A] Experimentos in-domain: train/val/test no mesmo dataset.
    Mede performance máxima possível — será o teto de comparação.
    """
    logger.info("\n" + "="*60)
    logger.info("[A] EXPERIMENTOS IN-DOMAIN")
    logger.info("="*60)

    results = {}

    for name, df in datasets.items():
        logger.info(f"\nDataset: {name} ({len(df):,} artigos)")

        if len(df) < 50:
            logger.warning(f"Dataset {name} muito pequeno ({len(df)}), pulando")
            continue

        train, val, test = stratified_split(df, seed=SEED)

        model = build_tfidf_pipeline()

        result = run_single_experiment(
            model=model,
            X_train=train["input_text"],
            y_train=train["label"].values,
            X_val=val["input_text"],
            y_val=val["label"].values,
            X_test=test["input_text"],
            y_test=test["label"].values,
            experiment_id=f"indomain_{name}",
        )
        results[name] = result

    return results


def run_crossdomain_experiments(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    [B] Experimentos cross-domain: treina em X, testa em Y.
    Mede generalização real — o número mais importante do projeto.
    """
    logger.info("\n" + "="*60)
    logger.info("[B] EXPERIMENTOS CROSS-DOMAIN")
    logger.info("="*60)

    results = {}

    for train_ds, test_ds, exp_name in CROSS_DOMAIN_EXPERIMENTS:
        # Verificar se ambos os datasets estão disponíveis
        missing = [d for d in train_ds + test_ds if d not in datasets]
        if missing:
            logger.warning(f"Pulando {exp_name}: datasets ausentes {missing}")
            continue

        logger.info(f"\nExperimento: {exp_name}")

        train, val, test = cross_domain_split(
            dfs=datasets,
            train_datasets=train_ds,
            test_datasets=test_ds,
            seed=SEED,
        )

        model = build_tfidf_pipeline()

        result = run_single_experiment(
            model=model,
            X_train=train["input_text"],
            y_train=train["label"].values,
            X_val=val["input_text"],
            y_val=val["label"].values,
            X_test=test["input_text"],
            y_test=test["label"].values,
            experiment_id=exp_name,
        )
        results[exp_name] = result

    return results


def run_lodo_experiments(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    [C] Leave-One-Domain-Out: protocolo mais rigoroso de generalização.
    """
    logger.info("\n" + "="*60)
    logger.info("[C] LEAVE-ONE-DOMAIN-OUT (LODO)")
    logger.info("="*60)

    results = {}

    for test_domain in LODO_EXPERIMENTS:
        if test_domain not in datasets:
            logger.warning(f"Pulando LODO({test_domain}): dataset não disponível")
            continue

        available_others = [d for d in datasets if d != test_domain]
        if len(available_others) < 2:
            logger.warning(f"Pulando LODO({test_domain}): poucos datasets de treino")
            continue

        logger.info(f"\nLODO: test={test_domain}")

        train, val, test = leave_one_domain_out(
            dfs=datasets,
            test_dataset=test_domain,
            seed=SEED,
        )

        model = build_tfidf_pipeline()

        result = run_single_experiment(
            model=model,
            X_train=train["input_text"],
            y_train=train["label"].values,
            X_val=val["input_text"],
            y_val=val["label"].values,
            X_test=test["input_text"],
            y_test=test["label"].values,
            experiment_id=f"lodo_{test_domain}",
        )
        results[test_domain] = result

    return results


def run_tfidf_ablation(
    datasets: Dict[str, pd.DataFrame],
    reference_dataset: str = "kaggle_fakenews",
) -> Dict[str, Any]:
    """
    [E] Ablation: impacto das configurações do TF-IDF.
    Compara unigrams vs bigrams, max_features, regularização C.
    """
    logger.info("\n" + "="*60)
    logger.info("[E] ABLATION: CONFIGURAÇÕES TF-IDF")
    logger.info("="*60)

    if reference_dataset not in datasets:
        logger.warning(f"Dataset de referência '{reference_dataset}' não disponível")
        return {}

    df = datasets[reference_dataset]
    train, val, test = stratified_split(df, seed=SEED)

    configurations = {
        "unigram_C1":         dict(ngram_range=(1,1), C=1.0,  max_features=50_000),
        "bigram_C1":          dict(ngram_range=(1,2), C=1.0,  max_features=100_000),
        "bigram_C01":         dict(ngram_range=(1,2), C=0.1,  max_features=100_000),
        "bigram_C10":         dict(ngram_range=(1,2), C=10.0, max_features=100_000),
        "bigram_no_balance":  dict(ngram_range=(1,2), C=1.0,  max_features=100_000,
                                   class_weight=None),
        "bigram_10k_feats":   dict(ngram_range=(1,2), C=1.0,  max_features=10_000),
    }

    results = {}
    for config_name, config in configurations.items():
        logger.info(f"  Config: {config_name}")
        model = build_tfidf_pipeline(**config)
        result = run_single_experiment(
            model=model,
            X_train=train["input_text"], y_train=train["label"].values,
            X_val=val["input_text"],     y_val=val["label"].values,
            X_test=test["input_text"],   y_test=test["label"].values,
            experiment_id=f"ablation_{config_name}",
            calibrate=False,
        )
        results[config_name] = result

    return results


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagrams(
    results_indomain: Dict,
    results_cd: Dict,
) -> None:
    """
    Fig B1 — Reliability Diagrams (calibração).

    Um modelo perfeitamente calibrado tem todos os pontos na diagonal.
    Pontos acima da diagonal: underconfident (modelo diz 0.7, acerta 0.9)
    Pontos abaixo: overconfident (modelo diz 0.9, acerta 0.7) — mais comum.
    """
    all_results = {
        **{f"indomain_{k}": v for k, v in results_indomain.items()},
        **{k: v for k, v in results_cd.items()},
    }

    # Selecionar até 6 para caber na figura
    selected = list(all_results.items())[:6]
    if not selected:
        return

    n  = len(selected)
    nc = min(3, n)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4.5 * nr))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (exp_name, result) in zip(axes_flat, selected):
        bin_data = result.get("bin_data", {})
        if not bin_data or not bin_data.get("bin_centers"):
            ax.set_visible(False)
            continue

        centers     = bin_data["bin_centers"]
        accuracies  = bin_data["bin_accuracies"]
        confidences = bin_data["bin_confidences"]
        counts      = bin_data["bin_counts"]

        # Diagonal perfeita
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Calibração perfeita")

        # Gaps (diferença acc - conf)
        for conf, acc in zip(confidences, accuracies):
            color = PALETTE["fake"] if acc < conf else PALETTE["real"]
            ax.bar(conf, abs(acc - conf), bottom=min(acc, conf),
                   width=0.05, alpha=0.3, color=color)

        # Pontos de calibração
        scatter = ax.scatter(
            confidences, accuracies,
            c=counts, cmap="Blues",
            s=80, zorder=5, edgecolors="gray", linewidths=0.5
        )

        ece = result.get("ece", float("nan"))
        brier = result.get("brier", float("nan"))

        ax.set_title(
            f"{exp_name}\nECE={ece:.4f} | Brier={brier:.4f}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Confiança Prevista")
        ax.set_ylabel("Acurácia Observada")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    # Desabilitar axes extras
    for ax in axes_flat[len(selected):]:
        ax.set_visible(False)

    fig.suptitle(
        "Reliability Diagrams — TF-IDF + LR\n"
        "(Pontos abaixo da diagonal = overconfidence)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figB1_reliability_diagrams.png")
    plt.close(fig)
    logger.info("Fig B1: reliability diagrams salvo")


def plot_generalization_gap(
    results_indomain: Dict,
    results_cd: Dict,
    results_lodo: Dict,
) -> None:
    """
    Fig B2 — Generalization Gap.

    Visualiza a queda de Macro F1 entre in-domain e cross-domain.
    Este é o gráfico mais importante do TCC — quantifica shortcut learning.
    """
    rows = []

    # In-domain como referência
    for ds_name, result in results_indomain.items():
        rows.append({
            "experiment":  f"indomain\n{ds_name}",
            "macro_f1":    result.get("macro_f1", 0),
            "mcc":         result.get("mcc", 0),
            "type":        "In-Domain",
            "color":       PALETTE["real"],
        })

    # Cross-domain
    for exp_name, result in results_cd.items():
        rows.append({
            "experiment":  exp_name.replace("→", "\n→"),
            "macro_f1":    result.get("macro_f1", 0),
            "mcc":         result.get("mcc", 0),
            "type":        "Cross-Domain",
            "color":       PALETTE["neutral"],
        })

    # LODO
    for test_ds, result in results_lodo.items():
        rows.append({
            "experiment":  f"LODO\n(test={test_ds})",
            "macro_f1":    result.get("macro_f1", 0),
            "mcc":         result.get("mcc", 0),
            "type":        "LODO",
            "color":       PALETTE["fake"],
        })

    if not rows:
        return

    df_plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(rows) * 0.45 + 2)))

    for ax, metric in zip(axes, ["macro_f1", "mcc"]):
        colors_map = {"In-Domain": PALETTE["real"],
                      "Cross-Domain": PALETTE["neutral"],
                      "LODO": PALETTE["fake"]}
        colors = [colors_map[t] for t in df_plot["type"]]

        bars = ax.barh(
            df_plot["experiment"], df_plot[metric],
            color=colors, edgecolor="white", linewidth=0.8
        )

        # Labels nos bars
        for bar, val in zip(bars, df_plot[metric]):
            ax.text(
                max(0.01, val - 0.08), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=8, fontweight="bold", color="white"
            )

        metric_label = "Macro F1" if metric == "macro_f1" else "MCC"
        ax.set_xlabel(metric_label)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                   label="Baseline aleatório")

        # Linha de referência: média in-domain
        indomain_mean = df_plot[df_plot["type"] == "In-Domain"][metric].mean()
        if not np.isnan(indomain_mean):
            ax.axvline(indomain_mean, color=PALETTE["real"], linestyle="--",
                       linewidth=1.5, alpha=0.7,
                       label=f"Média In-Domain ({indomain_mean:.3f})")

        ax.set_title(f"{metric_label} por Tipo de Experimento", fontweight="bold")
        ax.legend(fontsize=8)

        # Legenda de cores à parte
        if ax == axes[0]:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=PALETTE["real"],    label="In-Domain"),
                Patch(facecolor=PALETTE["neutral"], label="Cross-Domain"),
                Patch(facecolor=PALETTE["fake"],    label="LODO"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figB2_generalization_gap.png")
    plt.close(fig)
    logger.info("Fig B2: generalization gap salvo")


def plot_tfidf_ablation(results_ablation: Dict) -> None:
    """Fig B3 — Ablation das configurações do TF-IDF."""
    if not results_ablation:
        return

    configs = list(results_ablation.keys())
    metrics = ["macro_f1", "mcc", "roc_auc"]
    metric_labels = ["Macro F1", "MCC", "ROC-AUC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results_ablation[c].get(metric, 0) for c in configs]
        colors = [PALETTE["neutral"] if v < max(values) else PALETTE["real"]
                  for v in values]

        bars = ax.bar(range(len(configs)), values, color=colors,
                      edgecolor="white", linewidth=1.2)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label if ax == axes[0] else "")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=7.5, fontweight="bold"
            )

    fig.suptitle("Ablation Study: Configurações TF-IDF\n(kaggle_fakenews in-domain)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figB3_tfidf_ablation.png")
    plt.close(fig)
    logger.info("Fig B3: ablation TF-IDF salvo")


def plot_confusion_matrices(results: Dict, title_prefix: str = "") -> None:
    """Fig B4 — Matrizes de confusão normalizadas."""
    n = min(len(results), 4)
    if n == 0:
        return

    selected = list(results.items())[:n]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (exp_name, result) in zip(axes, selected):
        y_true = np.array(result.get("y_true_test", []))
        y_pred = np.array(result.get("y_pred_test", []))

        if len(y_true) == 0:
            ax.set_visible(False)
            continue

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            linewidths=0.5,
            annot_kws={"size": 11, "weight": "bold"},
        )
        macro_f1 = result.get("macro_f1", 0)
        ax.set_title(f"{exp_name}\nMacro F1={macro_f1:.4f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    fig.suptitle(f"Matrizes de Confusão Normalizadas — {title_prefix}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"figB4_confusion_matrices_{title_prefix.lower().replace(' ', '_')}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    logger.info(f"Fig B4 ({title_prefix}): matrizes de confusão salvas")


def plot_cross_domain_heatmap(
    results_indomain: Dict,
    results_cd: Dict,
) -> None:
    """
    Fig B5 — Heatmap de Macro F1 para todos os pares treino→teste.

    Visualiza a matriz completa de generalização.
    Diagonal = in-domain. Off-diagonal = cross-domain.
    Quanto mais escuro off-diagonal, menor a generalização.
    """
    datasets = list(results_indomain.keys())
    if len(datasets) < 2:
        return

    n = len(datasets)
    matrix = np.full((n, n), np.nan)
    ds_idx = {ds: i for i, ds in enumerate(datasets)}

    # In-domain na diagonal
    for ds, result in results_indomain.items():
        i = ds_idx[ds]
        matrix[i][i] = result.get("macro_f1", np.nan)

    # Cross-domain off-diagonal
    for exp_name, result in results_cd.items():
        # exp_name é "trainDS→testDS"
        if "→" in exp_name:
            train_ds, test_ds = exp_name.split("→")
            train_ds = train_ds.strip()
            test_ds  = test_ds.strip()
            if train_ds in ds_idx and test_ds in ds_idx:
                i = ds_idx[train_ds]
                j = ds_idx[test_ds]
                matrix[i][j] = result.get("macro_f1", np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))

    mask = np.isnan(matrix)
    cmap = sns.diverging_palette(10, 150, as_cmap=True)

    sns.heatmap(
        matrix,
        annot=True, fmt=".3f",
        xticklabels=datasets,
        yticklabels=datasets,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        mask=mask,
        ax=ax,
        linewidths=1,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Macro F1"},
    )

    ax.set_xlabel("Dataset de TESTE", fontweight="bold", labelpad=10)
    ax.set_ylabel("Dataset de TREINO", fontweight="bold", labelpad=10)
    ax.set_title(
        "Macro F1: Matriz Treino × Teste\n"
        "(Diagonal=In-Domain | Off-diagonal=Cross-Domain)\n"
        "TF-IDF + Logistic Regression",
        fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # Destacar diagonal
    for i in range(min(n, n)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor="gold", lw=3))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figB5_crossdomain_heatmap.png")
    plt.close(fig)
    logger.info("Fig B5: heatmap cross-domain salvo")


# ════════════════════════════════════════════════════════════════════════════
# TABELAS PARA O TCC
# ════════════════════════════════════════════════════════════════════════════

def build_results_tables(
    results_indomain: Dict,
    results_cd: Dict,
    results_lodo: Dict,
) -> pd.DataFrame:
    """
    Monta tabela consolidada de resultados para o TCC.
    Formato: uma linha por experimento, colunas = métricas.
    """
    rows = []

    for ds, result in results_indomain.items():
        row = format_metrics_table_row(result, include_calibration=True)
        row["Tipo"]      = "In-Domain"
        row["Experimento"] = f"Train+Test: {ds}"
        rows.append(row)

    for exp, result in results_cd.items():
        row = format_metrics_table_row(result, include_calibration=True)
        row["Tipo"]        = "Cross-Domain"
        row["Experimento"] = exp
        rows.append(row)

    for ds, result in results_lodo.items():
        row = format_metrics_table_row(result, include_calibration=True)
        row["Tipo"]        = "LODO"
        row["Experimento"] = f"LODO (test={ds})"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reordenar colunas para o TCC
    col_order = [
        "Tipo", "Experimento", "N",
        "Accuracy", "Macro F1", "MCC",
        "ROC-AUC", "PR-AUC",
        "F1 (Fake)", "F1 (Real)",
        "ECE", "Brier",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("NEXUS — Fase 3B: Baseline TF-IDF + Logistic Regression")
    logger.info("=" * 60)

    # ── Verificar dados disponíveis ────────────────────────────────────
    if not DATA_DIR.exists():
        logger.error(
            f"Diretório de dados não encontrado: {DATA_DIR}\n"
            "Execute primeiro: python scripts/run_preprocessing.py"
        )
        return

    parquets = list(DATA_DIR.glob("*.parquet"))
    logger.info(f"Arquivos disponíveis em {DATA_DIR}:")
    for p in sorted(parquets):
        size_mb = p.stat().st_size / 1024 / 1024
        logger.info(f"  {p.name:45s} {size_mb:.1f} MB")

    # ── Carregar dados ─────────────────────────────────────────────────
    logger.info("\n[1/6] Carregando datasets processados...")
    datasets = load_all_datasets(use_anonymized=False)

    if not datasets:
        logger.error("Nenhum dataset carregado. Verifique os arquivos .parquet")
        return

    # ── Experimentos ───────────────────────────────────────────────────
    logger.info("\n[2/6] Experimentos in-domain...")
    results_indomain = run_indomain_experiments(datasets)

    logger.info("\n[3/6] Experimentos cross-domain...")
    results_cd = run_crossdomain_experiments(datasets)

    logger.info("\n[4/6] Leave-One-Domain-Out...")
    results_lodo = run_lodo_experiments(datasets)

    logger.info("\n[5/6] Ablation TF-IDF...")
    results_ablation = run_tfidf_ablation(datasets)

    # ── Visualizações ──────────────────────────────────────────────────
    logger.info("\n[6/6] Gerando visualizações...")
    plot_reliability_diagrams(results_indomain, results_cd)
    plot_generalization_gap(results_indomain, results_cd, results_lodo)
    plot_tfidf_ablation(results_ablation)
    plot_confusion_matrices(results_indomain, "In-Domain")
    plot_cross_domain_heatmap(results_indomain, results_cd)

    # ── Tabela consolidada ─────────────────────────────────────────────
    df_results = build_results_tables(results_indomain, results_cd, results_lodo)
    csv_path = OUT_DIR / "results_summary.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Tabela de resultados salva: {csv_path}")

    # ── Salvar JSON completo ───────────────────────────────────────────
    all_results = {
        "indomain":  {k: {kk: vv for kk, vv in v.items()
                          if kk not in ["y_prob_test", "y_pred_test", "y_true_test", "bin_data"]}
                      for k, v in results_indomain.items()},
        "crossdomain": {k: {kk: vv for kk, vv in v.items()
                            if kk not in ["y_prob_test", "y_pred_test", "y_true_test", "bin_data"]}
                        for k, v in results_cd.items()},
        "lodo":      {k: {kk: vv for kk, vv in v.items()
                          if kk not in ["y_prob_test", "y_pred_test", "y_true_test", "bin_data"]}
                      for k, v in results_lodo.items()},
        "ablation":  {k: {kk: vv for kk, vv in v.items()
                          if kk not in ["y_prob_test", "y_pred_test", "y_true_test", "bin_data"]}
                      for k, v in results_ablation.items()},
    }

    json_path = OUT_DIR / "results_all.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON completo salvo: {json_path}")

    # ── Resumo científico no terminal ──────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMO CIENTÍFICO — FASE 3B: BASELINE TF-IDF + LR")
    print("=" * 70)

    print("\n📊 IN-DOMAIN (referência superior):")
    for ds, r in results_indomain.items():
        print(
            f"  {ds:20s} | "
            f"F1={r.get('macro_f1',0):.4f} | "
            f"MCC={r.get('mcc',0):.4f} | "
            f"AUC={r.get('roc_auc',float('nan')):.4f} | "
            f"ECE={r.get('ece',float('nan')):.4f}"
        )

    print("\n🌍 CROSS-DOMAIN (generalização real):")
    for exp, r in results_cd.items():
        gap = ""
        # Calcular gap vs in-domain mais próximo
        train_ds = exp.split("→")[0].strip() if "→" in exp else ""
        if train_ds in results_indomain:
            g = generalization_gap(results_indomain[train_ds], r)
            gap = f" | gap={g:+.4f}"
        print(
            f"  {exp:30s} | "
            f"F1={r.get('macro_f1',0):.4f} | "
            f"MCC={r.get('mcc',0):.4f}{gap}"
        )

    print("\n🔄 LODO (generalização máxima):")
    for ds, r in results_lodo.items():
        print(
            f"  LODO(test={ds:15s}) | "
            f"F1={r.get('macro_f1',0):.4f} | "
            f"MCC={r.get('mcc',0):.4f}"
        )

    if results_indomain and (results_cd or results_lodo):
        id_f1s  = [r.get("macro_f1", 0) for r in results_indomain.values()]
        cd_f1s  = [r.get("macro_f1", 0) for r in results_cd.values()]
        ld_f1s  = [r.get("macro_f1", 0) for r in results_lodo.values()]
        all_cd  = cd_f1s + ld_f1s

        mean_id = np.mean(id_f1s) if id_f1s else 0
        mean_cd = np.mean(all_cd) if all_cd else 0

        print(f"\n⚠  GENERALIZATION GAP:")
        print(f"  Média In-Domain  Macro F1 = {mean_id:.4f}")
        print(f"  Média Cross/LODO Macro F1 = {mean_cd:.4f}")
        print(f"  Gap médio = {mean_id - mean_cd:+.4f}")
        print(
            f"\n  {'⚠ SHORTCUT LEARNING SEVERO' if mean_id - mean_cd > 0.30 else '⚠ Generalização limitada' if mean_id - mean_cd > 0.15 else '✓ Generalização aceitável'}"
        )

    print(f"\n📁 Outputs: {OUT_DIR}")
    print(f"   Figuras:  {FIG_DIR}")
    print(f"   Tabela:   {csv_path}")
    print(f"   JSON:     {json_path}")
    print("=" * 70)
    print("\n✅ PRÓXIMO PASSO: Fase 3B-II — Ablation anonimização")
    print("   python scripts/run_anonymization_ablation.py")


if __name__ == "__main__":
    main()