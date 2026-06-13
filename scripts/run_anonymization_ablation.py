"""
scripts/run_anonymization_ablation.py

Ablation Study: Impacto da Anonimização de Entidades

Hipóteses testadas:
  H1: Anonimização reduz Macro F1 in-domain
      Razão: remove features discriminativas legítimas (bylines Reuters)
      Predição: queda de 1-5% in-domain

  H2: Anonimização aumenta Macro F1 cross-domain
      Razão: remove source memorization e entity shortcuts
      Predição: ganho de 2-10% cross-domain

  H3: O benefício é maior para modelos léxicos (TF-IDF) que para transformers
      Razão: transformers aprendem representações contextuais mais ricas
      que são menos dependentes de entidades específicas
      (Esta hipótese será testada na Fase 4 com RoBERTa)

  H4: O benefício varia por domínio
      Kaggle (fonte única): maior benefício
      LIAR (claims políticos): menor benefício (speaker é sinal legítimo)

Execução:
    python scripts/run_anonymization_ablation.py

Output:
    experiments/ablation_anonymization/
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_baseline import (
    load_all_datasets,
    build_tfidf_pipeline,
    run_single_experiment,
    SEED,
)
from src.data.splitters import (
    stratified_split,
    cross_domain_split,
    CROSS_DOMAIN_EXPERIMENTS,
)
from src.evaluation.metrics import generalization_gap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s"
)
logger = logging.getLogger(__name__)

PALETTE = {"original": "#3498DB", "anonymized": "#E67E22",
           "gain": "#2ECC71", "loss": "#E74C3C"}

OUT_DIR = ROOT / "experiments/ablation_anonymization"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_condition(
    datasets: Dict,
    condition: str,  # "original" ou "anonymized"
) -> Dict:
    """
    Roda todos os experimentos para uma condição (com/sem anonimização).
    """
    logger.info(f"\n{'─'*50}")
    logger.info(f"Condição: {condition.upper()}")
    logger.info(f"{'─'*50}")

    results_id = {}
    results_cd = {}

    # In-domain
    for name, df in datasets.items():
        logger.info(f"  In-domain [{name}]")
        train, val, test = stratified_split(df, seed=SEED)
        model  = build_tfidf_pipeline()
        result = run_single_experiment(
            model=model,
            X_train=train["input_text"], y_train=train["label"].values,
            X_val=val["input_text"],     y_val=val["label"].values,
            X_test=test["input_text"],   y_test=test["label"].values,
            experiment_id=f"{condition}_indomain_{name}",
            calibrate=False,
        )
        results_id[name] = result

    # Cross-domain (subset dos experimentos principais)
    priority_cd = [
        (["kaggle_fakenews"], ["liar"],       "kaggle→liar"),
        (["kaggle_fakenews"], ["gossipcop"],  "kaggle→gossipcop"),
        (["gossipcop"],       ["liar"],       "gossipcop→liar"),
        (["liar"],            ["gossipcop"],  "liar→gossipcop"),
    ]

    for train_ds, test_ds, exp_name in priority_cd:
        missing = [d for d in train_ds + test_ds if d not in datasets]
        if missing:
            continue
        logger.info(f"  Cross-domain [{exp_name}]")
        train, val, test = cross_domain_split(
            dfs=datasets,
            train_datasets=train_ds,
            test_datasets=test_ds,
            seed=SEED,
        )
        model  = build_tfidf_pipeline()
        result = run_single_experiment(
            model=model,
            X_train=train["input_text"], y_train=train["label"].values,
            X_val=val["input_text"],     y_val=val["label"].values,
            X_test=test["input_text"],   y_test=test["label"].values,
            experiment_id=f"{condition}_{exp_name}",
            calibrate=False,
        )
        results_cd[exp_name] = result

    return {"indomain": results_id, "crossdomain": results_cd}


def compare_conditions(
    results_orig: Dict,
    results_anon: Dict,
) -> pd.DataFrame:
    """
    Compara métricas entre condição original e anonimizada.
    Calcula delta e valida hipóteses H1-H4.
    """
    rows = []

    metrics = ["macro_f1", "mcc", "roc_auc"]

    for exp_type in ["indomain", "crossdomain"]:
        exp_names = set(results_orig[exp_type].keys()) & set(results_anon[exp_type].keys())

        for exp_name in sorted(exp_names):
            r_orig = results_orig[exp_type][exp_name]
            r_anon = results_anon[exp_type][exp_name]

            row = {
                "experiment": exp_name,
                "type":       exp_type,
            }
            for m in metrics:
                orig_val = r_orig.get(m, float("nan"))
                anon_val = r_anon.get(m, float("nan"))
                delta    = anon_val - orig_val if not (np.isnan(orig_val) or np.isnan(anon_val)) else float("nan")

                row[f"orig_{m}"]  = orig_val
                row[f"anon_{m}"]  = anon_val
                row[f"delta_{m}"] = delta

            rows.append(row)

    return pd.DataFrame(rows)


def plot_anonymization_impact(df_comparison: pd.DataFrame) -> None:
    """
    Fig A1 — Impacto da anonimização por tipo de experimento.
    Mostra delta (anonimizado - original) para cada experimento.
    Verde = anonimização ajuda. Vermelho = prejudica.
    """
    metrics = ["macro_f1", "mcc"]
    metric_labels = ["Macro F1", "MCC"]

    # Separar in-domain e cross-domain
    df_id = df_comparison[df_comparison["type"] == "indomain"]
    df_cd = df_comparison[df_comparison["type"] == "crossdomain"]

    fig, axes = plt.subplots(2, len(metrics), figsize=(5 * len(metrics), 8))

    for row_idx, (df_sub, type_label) in enumerate([(df_id, "In-Domain"), (df_cd, "Cross-Domain")]):
        for col_idx, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]
            delta_col = f"delta_{metric}"

            if delta_col not in df_sub.columns or df_sub.empty:
                ax.set_visible(False)
                continue

            deltas     = df_sub[delta_col].values
            exp_names  = df_sub["experiment"].values
            colors     = [PALETTE["gain"] if d >= 0 else PALETTE["loss"]
                         for d in deltas]

            bars = ax.barh(exp_names, deltas, color=colors,
                          edgecolor="white", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=1.5)

            for bar, val in zip(bars, deltas):
                ha = "left" if val >= 0 else "right"
                offset = 0.002 if val >= 0 else -0.002
                ax.text(
                    val + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}",
                    va="center", ha=ha, fontsize=8
                )

            mean_delta = np.nanmean(deltas)
            ax.set_title(
                f"{type_label} — {m_label}\n"
                f"Média delta={mean_delta:+.4f}",
                fontweight="bold", fontsize=9
            )
            ax.set_xlabel(f"Δ {m_label} (anon − original)")

    fig.suptitle(
        "Ablation Study: Impacto da Anonimização de Entidades\n"
        "Verde = anonimização melhora | Vermelho = piora",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figA1_anonymization_impact.png")
    plt.close(fig)
    logger.info("Fig A1: impacto da anonimização salvo")


def plot_hypothesis_validation(df_comparison: pd.DataFrame) -> None:
    """
    Fig A2 — Validação visual das hipóteses H1 e H2.
    Scatter: delta_indomain vs delta_crossdomain por experimento.
    Quadrante ideal (H1+H2): cima-esquerda (piora in-domain, melhora cross-domain).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    df_id = df_comparison[df_comparison["type"] == "indomain"].rename(
        columns={"delta_macro_f1": "delta_id"}
    )[["experiment", "delta_id"]]

    df_cd = df_comparison[df_comparison["type"] == "crossdomain"].rename(
        columns={"delta_macro_f1": "delta_cd"}
    )

    # Tentar fazer match por dataset de treino
    if df_id.empty or df_cd.empty:
        ax.text(0.5, 0.5, "Dados insuficientes para este plot",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(FIG_DIR / "figA2_hypothesis_validation.png")
        plt.close(fig)
        return

    # Quadrantes
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.fill_between([-1, 0], [0, 0], [1, 1], alpha=0.08, color="#2ECC71",
                    label="H1+H2 confirmadas (ideal)")
    ax.fill_between([0, 1], [-1, -1], [0, 0], alpha=0.08, color="#E74C3C",
                    label="Hipóteses refutadas")

    # Pontos para cross-domain
    for _, row in df_cd.iterrows():
        exp = row["experiment"]
        delta_cd = row.get("delta_cd", 0)
        # Tentar encontrar delta in-domain correspondente
        train_ds = exp.split("→")[0].strip() if "→" in exp else ""
        id_match = df_id[df_id["experiment"] == train_ds]
        delta_id = id_match["delta_id"].values[0] if not id_match.empty else 0

        ax.scatter(delta_id, delta_cd, s=120, zorder=5,
                  edgecolors="black", linewidths=0.8)
        ax.annotate(exp, (delta_id, delta_cd), fontsize=7.5,
                   xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Δ Macro F1 In-Domain (anon − original)", fontsize=11)
    ax.set_ylabel("Δ Macro F1 Cross-Domain (anon − original)", fontsize=11)
    ax.set_title(
        "Validação de Hipóteses: Anonimização\n"
        "H1: piora in-domain | H2: melhora cross-domain",
        fontweight="bold"
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figA2_hypothesis_validation.png")
    plt.close(fig)
    logger.info("Fig A2: validação de hipóteses salva")


def main():
    logger.info("=" * 60)
    logger.info("NEXUS — Ablation Study: Anonimização de Entidades")
    logger.info("=" * 60)

    DATA_DIR = ROOT / "data/processed/unified"

    # ── Condição 1: Dataset original ──────────────────────────────────
    logger.info("\n[1/4] Carregando datasets ORIGINAIS...")
    datasets_orig = load_all_datasets(use_anonymized=False)

    # ── Condição 2: Dataset anonimizado ───────────────────────────────
    logger.info("\n[2/4] Carregando datasets ANONIMIZADOS...")
    datasets_anon = load_all_datasets(use_anonymized=True)

    if not datasets_anon:
        logger.warning(
            "Datasets anonimizados não encontrados. "
            "Execute run_preprocessing.py com spaCy instalado."
        )
        return

    # ── Experimentos ──────────────────────────────────────────────────
    logger.info("\n[3/4] Rodando experimentos...")
    results_orig = run_condition(datasets_orig, "original")
    results_anon = run_condition(datasets_anon, "anonymized")

    # ── Comparação ────────────────────────────────────────────────────
    logger.info("\n[4/4] Comparando condições e gerando figuras...")
    df_comparison = compare_conditions(results_orig, results_anon)

    plot_anonymization_impact(df_comparison)
    plot_hypothesis_validation(df_comparison)

    # ── Salvar resultados ─────────────────────────────────────────────
    df_comparison.to_csv(OUT_DIR / "anonymization_comparison.csv", index=False)

    all_results = {"original": results_orig, "anonymized": results_anon}
    def clean_for_json(d):
        if isinstance(d, dict):
            return {k: clean_for_json(v) for k, v in d.items()
                    if k not in ["y_prob_test", "y_pred_test", "y_true_test", "bin_data"]}
        if isinstance(d, (np.integer, np.floating)):
            return float(d)
        return d

    with open(OUT_DIR / "results_ablation.json", "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2, default=str)

    # ── Resumo das hipóteses ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTADO DO ABLATION — ANONIMIZAÇÃO DE ENTIDADES")
    print("=" * 65)

    df_id = df_comparison[df_comparison["type"] == "indomain"]
    df_cd = df_comparison[df_comparison["type"] == "crossdomain"]

    if not df_id.empty and "delta_macro_f1" in df_id.columns:
        mean_delta_id = df_id["delta_macro_f1"].mean()
        print(f"\nH1 (anonimização reduz in-domain F1):")
        print(f"   Δ médio in-domain  = {mean_delta_id:+.4f} "
              f"{'✓ CONFIRMADA' if mean_delta_id < -0.005 else '✗ NÃO CONFIRMADA'}")

    if not df_cd.empty and "delta_macro_f1" in df_cd.columns:
        mean_delta_cd = df_cd["delta_macro_f1"].mean()
        print(f"\nH2 (anonimização melhora cross-domain F1):")
        print(f"   Δ médio cross-domain = {mean_delta_cd:+.4f} "
              f"{'✓ CONFIRMADA' if mean_delta_cd > 0.005 else '✗ NÃO CONFIRMADA'}")

    print(f"\n📁 Outputs: {OUT_DIR}")
    print(f"   Comparação: {OUT_DIR / 'anonymization_comparison.csv'}")
    print("=" * 65)


if __name__ == "__main__":
    main()