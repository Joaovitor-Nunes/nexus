"""
scripts/consolidate_results.py

Consolida APENAS os experimentos realizados em AMBOS os modelos:
  - TF-IDF + LR  → experiments/results_all.json
  - RoBERTa      → experiments/roberta/*.json

Experimentos pareados (RoBERTa define o conjunto):
  indomain    : kaggle, gossipcop                              (2/4)
  crossdomain : gossipcop→liar, kaggle→gossipcop,
                kaggle→liar, kaggle→politifact                 (4/8)
  lodo        : gossipcop, liar, politifact                   (3/4)

O TF-IDF tem experimentos adicionais (liar, politifact indomain;
gossipcop→kaggle etc.) que NÃO entram na comparação principal.
Eles ficam disponíveis separadamente via load_tfidf_results().

Outputs:
    experiments/final/tables/
        table_indomain_paired.csv
        table_crossdomain_paired.csv
        table_lodo_paired.csv
        table_tfidf_only.csv          ← experimentos sem par RoBERTa
        table_ablation.csv
    experiments/final/figures/
        figC1_indomain_comparison.png
        figC2_generalization_gap.png
        figC3_crossdomain_heatmap.png
        figC4_calibration.png
        figC5_ablation.png
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def _find_project_root() -> Path:
    """
    Sobe a árvore a partir do diretório do script até encontrar
    a pasta 'experiments/' — essa é a raiz do projeto (nexus/).
    Funciona independente de onde o script está salvo.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(6):  # sobe até 6 níveis
        if (candidate / "experiments").is_dir():
            return candidate
        candidate = candidate.parent
    # fallback: assume que o CWD é a raiz do projeto
    cwd = Path.cwd()
    if (cwd / "experiments").is_dir():
        return cwd
    raise RuntimeError(
        "Não foi possível encontrar a raiz do projeto (diretório com 'experiments/').\n"
        f"Script em: {Path(__file__).resolve()}\n"
        f"CWD: {Path.cwd()}\n"
        "Certifique-se de que 'experiments/' existe na raiz do projeto."
    )

ROOT = _find_project_root()
sys.path.insert(0, str(ROOT))
print(f"[INFO] ROOT detectado: {ROOT}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONJUNTOS PAREADOS — definidos pelos 9 arquivos RoBERTa existentes
# Qualquer novo arquivo RoBERTa: adicione aqui e no ROBERTA_FILE_MAP abaixo.
# ─────────────────────────────────────────────────────────────────────────────

PAIRED = {
    "indomain": [
        "kaggle",
        "gossipcop",
    ],
    "crossdomain": [
        "gossipcop→liar",
        "kaggle→gossipcop",
        "kaggle→liar",
        "kaggle→politifact",
    ],
    "lodo": [
        "gossipcop",
        "liar",
        "politifact",
    ],
}

# Mapeamento nome-de-arquivo (sem extensão, sem _results) → (exp_type, chave canônica)
ROBERTA_FILE_MAP: Dict[str, Tuple[str, str]] = {
    # indomain
    "roberta_indomain_kaggle_fakenews":  ("indomain",    "kaggle"),
    "roberta_indomain_gossipcop":        ("indomain",    "gossipcop"),
    "roberta_indomain_liar":             ("indomain",    "liar"),       # caso exista
    "roberta_indomain_politifact":       ("indomain",    "politifact"), # caso exista
    # crossdomain
    "roberta_kaggle_to_liar":            ("crossdomain", "kaggle→liar"),
    "roberta_kaggle_to_gossipcop":       ("crossdomain", "kaggle→gossipcop"),
    "roberta_kaggle_to_politifact":      ("crossdomain", "kaggle→politifact"),
    "roberta_gossipcop_to_liar":         ("crossdomain", "gossipcop→liar"),
    "roberta_gossipcop_to_kaggle":       ("crossdomain", "gossipcop→kaggle"),
    "roberta_liar_to_gossipcop":         ("crossdomain", "liar→gossipcop"),
    "roberta_liar_to_kaggle":            ("crossdomain", "liar→kaggle"),
    # lodo
    "roberta_lodo_test_kaggle_fakenews": ("lodo",        "kaggle"),
    "roberta_lodo_test_gossipcop":       ("lodo",        "gossipcop"),
    "roberta_lodo_test_liar":            ("lodo",        "liar"),
    "roberta_lodo_test_politifact":      ("lodo",        "politifact"),
}

DATASET_CANONICAL = {
    "kaggle_fakenews": "kaggle",
    "kaggle":          "kaggle",
    "liar":            "liar",
    "gossipcop":       "gossipcop",
    "politifact":      "politifact",
}

PALETTE = {
    "tfidf":      "#3498DB",
    "tfidf_only": "#AED6F1",
    "tfidf_anon": "#85C1E9",
    "roberta":    "#E74C3C",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})

OUT_DIR = ROOT / "experiments/final"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in [TAB_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _nan() -> float:
    return float("nan")

def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}" if not np.isnan(v) else "—"

def _delta(a: float, b: float) -> str:
    if np.isnan(a) or np.isnan(b):
        return "—"
    return f"{b - a:+.4f}"

def canon(name: str) -> str:
    return DATASET_CANONICAL.get(name.strip(), name.strip())

def extract_metrics(data: dict) -> dict:
    """
    Extrai métricas de um dicionário de resultado.
    Compatível com results_all.json e formato RoBERTa.
    """
    calib = data.get("calibration", {})
    return {
        "accuracy":   data.get("accuracy",  _nan()),
        "macro_f1":   data.get("macro_f1",  _nan()),
        "f1_fake":    data.get("f1_fake",   _nan()),
        "f1_real":    data.get("f1_real",   _nan()),
        "mcc":        data.get("mcc",       _nan()),
        "roc_auc":    data.get("roc_auc",   _nan()),
        "pr_auc":     data.get("pr_auc",    _nan()),
        "brier":      data.get("brier",     _nan()),
        # ECE pós-calibração — dois formatos possíveis
        "ece": data.get(
            "ece",
            data.get("ece_after_calibration", _nan())
        ),
        "ece_before": calib.get(
            "ece_before",
            data.get("ece_before_calibration", _nan())
        ),
        "n_samples":  data.get("n_samples", data.get("n_test", _nan())),
    }

def filter_paired(results: Dict, exp_type: str) -> Dict:
    """Retorna somente as chaves que estão em PAIRED[exp_type]."""
    keys = set(PAIRED[exp_type])
    return {k: v for k, v in results.get(exp_type, {}).items() if k in keys}

def filter_unpaired(results: Dict, exp_type: str) -> Dict:
    """Retorna somente as chaves que NÃO estão em PAIRED[exp_type]."""
    keys = set(PAIRED[exp_type])
    return {k: v for k, v in results.get(exp_type, {}).items() if k not in keys}


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_tfidf_results() -> Dict:
    """
    Lê experiments/results_all.json.
    Retorna dict com TODOS os experimentos TF-IDF
    (pareados e não-pareados). Filtragem feita depois.
    """
    path = ROOT / "experiments/baseline/results_all.json"
    if not path.exists():
        logger.error(f"Não encontrado: {path}")
        return {"indomain": {}, "crossdomain": {}, "lodo": {}}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results: Dict = {"indomain": {}, "crossdomain": {}, "lodo": {}}

    for ds_key, data in raw.get("indomain", {}).items():
        results["indomain"][canon(ds_key)] = extract_metrics(data)

    for exp_key, data in raw.get("crossdomain", {}).items():
        # chave já vem como "kaggle→liar" etc.
        results["crossdomain"][exp_key] = extract_metrics(data)

    for ds_key, data in raw.get("lodo", {}).items():
        results["lodo"][canon(ds_key)] = extract_metrics(data)

    logger.info(
        f"TF-IDF total: {len(results['indomain'])} indomain | "
        f"{len(results['crossdomain'])} crossdomain | "
        f"{len(results['lodo'])} lodo"
    )
    for exp_type in ("indomain", "crossdomain", "lodo"):
        paired   = set(results[exp_type]) & set(PAIRED[exp_type])
        unpaired = set(results[exp_type]) - set(PAIRED[exp_type])
        logger.info(f"  [{exp_type}] pareados={sorted(paired)} | só-TF-IDF={sorted(unpaired)}")

    return results


def load_ablation_results() -> Dict:
    """Lê experiments/results_ablation.json → original vs anonymized."""
    path = ROOT / "experiments/ablation_anonymization/results_ablation.json"
    if not path.exists():
        logger.warning(f"Não encontrado: {path}")
        return {"original": {}, "anonymized": {}}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results: Dict = {}
    for variant in ("original", "anonymized"):
        results[variant] = {"indomain": {}, "crossdomain": {}}
        vdata = raw.get(variant, {})

        for ds_key, data in vdata.get("indomain", {}).items():
            results[variant]["indomain"][canon(ds_key)] = extract_metrics(data)

        for exp_key, data in vdata.get("crossdomain", {}).items():
            # normaliza unicode arrow
            results[variant]["crossdomain"][
                exp_key.replace("\u2192", "→")
            ] = extract_metrics(data)

        logger.info(
            f"Ablation [{variant}]: "
            f"{len(results[variant]['indomain'])} indomain | "
            f"{len(results[variant]['crossdomain'])} crossdomain"
        )

    return results


def load_roberta_results() -> Dict:
    """
    Lê experiments/roberta/*.json.
    Usa ROBERTA_FILE_MAP para chave canônica.
    Fallback via experiment_id/experiment_type no JSON.
    """
    roberta_dir = ROOT / "experiments/roberta"
    results: Dict = {"indomain": {}, "crossdomain": {}, "lodo": {}}

    if not roberta_dir.exists():
        logger.error(f"Diretório não encontrado: {roberta_dir}")
        return results

    json_files = sorted(roberta_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"Nenhum JSON em {roberta_dir}")
        return results

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Erro lendo {json_path.name}: {e}")
            continue

        stem    = json_path.stem.replace("_results", "")
        mapping = ROBERTA_FILE_MAP.get(stem)

        if mapping:
            exp_type, canon_key = mapping
            results[exp_type][canon_key] = extract_metrics(data)
            logger.info(f"  RoBERTa {exp_type}/{canon_key}  ← {json_path.name}")
            continue

        # fallback via campos do JSON
        exp_type_raw = data.get("experiment_type", "").lower()
        exp_id       = data.get("experiment_id", stem)
        if exp_type_raw not in ("indomain", "crossdomain", "lodo"):
            logger.warning(f"  Tipo desconhecido: {json_path.name}")
            continue

        metrics = extract_metrics(data)

        if exp_type_raw == "indomain":
            ds = canon(exp_id.replace("roberta_indomain_", ""))
            results["indomain"][ds] = metrics
            logger.info(f"  RoBERTa indomain/{ds} [fallback]")

        elif exp_type_raw == "crossdomain":
            name = exp_id.replace("roberta_", "")
            if "_to_" in name:
                train, test = name.split("_to_", 1)
                key = f"{canon(train)}→{canon(test)}"
                results["crossdomain"][key] = metrics
                logger.info(f"  RoBERTa crossdomain/{key} [fallback]")

        elif exp_type_raw == "lodo":
            ds = canon(
                exp_id.replace("roberta_lodo_test_", "")
                       .replace("roberta_lodo_", "")
            )
            results["lodo"][ds] = metrics
            logger.info(f"  RoBERTa lodo/{ds} [fallback]")

    logger.info(
        f"RoBERTa carregado: "
        f"{len(results['indomain'])} indomain | "
        f"{len(results['crossdomain'])} crossdomain | "
        f"{len(results['lodo'])} lodo"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TABELAS
# ─────────────────────────────────────────────────────────────────────────────

SHOW_METRICS = [
    ("macro_f1", "Macro F1"),
    ("mcc",      "MCC"),
    ("roc_auc",  "ROC-AUC"),
    ("brier",    "Brier↓"),
    ("ece",      "ECE↓"),
]
DELTA_METRICS = {"macro_f1", "mcc", "roc_auc"}  # maiores = melhor


def build_paired_table(
    tfidf: Dict,
    roberta: Dict,
    exp_type: str,
) -> pd.DataFrame:
    """
    Tabela comparativa usando SOMENTE experimentos em PAIRED[exp_type].
    Ambos os modelos precisam ter o experimento; caso contrário,
    o valor aparece como '—'.
    """
    keys = PAIRED[exp_type]  # ordem definida em PAIRED
    tf_data  = tfidf.get(exp_type,   {})
    rob_data = roberta.get(exp_type, {})

    rows = []
    for key in keys:
        b = tf_data.get(key,  {})
        r = rob_data.get(key, {})
        row = {"Experimento": key}
        for metric, label in SHOW_METRICS:
            bv = b.get(metric, _nan())
            rv = r.get(metric, _nan())
            row[f"TF-IDF {label}"]  = _fmt(bv)
            row[f"RoBERTa {label}"] = _fmt(rv)
            if metric in DELTA_METRICS:
                row[f"Δ {label}"] = _delta(bv, rv)
        rows.append(row)
    return pd.DataFrame(rows)


def build_tfidf_only_table(tfidf: Dict) -> pd.DataFrame:
    """
    Tabela dos experimentos TF-IDF sem par RoBERTa.
    Serve para documentar o que ficou de fora da comparação.
    """
    rows = []
    for exp_type in ("indomain", "crossdomain", "lodo"):
        unpaired = filter_unpaired(tfidf, exp_type)
        for key, data in sorted(unpaired.items()):
            row = {"Tipo": exp_type, "Experimento": key}
            for metric, label in SHOW_METRICS:
                row[f"TF-IDF {label}"] = _fmt(data.get(metric, _nan()))
            rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(ablation: Dict) -> pd.DataFrame:
    """Original vs Anonymized para indomain + crossdomain."""
    orig = ablation.get("original",   {})
    anon = ablation.get("anonymized", {})
    rows = []
    for exp_type in ("indomain", "crossdomain"):
        o_data = orig.get(exp_type, {})
        a_data = anon.get(exp_type, {})
        for key in sorted(set(o_data) | set(a_data)):
            o = o_data.get(key, {})
            a = a_data.get(key, {})
            row = {"Tipo": exp_type, "Experimento": key}
            for metric, label in SHOW_METRICS[:3]:  # F1, MCC, ROC
                ov = o.get(metric, _nan())
                av = a.get(metric, _nan())
                row[f"Original {label}"]   = _fmt(ov)
                row[f"Anon {label}"]       = _fmt(av)
                row[f"Δ {label}"]          = _delta(ov, av)
            rows.append(row)
    return pd.DataFrame(rows)


def generate_tables(tfidf: Dict, roberta: Dict, ablation: Dict) -> None:
    for exp_type, fname in [
        ("indomain",    "table_indomain_paired.csv"),
        ("crossdomain", "table_crossdomain_paired.csv"),
        ("lodo",        "table_lodo_paired.csv"),
    ]:
        df = build_paired_table(tfidf, roberta, exp_type)
        if not df.empty:
            p = TAB_DIR / fname
            df.to_csv(p, index=False)
            logger.info(f"Tabela: {p.name}")

    df_only = build_tfidf_only_table(tfidf)
    if not df_only.empty:
        p = TAB_DIR / "table_tfidf_only.csv"
        df_only.to_csv(p, index=False)
        logger.info(f"Tabela: {p.name}")

    df_abl = build_ablation_table(ablation)
    if not df_abl.empty:
        p = TAB_DIR / "table_ablation.csv"
        df_abl.to_csv(p, index=False)
        logger.info(f"Tabela: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS  (todas usam filter_paired — nunca dados não-pareados)
# ─────────────────────────────────────────────────────────────────────────────

def _bar_labels(ax, bars, vals, color, fontsize=8):
    for bar, val in zip(bars, vals):
        if not np.isnan(val) and val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=fontsize, color=color, fontweight="bold",
            )


def fig_c1_indomain(tfidf: Dict, roberta: Dict) -> None:
    """Fig C1 — In-domain pareado: TF-IDF+LR vs RoBERTa."""
    keys   = PAIRED["indomain"]
    tf_id  = tfidf.get("indomain",   {})
    rob_id = roberta.get("indomain", {})

    x, width = np.arange(len(keys)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in [
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "mcc",      "MCC"),
    ]:
        bv = [tf_id.get(k,  {}).get(metric, 0) for k in keys]
        rv = [rob_id.get(k, {}).get(metric, 0) for k in keys]

        bars_b = ax.bar(x - width/2, bv, width, label="TF-IDF + LR",
                        color=PALETTE["tfidf"],   alpha=0.85, edgecolor="white")
        bars_r = ax.bar(x + width/2, rv, width, label="RoBERTa",
                        color=PALETTE["roberta"], alpha=0.85, edgecolor="white")
        _bar_labels(ax, bars_b, bv, PALETTE["tfidf"])
        _bar_labels(ax, bars_r, rv, PALETTE["roberta"])

        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=10)
        ax.set_title(f"In-Domain (pareado): {label}", fontweight="bold")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Comparação In-Domain — TF-IDF + LR vs RoBERTa\n"
        f"(datasets comparáveis: {', '.join(keys)})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figC1_indomain_comparison.png")
    plt.close(fig)
    logger.info("Fig C1 salva.")


def fig_c2_generalization_gap(tfidf: Dict, roberta: Dict) -> None:
    """
    Fig C2 — Generalization Gap usando SOMENTE experimentos pareados.
    Média por tipo de avaliação.
    """
    categories = ["In-Domain", "Cross-Domain", "LODO"]
    exp_types  = ["indomain",   "crossdomain",  "lodo"]

    models = {
        "TF-IDF + LR": (tfidf,   PALETTE["tfidf"]),
        "RoBERTa":     (roberta,  PALETTE["roberta"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, label in [
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "mcc",      "MCC"),
    ]:
        x, width = np.arange(len(categories)), 0.35
        model_means: Dict[str, list] = {}

        for mi, (mname, (mdata, color)) in enumerate(models.items()):
            means = []
            for exp_type in exp_types:
                # somente chaves pareadas
                paired_data = filter_paired(mdata, exp_type)
                vals = [
                    v.get(metric, _nan())
                    for v in paired_data.values()
                    if not np.isnan(v.get(metric, _nan()))
                ]
                means.append(np.mean(vals) if vals else _nan())
            model_means[mname] = means

            safe = [m if not np.isnan(m) else 0 for m in means]
            xpos = x + (mi - 0.5) * width
            bars = ax.bar(xpos, safe, width, label=mname,
                          color=color, alpha=0.85, edgecolor="white")
            _bar_labels(ax, bars, safe, color, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(f"Média Pareada por Tipo — {label}", fontweight="bold")
        ax.set_ylabel(f"Média {label}")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

        # Anotações de gap (in-domain → cross-domain)
        for mi, (mname, (_, color)) in enumerate(models.items()):
            m = model_means[mname]
            if not (np.isnan(m[0]) or np.isnan(m[1])):
                gap  = m[0] - m[1]
                xcol = 1 + (mi - 0.5) * width + width / 2
                ax.annotate(
                    f"gap={gap:+.3f}",
                    xy=(xcol, m[1]),
                    xytext=(xcol, max(m[1] - 0.11, 0.02)),
                    fontsize=8, ha="center", color=color,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
                )

    fig.suptitle(
        "Generalization Gap (somente experimentos pareados)\n"
        "TF-IDF + LR vs RoBERTa — Gap menor = melhor generalização",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figC2_generalization_gap.png")
    plt.close(fig)
    logger.info("Fig C2 salva.")


def fig_c3_crossdomain_heatmap(tfidf: Dict, roberta: Dict) -> None:
    """
    Fig C3 — Heatmap cross-domain: apenas os 4 pares comparáveis.
    Datasets do eixo limitados ao universo de PAIRED[crossdomain].
    """
    # universo de datasets que aparecem nos pares comparáveis
    DATASETS = ["kaggle", "gossipcop", "liar", "politifact"]
    n   = len(DATASETS)
    idx = {d: i for i, d in enumerate(DATASETS)}

    def fill_matrix(cd: Dict) -> np.ndarray:
        mat = np.full((n, n), np.nan)
        for exp_key, r in cd.items():
            if "→" not in exp_key:
                continue
            train, test = exp_key.split("→", 1)
            if train in idx and test in idx:
                mat[idx[train]][idx[test]] = r.get("macro_f1", _nan())
        return mat

    # somente pares comparáveis
    tf_cd  = filter_paired(tfidf,   "crossdomain")
    rob_cd = filter_paired(roberta, "crossdomain")

    mat_tf  = fill_matrix(tf_cd)
    mat_rob = fill_matrix(rob_cd)
    mat_d   = mat_rob - mat_tf

    labels = DATASETS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (axes[0], mat_tf,  "TF-IDF + LR — Macro F1",          "RdYlGn",  0,    1,    ".3f"),
        (axes[1], mat_rob, "RoBERTa — Macro F1",               "RdYlGn",  0,    1,    ".3f"),
        (axes[2], mat_d,   "Δ (RoBERTa − TF-IDF)",            "RdBu",   -0.5,  0.5,  "+.3f"),
    ]:
        mask = np.isnan(mat)
        display = np.where(mask, 0.0, mat)
        sns.heatmap(
            display, annot=True, fmt=fmt,
            xticklabels=labels, yticklabels=labels,
            cmap=cmap, vmin=vmin, vmax=vmax,
            mask=mask, ax=ax, linewidths=0.8,
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"label": "Macro F1"},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Teste")
        ax.set_ylabel("Treino")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Cross-Domain Evaluation (4 pares comparáveis)\n"
        "Azul no Δ = RoBERTa generaliza melhor | Vermelho = TF-IDF melhor",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figC3_crossdomain_heatmap.png")
    plt.close(fig)
    logger.info("Fig C3 salva.")


def fig_c4_calibration(tfidf: Dict, roberta: Dict) -> None:
    """Fig C4 — ECE por tipo de avaliação (apenas pareados)."""
    exp_types  = ["indomain", "crossdomain", "lodo"]
    type_label = {"indomain": "In-Domain", "crossdomain": "Cross-Domain", "lodo": "LODO"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, exp_type in zip(axes, exp_types):
        keys     = PAIRED[exp_type]
        tf_data  = tfidf.get(exp_type,   {})
        rob_data = roberta.get(exp_type, {})

        x, width = np.arange(len(keys)), 0.35
        b_ece = [tf_data.get(k,  {}).get("ece", _nan()) for k in keys]
        r_ece = [rob_data.get(k, {}).get("ece", _nan()) for k in keys]

        bars_b = ax.bar(x - width/2,
                        [v if not np.isnan(v) else 0 for v in b_ece],
                        width, label="TF-IDF + LR",
                        color=PALETTE["tfidf"],   alpha=0.85, edgecolor="white")
        bars_r = ax.bar(x + width/2,
                        [v if not np.isnan(v) else 0 for v in r_ece],
                        width, label="RoBERTa",
                        color=PALETTE["roberta"], alpha=0.85, edgecolor="white")

        _bar_labels(ax, bars_b,
                    [v if not np.isnan(v) else 0 for v in b_ece],
                    PALETTE["tfidf"],   fontsize=7)
        _bar_labels(ax, bars_r,
                    [v if not np.isnan(v) else 0 for v in r_ece],
                    PALETTE["roberta"], fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [k.replace("→", "\n→\n") for k in keys], fontsize=8
        )
        ax.set_title(f"{type_label[exp_type]} — ECE ↓", fontweight="bold")
        ax.set_ylabel("ECE (↓ melhor)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Calibração — ECE pós-calibração (apenas experimentos pareados)\n"
        "Barras menores = melhor calibração de probabilidade",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figC4_calibration.png")
    plt.close(fig)
    logger.info("Fig C4 salva.")


def fig_c5_ablation(ablation: Dict) -> None:
    """Fig C5 — Impacto da anonimização: Original vs Anonymized."""
    orig = ablation.get("original",   {})
    anon = ablation.get("anonymized", {})

    exp_types  = [e for e in ("indomain", "crossdomain")
                  if orig.get(e) or anon.get(e)]

    if not exp_types:
        logger.warning("Fig C5: sem dados de ablation.")
        return

    fig, axes = plt.subplots(
        len(exp_types), 2,
        figsize=(13, 5 * len(exp_types)),
        squeeze=False,
    )

    for row_i, exp_type in enumerate(exp_types):
        o_data   = orig.get(exp_type, {})
        a_data   = anon.get(exp_type, {})
        all_keys = sorted(set(o_data) | set(a_data))
        if not all_keys:
            continue

        x, width = np.arange(len(all_keys)), 0.35

        for col_i, (metric, label) in enumerate([("macro_f1", "Macro F1"), ("mcc", "MCC")]):
            ax = axes[row_i][col_i]
            ov = [o_data.get(k, {}).get(metric, 0) for k in all_keys]
            av = [a_data.get(k, {}).get(metric, 0) for k in all_keys]

            bars_o = ax.bar(x - width/2, ov, width, label="Original",
                            color=PALETTE["tfidf"],      alpha=0.85, edgecolor="white")
            bars_a = ax.bar(x + width/2, av, width, label="Anonymized",
                            color=PALETTE["tfidf_anon"], alpha=0.85, edgecolor="white")
            _bar_labels(ax, bars_o, ov, PALETTE["tfidf"],      fontsize=7)
            _bar_labels(ax, bars_a, av, PALETTE["tfidf_anon"], fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [k.replace("→", "\n→\n") for k in all_keys], fontsize=7
            )
            type_label = {"indomain": "In-Domain", "crossdomain": "Cross-Domain"}
            ax.set_title(
                f"{type_label.get(exp_type, exp_type)} — {label}",
                fontweight="bold",
            )
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=8)
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    fig.suptitle(
        "Ablation: Impacto da Anonimização (TF-IDF + LR)\nOriginal vs Anonymized",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figC5_ablation.png")
    plt.close(fig)
    logger.info("Fig C5 salva.")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMO CIENTÍFICO
# ─────────────────────────────────────────────────────────────────────────────

def _paired_mean(
    tfidf: Dict,
    roberta: Dict,
    exp_type: str,
    metric: str = "macro_f1",
) -> Tuple[float, float, int]:
    """
    Média usando SOMENTE os experimentos em PAIRED[exp_type].
    Requer que os dois modelos tenham o experimento; caso contrário ignora.
    """
    keys = PAIRED[exp_type]
    tf_d  = tfidf.get(exp_type,   {})
    rob_d = roberta.get(exp_type, {})
    bvals, rvals = [], []
    for k in keys:
        bv = tf_d.get(k,  {}).get(metric, _nan())
        rv = rob_d.get(k, {}).get(metric, _nan())
        if not (np.isnan(bv) or np.isnan(rv)):
            bvals.append(bv)
            rvals.append(rv)
    return (
        np.mean(bvals) if bvals else _nan(),
        np.mean(rvals) if rvals else _nan(),
        len(bvals),
    )


def print_scientific_summary(tfidf: Dict, roberta: Dict) -> None:
    line = "=" * 70
    print(f"\n{line}")
    print("RESUMO CIENTÍFICO — TF-IDF + LR vs RoBERTa (experimentos pareados)")
    print(line)

    b_id, r_id, n_id = _paired_mean(tfidf, roberta, "indomain")
    b_cd, r_cd, n_cd = _paired_mean(tfidf, roberta, "crossdomain")
    b_ld, r_ld, n_ld = _paired_mean(tfidf, roberta, "lodo")

    print(f"\n{'Conjuntos pareados':}")
    print(f"  In-Domain    : {n_id} datasets  {PAIRED['indomain']}")
    print(f"  Cross-Domain : {n_cd} pares     {PAIRED['crossdomain']}")
    print(f"  LODO         : {n_ld} datasets  {PAIRED['lodo']}")

    def row(label, bv, rv):
        print(
            f"  {label:<14}  TF-IDF={_fmt(bv)}  "
            f"RoBERTa={_fmt(rv)}  Δ={_delta(bv, rv)}"
        )

    print("\nMACRO F1 (média dos experimentos pareados):")
    row("In-Domain",    b_id, r_id)
    row("Cross-Domain", b_cd, r_cd)
    row("LODO",         b_ld, r_ld)

    # OOD = média de cross + lodo
    b_ood_list = [v for v in [b_cd, b_ld] if not np.isnan(v)]
    r_ood_list = [v for v in [r_cd, r_ld] if not np.isnan(v)]
    b_ood = np.mean(b_ood_list) if b_ood_list else _nan()
    r_ood = np.mean(r_ood_list) if r_ood_list else _nan()
    b_gap = b_id - b_ood if not (np.isnan(b_id) or np.isnan(b_ood)) else _nan()
    r_gap = r_id - r_ood if not (np.isnan(r_id) or np.isnan(r_ood)) else _nan()

    print("\nGENERALIZATION GAP (In-Domain − OOD):")
    print(f"  TF-IDF + LR : {_fmt(b_gap)}")
    print(f"  RoBERTa     : {_fmt(r_gap)}")
    if not (np.isnan(b_gap) or np.isnan(r_gap)):
        red = b_gap - r_gap
        print(f"  Redução     : {red:+.4f}")
        msg = (
            "✓ RoBERTa generaliza melhor (gap significativamente menor)."
            if red > 0.05 else
            "~ Leve redução de gap com RoBERTa." if red > 0 else
            "✗ RoBERTa não reduziu o gap de generalização."
        )
        print(f"  → {msg}")

    # Experimentos SÓ no TF-IDF (não entram na comparação)
    print("\nEXPERIMENTOS NÃO COMPARÁVEIS (somente TF-IDF, sem par RoBERTa):")
    for exp_type in ("indomain", "crossdomain", "lodo"):
        only = sorted(filter_unpaired(tfidf, exp_type).keys())
        if only:
            print(f"  {exp_type}: {only}")

    print(f"\nOutputs em: {OUT_DIR}")
    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("NEXUS — Consolidação (apenas experimentos pareados)")
    logger.info("=" * 60)
    logger.info(f"Conjuntos pareados definidos em PAIRED:")
    for et, keys in PAIRED.items():
        logger.info(f"  {et}: {keys}")

    logger.info("\n[1/4] TF-IDF + LR (results_all.json)...")
    tfidf = load_tfidf_results()

    logger.info("\n[2/4] Ablation (results_ablation.json)...")
    ablation = load_ablation_results()

    logger.info("\n[3/4] RoBERTa (experiments/roberta/*.json)...")
    roberta = load_roberta_results()

    # Integrity check: avisa se algum experimento de PAIRED falta em algum modelo
    logger.info("\n[Integrity Check] Verificando cobertura dos pares...")
    all_ok = True
    for exp_type in ("indomain", "crossdomain", "lodo"):
        for key in PAIRED[exp_type]:
            has_tf  = key in tfidf.get(exp_type, {})
            has_rob = key in roberta.get(exp_type, {})
            if not has_tf or not has_rob:
                logger.warning(
                    f"  PAR INCOMPLETO: {exp_type}/{key} — "
                    f"TF-IDF={'OK' if has_tf else 'FALTANDO'} | "
                    f"RoBERTa={'OK' if has_rob else 'FALTANDO'}"
                )
                all_ok = False
    if all_ok:
        logger.info("  Todos os pares completos.")

    logger.info("\n[4/4] Gerando tabelas e figuras...")
    generate_tables(tfidf, roberta, ablation)

    fig_c1_indomain(tfidf, roberta)
    fig_c2_generalization_gap(tfidf, roberta)
    fig_c3_crossdomain_heatmap(tfidf, roberta)
    fig_c4_calibration(tfidf, roberta)
    fig_c5_ablation(ablation)

    print_scientific_summary(tfidf, roberta)

    logger.info(f"\nTabelas : {TAB_DIR}")
    logger.info(f"Figuras : {FIG_DIR}")


if __name__ == "__main__":
    main()