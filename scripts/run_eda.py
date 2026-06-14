"""
scripts/run_eda.py

Análise Exploratória de Dados (EDA) científica completa.

Execução:
    python scripts/run_eda.py

Outputs:
    experiments/eda/figures/      → todos os gráficos PNG
    experiments/eda/report.json   → métricas estruturadas
    experiments/eda/bias_flags.json → alertas de bias detectados

Objetivo científico:
    Não apenas descrever os dados, mas DETECTAR e QUANTIFICAR os
    problemas que invalidam modelos treinados neles:
    - dataset bias estrutural
    - source contamination
    - domain mismatch entre datasets
    - distribuição de comprimentos
    - overlap de vocabulário entre fake/real
    - shortcuts léxicos aprendíveis
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.schema import Article, Label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Estilo visual ─────────────────────────────────────────────────────────────
PALETTE = {"fake": "#E74C3C", "real": "#2ECC71", "neutral": "#3498DB"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})



def infer_domain(article) -> str:
    """
    Heurística simples de domínio baseada em subject/title/text.
    """

    text = (article.text or "").lower()
    subject = (article.subject or "").lower()
    title = (article.title or "").lower()

    combined = f"{subject} {title} {text[:300]}"

    politics_keywords = [
        "election", "president", "senate", "congress", "vote",
        "government", "policy", "minister", "trump", "biden",
        "democrat", "republican"
    ]

    entertainment_keywords = [
        "movie", "actor", "actress", "hollywood", "netflix",
        "music", "song", "album", "celebrity", "film"
    ]

    politics_score = sum(k in combined for k in politics_keywords)
    entertainment_score = sum(k in combined for k in entertainment_keywords)

    if politics_score > entertainment_score and politics_score > 0:
        return "politics"
    elif entertainment_score > politics_score and entertainment_score > 0:
        return "entertainment"
    else:
        return "general"

# ════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE FEATURES PARA EDA
# ════════════════════════════════════════════════════════════════════════════

def extract_eda_features(articles: List[Article]) -> pd.DataFrame:
    """
    Extrai features básicas de cada artigo para análise estatística.
    Não usa modelos — apenas contagens e heurísticas simples.
    """
    rows = []
    for a in articles:
        text = a.text or ""
        words = text.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        unique_words = set(w.lower() for w in words)

        rows.append({
            "id":             a.id,
            "dataset":        a.source_dataset,
            "label":          int(a.label),
            "label_name":     "fake" if a.label == Label.FAKE else "real",
            "label_original": a.label_original,

            # Comprimento
            "word_count":     len(words),
            "char_count":     len(text),
            "sentence_count": len(sentences),
            "avg_word_len":   np.mean([len(w) for w in words]) if words else 0,
            "avg_sent_len":   len(words) / max(len(sentences), 1),

            # Riqueza lexical
            "unique_words":   len(unique_words),
            "type_token_ratio": len(unique_words) / max(len(words), 1),

            # Estilo / sinais de desinformação
            "caps_ratio":     sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "exclaim_count":  text.count("!"),
            "question_count": text.count("?"),
            "exclaim_density": text.count("!") / max(len(sentences), 1),

            # Metadados
            "subject":        a.subject or "unknown",
            "has_title":      int(a.title is not None),
            "domain":        infer_domain(a),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# ANÁLISES INDIVIDUAIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_class_distribution(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Fig 1 — Distribuição de classes por dataset.
    Detecta desbalanceamento e documenta para decisão de weighted loss.
    """
    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]

    results = {}

    for ax, (name, df) in zip(axes, dfs.items()):
        counts = df["label_name"].value_counts()
        total  = len(df)
        imbalance = counts.get("fake", 0) / total

        colors = [PALETTE["fake"], PALETTE["real"]]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)

        for bar, (label, count) in zip(bars, counts.items()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{count:,}\n({count/total:.1%})",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        ax.set_title(f"{name}\n(n={total:,})", fontweight="bold")
        ax.set_ylabel("Contagem" if ax == axes[0] else "")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.tick_params(axis="x", rotation=15)

        results[name] = {
            "total": total,
            "n_fake": int(counts.get("fake", 0)),
            "n_real": int(counts.get("real", 0)),
            "imbalance_ratio": float(imbalance),
            "needs_weighted_loss": abs(imbalance - 0.5) > 0.1,
        }

    fig.suptitle("Distribuição de Classes por Dataset", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig01_class_distribution.png")
    plt.close(fig)
    logger.info("Fig 01: distribuição de classes salva")
    return results


def analyze_text_length(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Fig 2 — Distribuição de comprimento de texto.

    Relevância científica:
    - Determina MAX_LEN adequado para tokenização
    - Detecta se fake e real têm comprimentos sistematicamente diferentes
      (shortcut: modelo pode aprender a classificar por tamanho)
    - Justifica estratégia de truncamento para transformers (512 tokens)
    """
    n_datasets = len(dfs)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    results = {}

    for col, (name, df) in enumerate(dfs.items()):
        fake_df = df[df["label_name"] == "fake"]
        real_df = df[df["label_name"] == "real"]

        # Row 0: histograma de word count
        ax = axes[0, col]
        max_words = df["word_count"].quantile(0.99)
        bins = min(60, int(max_words / 5) + 1)

        ax.hist(fake_df["word_count"].clip(upper=max_words), bins=bins,
                alpha=0.6, color=PALETTE["fake"], label="Fake", density=True)
        ax.hist(real_df["word_count"].clip(upper=max_words), bins=bins,
                alpha=0.6, color=PALETTE["real"], label="Real", density=True)

        ax.axvline(512, color="orange", linestyle="--", linewidth=1.5,
                   label="BERT limit (512 tokens ≈ words)")
        ax.set_title(f"{name}\nDistribuição de Palavras", fontweight="bold")
        ax.set_xlabel("Palavras por texto")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)

        # Row 1: boxplot comparativo
        ax2 = axes[1, col]
        data_to_plot = [
            fake_df["word_count"].clip(upper=max_words).values,
            real_df["word_count"].clip(upper=max_words).values,
        ]
        bp = ax2.boxplot(data_to_plot, labels=["Fake", "Real"],
                         patch_artist=True, notch=True,
                         medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor(PALETTE["fake"] + "99")
        bp["boxes"][1].set_facecolor(PALETTE["real"] + "99")

        # Teste Mann-Whitney (não paramétrico — distribuições geralmente não normais)
        u_stat, p_val = stats.mannwhitneyu(
            fake_df["word_count"], real_df["word_count"], alternative="two-sided"
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax2.set_title(f"Boxplot (Mann-Whitney p{sig})", fontsize=9)
        ax2.set_ylabel("Palavras")

        results[name] = {
            "fake_word_mean":   float(fake_df["word_count"].mean()),
            "fake_word_median": float(fake_df["word_count"].median()),
            "real_word_mean":   float(real_df["word_count"].mean()),
            "real_word_median": float(real_df["word_count"].median()),
            "mannwhitney_p":    float(p_val),
            "length_is_shortcut": p_val < 0.01,  # Diferença significativa = shortcut potencial
            "pct_above_512":    float((df["word_count"] > 512).mean()),
            "recommended_max_len": int(df["word_count"].quantile(0.90)),
        }

    fig.suptitle("Análise de Comprimento de Texto", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig02_text_length.png")
    plt.close(fig)
    logger.info("Fig 02: comprimento de texto salva")
    return results


def analyze_vocabulary_overlap(
    dfs: Dict[str, pd.DataFrame],
    datasets_raw: Dict[str, List[Article]],
    out_dir: Path,
    top_n: int = 5000,
) -> Dict[str, Any]:
    """
    Fig 3 — Overlap de vocabulário entre datasets.

    Relevância científica:
    - Baixo overlap entre train/test datasets = maior challenge de generalização
    - Alto overlap fake/real DENTRO do mesmo dataset = features mais discriminativas
    - Detecta se modelos podem usar vocabulary memorization

    Métrica: Jaccard similarity = |A ∩ B| / |A ∪ B|
    """
    # Construir vocabulários top-N por dataset
    vocabs = {}
    for name, articles in datasets_raw.items():
        all_words = []
        for a in articles:
            all_words.extend(a.text.lower().split())
        counter = Counter(all_words)
        vocabs[name] = set(w for w, _ in counter.most_common(top_n))

    names = list(vocabs.keys())
    n = len(names)
    jaccard_matrix = np.zeros((n, n))

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            inter = len(vocabs[n1] & vocabs[n2])
            union = len(vocabs[n1] | vocabs[n2])
            jaccard_matrix[i, j] = inter / union if union > 0 else 0

    # Fig 3a: heatmap de overlap entre datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    mask = np.zeros_like(jaccard_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        jaccard_matrix, annot=True, fmt=".3f",
        xticklabels=names, yticklabels=names,
        cmap="Blues", ax=ax,
        vmin=0, vmax=1,
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title("Jaccard Similarity de Vocabulário\nentre Datasets", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # Fig 3b: top palavras discriminativas (fake vs real) por dataset
    ax2 = axes[1]
    ax2.axis("off")

    text_lines = ["TOP PALAVRAS DISCRIMINATIVAS (Fake - Real)\n"]
    for name, df in list(dfs.items())[:2]:  # Máximo 2 datasets para caber
        articles = datasets_raw[name]
        fake_words = Counter()
        real_words = Counter()
        for a in articles:
            words = a.text.lower().split()
            if a.label == Label.FAKE:
                fake_words.update(words)
            else:
                real_words.update(words)

        # Palavras com maior razão fake/real (mínimo 10 ocorrências)
        all_words_set = set(list(fake_words.keys())[:2000])
        ratios = {}
        for w in all_words_set:
            f = fake_words.get(w, 0)
            r = real_words.get(w, 0)
            if f + r >= 10:
                # Log-odds ratio
                ratios[w] = np.log((f + 1) / (r + 1))

        top_fake = sorted(ratios, key=lambda x: -ratios[x])[:8]
        top_real = sorted(ratios, key=lambda x:  ratios[x])[:8]

        text_lines.append(f"━━ {name} ━━")
        text_lines.append(f"  Fake: {', '.join(top_fake)}")
        text_lines.append(f"  Real: {', '.join(top_real)}")
        text_lines.append("")

    ax2.text(
        0.05, 0.95, "\n".join(text_lines),
        transform=ax2.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "#F8F9FA", "alpha": 0.8}
    )

    plt.tight_layout()
    fig.savefig(out_dir / "fig03_vocabulary_overlap.png")
    plt.close(fig)

    results = {
        "jaccard_matrix": jaccard_matrix.tolist(),
        "dataset_names": names,
        "cross_domain_challenge": {
            f"{n1}_vs_{n2}": float(jaccard_matrix[i][j])
            for i, n1 in enumerate(names)
            for j, n2 in enumerate(names)
            if i < j
        }
    }
    logger.info("Fig 03: overlap de vocabulário salva")
    return results


def analyze_stylometric_signals(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Fig 4 — Sinais estilométricos: shortcuts aprendíveis.

    Relevância científica crítica:
    Demonstra que fake e real têm propriedades estilísticas diferentes
    que um modelo pode usar como SHORTCUTS em vez de aprender factualidade.
    Esses sinais devem ser documentados, não eliminados — eles são parte
    da análise de "o que o modelo aprende vs o que deveria aprender".
    """
    features = ["caps_ratio", "exclaim_density", "type_token_ratio", "avg_word_len"]
    feature_labels = ["Razão de Maiúsculas", "Densidade de !", "Type-Token Ratio", "Comprimento Médio de Palavra"]

    n_datasets = len(dfs)
    fig, axes = plt.subplots(len(features), n_datasets,
                             figsize=(4.5 * n_datasets, 3.5 * len(features)))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    results = {}

    for col, (name, df) in enumerate(dfs.items()):
        results[name] = {}
        fake_df = df[df["label_name"] == "fake"]
        real_df = df[df["label_name"] == "real"]

        for row, (feat, feat_label) in enumerate(zip(features, feature_labels)):
            ax = axes[row, col]

            fake_vals = fake_df[feat].dropna().values
            real_vals = real_df[feat].dropna().values

            # KDE plot
            try:
                from scipy.stats import gaussian_kde
                for vals, color, lbl in [
                    (fake_vals, PALETTE["fake"], "Fake"),
                    (real_vals, PALETTE["real"], "Real"),
                ]:
                    if len(vals) > 10:
                        kde = gaussian_kde(vals, bw_method=0.3)
                        x_range = np.linspace(
                            min(vals.min(), real_vals.min() if lbl == "Fake" else fake_vals.min()),
                            max(vals.max(), real_vals.max() if lbl == "Fake" else fake_vals.max()),
                            200
                        )
                        ax.fill_between(x_range, kde(x_range), alpha=0.4, color=color, label=lbl)
                        ax.plot(x_range, kde(x_range), color=color, linewidth=1.5)
            except Exception:
                ax.hist(fake_vals, alpha=0.5, color=PALETTE["fake"], label="Fake", density=True, bins=20)
                ax.hist(real_vals, alpha=0.5, color=PALETTE["real"], label="Real", density=True, bins=20)

            # Teste estatístico
            _, p_val = stats.mannwhitneyu(fake_vals, real_vals, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            title = f"{name if row == 0 else ''}\n{feat_label} (p{sig})"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Densidade" if col == 0 else "")

            if row == 0 and col == 0:
                ax.legend(fontsize=7)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((fake_vals.std()**2 + real_vals.std()**2) / 2)
            cohens_d = (fake_vals.mean() - real_vals.mean()) / (pooled_std + 1e-10)

            results[name][feat] = {
                "fake_mean":  float(fake_vals.mean()),
                "real_mean":  float(real_vals.mean()),
                "mannwhitney_p": float(p_val),
                "cohens_d":   float(cohens_d),
                "is_shortcut": p_val < 0.01 and abs(cohens_d) > 0.2,
            }

    fig.suptitle("Sinais Estilométricos: Fake vs Real\n(Potenciais Shortcuts de Aprendizado)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig04_stylometric_signals.png")
    plt.close(fig)
    logger.info("Fig 04: sinais estilométricos salva")
    return results


def analyze_domain_shift(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Fig 5 — Domain shift entre datasets.

    Visualiza quão diferentes são os datasets em termos de features
    básicas — quanto maior a diferença, maior o desafio de cross-domain
    generalization e mais informativo o cross-domain evaluation.
    """
    features = ["word_count", "type_token_ratio", "caps_ratio", "exclaim_density", "avg_word_len"]

    # Normalizar features para comparação
    all_data = pd.concat(dfs.values(), ignore_index=True)
    normalized = {}

    for name, df in dfs.items():
        row = {}
        for feat in features:
            col_mean = all_data[feat].mean()
            col_std  = all_data[feat].std() + 1e-10
            row[feat] = float((df[feat].mean() - col_mean) / col_std)
        normalized[name] = row

    norm_df = pd.DataFrame(normalized).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap de features normalizadas
    ax = axes[0]
    feature_labels = ["Word Count", "Type-Token Ratio", "Caps Ratio", "Exclaim Density", "Avg Word Len"]
    sns.heatmap(
        norm_df.values,
        annot=True, fmt=".2f",
        xticklabels=feature_labels,
        yticklabels=norm_df.index,
        cmap="RdBu_r", center=0,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Domain Shift: Features Normalizadas\n(z-score relativo ao conjunto total)",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=35)

    # Radar chart dos datasets
    ax2 = axes[1]
    ax2.axis("off")

    # Tabela de domain shift distances
    names = list(normalized.keys())
    n = len(names)
    dist_matrix = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            v1 = np.array(list(normalized[n1].values()))
            v2 = np.array(list(normalized[n2].values()))
            dist_matrix[i, j] = np.linalg.norm(v1 - v2)

    rows_text = ["DOMAIN SHIFT DISTANCES (Euclidean)\n"]
    rows_text.append(f"{'':20s}" + " ".join(f"{n:12s}" for n in names))
    for i, n1 in enumerate(names):
        row_str = f"{n1:20s}" + " ".join(f"{dist_matrix[i,j]:12.3f}" for j in range(n))
        rows_text.append(row_str)

    rows_text.append("\n⚠ Alta distância = maior desafio de cross-domain")
    rows_text.append("   LIAR vs Kaggle: statements vs artigos (máximo esperado)")

    ax2.text(
        0.05, 0.95, "\n".join(rows_text),
        transform=ax2.transAxes,
        fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "#FFF9E6", "alpha": 0.9},
    )

    plt.tight_layout()
    fig.savefig(out_dir / "fig05_domain_shift.png")
    plt.close(fig)

    results = {
        "normalized_features": normalized,
        "pairwise_distances": {
            f"{names[i]}_vs_{names[j]}": float(dist_matrix[i][j])
            for i in range(n) for j in range(n) if i < j
        }
    }
    logger.info("Fig 05: domain shift salva")
    return results


def analyze_liar_label_distribution(
    liar_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Fig 6 — Distribuição original do LIAR (6 classes).

    Relevância: demonstra a perda de informação da binarização.
    Documenta quantos exemplos ambíguos foram descartados e justifica
    a decisão metodológica no TCC.
    """
    if "label_original" not in liar_df.columns:
        logger.warning("LIAR: label_original não disponível")
        return {}

    order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    colors_6 = ["#C0392B", "#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#1ABC9C"]

    counts = liar_df["label_original"].value_counts()
    counts = counts.reindex([o for o in order if o in counts.index]).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Barplot das 6 classes
    ax = axes[0]
    bars = ax.bar(
        range(len(counts)), counts.values,
        color=[colors_6[order.index(l)] for l in counts.index],
        edgecolor="white", linewidth=1.5
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_title("LIAR: Distribuição Original (6 Classes)", fontweight="bold")
    ax.set_ylabel("Contagem")

    for bar, (_, count) in zip(bars, counts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Visualização da binarização
    ax2 = axes[1]
    binary_groups = {
        "FAKE\n(pants-fire + false\n+ barely-true)": counts.get("pants-fire", 0) + counts.get("false", 0) + counts.get("barely-true", 0),
        "REAL\n(half-true + mostly-true\n+ true)":    counts.get("half-true", 0)  + counts.get("mostly-true", 0) + counts.get("true", 0),
    }

    bars2 = ax2.bar(
        list(binary_groups.keys()),
        list(binary_groups.values()),
        color=[PALETTE["fake"], PALETTE["real"]],
        edgecolor="white", linewidth=1.5, width=0.5
    )
    ax2.set_title("LIAR: Após Binarização\n(perda de gradação de veracidade)", fontweight="bold")
    ax2.set_ylabel("Contagem")

    total_bin = sum(binary_groups.values()) or 1
    for bar, (_, count) in zip(bars2, binary_groups.items()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_bin * 0.01,
            f"{count:,}\n({count/total_bin:.1%})",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    fig.savefig(out_dir / "fig06_liar_label_distribution.png")
    plt.close(fig)
    logger.info("Fig 06: distribuição LIAR salva")

    return {
        "original_distribution": counts.to_dict(),
        "binarized": binary_groups,
        "information_loss_note": "barely-true e half-true são semanticamente ambíguos"
    }


def generate_bias_report(
    class_results: Dict,
    length_results: Dict,
    stylo_results: Dict,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Compila flags de bias detectados automaticamente.
    Cada flag é uma evidência de problema metodológico documentável no TCC.
    """
    bias_flags = {}

    for dataset_name in class_results:
        flags = []

        # Desbalanceamento de classes
        cr = class_results[dataset_name]
        if cr.get("needs_weighted_loss"):
            imb = cr["imbalance_ratio"]
            flags.append({
                "type": "CLASS_IMBALANCE",
                "severity": "HIGH" if abs(imb - 0.5) > 0.2 else "MEDIUM",
                "detail": f"Razão fake/total = {imb:.3f}. Usar weighted loss ou oversampling.",
                "recommendation": "weighted_loss or SMOTE"
            })

        # Comprimento como shortcut
        if dataset_name in length_results:
            lr = length_results[dataset_name]
            if lr.get("length_is_shortcut"):
                flags.append({
                    "type": "LENGTH_SHORTCUT",
                    "severity": "MEDIUM",
                    "detail": f"Comprimento difere significativamente (p<0.01) entre fake/real. "
                              f"Fake médio: {lr['fake_word_mean']:.0f}w, Real médio: {lr['real_word_mean']:.0f}w",
                    "recommendation": "Analisar se modelo aprende comprimento como feature"
                })

            pct_trunc = lr.get("pct_above_512", 0)
            if pct_trunc > 0.3:
                flags.append({
                    "type": "HIGH_TRUNCATION_RATE",
                    "severity": "HIGH",
                    "detail": f"{pct_trunc:.1%} dos textos excedem 512 tokens. "
                              f"Truncamento naive perde informação crítica.",
                    "recommendation": "Usar estratégia início+fim ou chunking hierárquico"
                })

        # Shortcuts estilométricos
        if dataset_name in stylo_results:
            sr = stylo_results[dataset_name]
            shortcuts = [f for f, vals in sr.items() if vals.get("is_shortcut")]
            if shortcuts:
                flags.append({
                    "type": "STYLOMETRIC_SHORTCUTS",
                    "severity": "HIGH",
                    "detail": f"Features estilométricas discriminativas detectadas: {shortcuts}. "
                              f"Modelo pode aprender estilo em vez de factualidade.",
                    "recommendation": "Documentar como confound; testar modelo sem essas features"
                })

        bias_flags[dataset_name] = flags

    # Salvar relatório
    report = {"bias_flags": bias_flags, "total_flags": sum(len(v) for v in bias_flags.values())}
    with open(out_dir.parent / "bias_flags.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Bias report: {report['total_flags']} flags detectados")
    return report

def analyze_domain_distribution(dfs, out_dir):
    """
    Analisa distribuição de domínios por dataset e por classe.
    """

    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]

    results = {}

    for ax, (name, df) in zip(axes, dfs.items()):

        dist = pd.crosstab(df["domain"], df["label_name"], normalize="index")

        dist.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[PALETTE["fake"], PALETTE["real"]]
        )

        ax.set_title(f"{name} — Domain vs Label")
        ax.set_ylabel("Proporção")
        ax.legend(["fake", "real"])

        results[name] = dist.to_dict()

    fig.suptitle("Distribuição de Domínio por Dataset", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig08_domain_distribution.png")
    plt.close(fig)

    return results

def analyze_topic_leakage_bias(dfs, out_dir):
    """
    Detecta se o label pode ser inferido pelo domínio (topic leakage).
    """

    results = {}

    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items()):

        # probabilidade de fake por domínio
        leak = df.groupby("domain")["label"].mean()

        ax.bar(leak.index, leak.values, color="#8E44AD")

        ax.axhline(0.5, linestyle="--", color="black", linewidth=1)

        ax.set_title(f"{name} — Topic Leakage")
        ax.set_ylabel("P(fake | domain)")

        results[name] = {
            d: float(v) for d, v in leak.items()
        }

        # interpretação automática
        bias = any(abs(v - 0.5) > 0.15 for v in leak.values)

        results[name]["leakage_detected"] = bias

    fig.suptitle("Topic Leakage Bias Analysis", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig09_topic_leakage.png")
    plt.close(fig)

    return results

def generate_summary_table(
    datasets_raw: Dict[str, List[Article]],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Fig 7 — Tabela resumo de todos os datasets.
    Formatada para inclusão direta no TCC.
    """
    rows = []
    for name, articles in datasets_raw.items():
        n_fake = sum(1 for a in articles if a.label == Label.FAKE)
        n_real = len(articles) - n_fake
        lengths = [a.word_count() for a in articles]
        has_title = sum(1 for a in articles if a.title)

        rows.append({
            "Dataset":        name,
            "Total":          len(articles),
            "Fake":           n_fake,
            "Real":           n_real,
            "Bal. (fake%)":   f"{n_fake/len(articles):.1%}",
            "Avg Words":      f"{np.mean(lengths):.0f}",
            "Median Words":   f"{np.median(lengths):.0f}",
            "Max Words":      f"{np.max(lengths):.0f}",
            "Has Title":      f"{has_title/len(articles):.0%}",
            "Domain":         {
                "kaggle_fakenews": "Notícias (Reuters vs desinform.)",
                "liar":            "Statements políticos",
                "politifact":      "Fact-checking político",
                "gossipcop":       "Fact-checking de notícias"
            }.get(name, "N/A"),
        })

    summary_df = pd.DataFrame(rows)

    # Figura da tabela
    fig, ax = plt.subplots(figsize=(16, max(3, len(rows) * 0.8 + 1.5)))
    ax.axis("off")

    col_widths = [0.13, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.07, 0.07, 0.22]
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.8)

    # Estilo do header
    for j in range(len(summary_df.columns)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Linhas alternadas
    for i in range(1, len(rows) + 1):
        color = "#F8F9FA" if i % 2 == 0 else "white"
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(color)

    ax.set_title("Tabela 1 — Estatísticas dos Datasets\n(para inclusão no TCC)",
                 fontweight="bold", fontsize=12, pad=20)

    fig.savefig(out_dir / "fig07_dataset_summary_table.png")
    plt.close(fig)

    # Salvar também como CSV
    csv_out = ROOT / "data" / "processed" / "unified" / "dataset_stats.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_out, index=False)
    logger.info("Fig 07: tabela resumo salva")
    return summary_df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("NEXUS EDA — Análise Exploratória Científica")
    logger.info("=" * 60)

    # Diretórios de output
    out_dir = ROOT / "experiments" / "eda"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Carregar dados ────────────────────────────────────────────────────
    logger.info("\n[1/7] Carregando datasets...")

    # PRODUÇÃO: substituir por loaders reais
    # from src.data.loaders import load_dataset
    # datasets_raw = {
    #     "kaggle_fakenews": load_dataset("kaggle_fakenews", ROOT / "data/raw/kaggle_fakenews"),
    #     "liar":            load_dataset("liar",            ROOT / "data/raw/liar"),
    #     "politifact":      load_dataset("politifact",      ROOT / "data/raw/politifact"),
    # }
    from src.data.loaders import load_dataset

    datasets_raw = {
        "kaggle_fakenews": load_dataset("kaggle_fakenews", ROOT / "data/raw/kaggle_fakenews"),
        "liar":            load_dataset("liar",            ROOT / "data/raw/liar"),
        "politifact":      load_dataset("politifact",      ROOT / "data/raw/politifact"),
        "gossipcop":       load_dataset("gossipcop",       ROOT / "data/raw/gossipcop"),
    }

    # ── 2. Extrair features para EDA ────────────────────────────────────────
    logger.info("\n[2/7] Extraindo features...")
    dfs = {name: extract_eda_features(articles) for name, articles in datasets_raw.items()}

    # ── 3. Análises ─────────────────────────────────────────────────────────
    logger.info("\n[3/7] Análise de distribuição de classes...")
    class_results = analyze_class_distribution(dfs, fig_dir)

    logger.info("\n[4/7] Análise de comprimento de texto...")
    length_results = analyze_text_length(dfs, fig_dir)

    logger.info("\n[5/7] Análise de overlap de vocabulário...")
    vocab_results = analyze_vocabulary_overlap(dfs, datasets_raw, fig_dir)

    logger.info("\n[6/7] Análise de sinais estilométricos...")
    stylo_results = analyze_stylometric_signals(dfs, fig_dir)

    logger.info("\n[6b/7] Análise de domain shift...")
    domain_dist_results = analyze_domain_distribution(dfs, fig_dir)
    
    logger.info("\n[6c/7] Análise de domínio e topic leakage...")
    topic_bias_results = analyze_topic_leakage_bias(dfs, fig_dir)

    logger.info("\n[6b/7] Análise de domain shift...")
    domain_results = analyze_domain_shift(dfs, fig_dir)

    if "liar" in dfs:
        logger.info("\n[6c/7] Análise especial LIAR...")
        liar_results = analyze_liar_label_distribution(dfs["liar"], fig_dir)

    logger.info("\n[7/7] Gerando tabela resumo...")
    summary_df = generate_summary_table(datasets_raw, fig_dir)

    # ── 4. Relatório de bias ─────────────────────────────────────────────────
    bias_report = generate_bias_report(class_results, length_results, stylo_results, out_dir)

    # ── 5. Salvar report.json ────────────────────────────────────────────────
    report = {
        "class_distribution":  class_results,
        "text_length":         length_results,
        "vocabulary_overlap":  vocab_results,
        "stylometric_signals": stylo_results,
        "domain_shift":        domain_results,
        "domain_distribution": domain_dist_results,
        "topic_leakage":       topic_bias_results,
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # ── 6. Imprimir resumo científico ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMO CIENTÍFICO — ACHADOS DO EDA")
    print("=" * 60)

    print("\n📊 DATASETS:")
    for name, articles in datasets_raw.items():
        n_f = sum(1 for a in articles if a.label == Label.FAKE)
        print(f"  {name:20s}: {len(articles):,} artigos | fake={n_f/len(articles):.1%}")

    print("\n⚠  SHORTCUTS DETECTADOS:")
    for ds, flags in bias_report["bias_flags"].items():
        if flags:
            print(f"  [{ds}]")
            for flag in flags:
                print(f"    [{flag['severity']}] {flag['type']}: {flag['detail'][:80]}...")

    print("\n📐 DOMAIN SHIFT (Jaccard vocabulary similarity):")
    for pair, sim in vocab_results.get("cross_domain_challenge", {}).items():
        challenge = "ALTO" if sim < 0.3 else "MÉDIO" if sim < 0.5 else "BAIXO"
        print(f"  {pair:35s}: {sim:.3f} (desafio {challenge})")

    print("\n✅ RECOMENDAÇÕES PARA EXPERIMENTOS:")
    print("  2. Usar weighted loss para datasets desbalanceados")
    print("  3. Implementar estratégia início+fim para textos >512 tokens")
    print("  4. Features estilométricas são confounds — documentar no TCC")
    print("  5. Executar ablation: com e sem anonimização de entidades")

    print(f"\n📁 Outputs salvos em: {out_dir}")
    print(f"   Figuras: {fig_dir}")
    print(f"   Relatório: {out_dir / 'report.json'}")
    print(f"   Bias flags: {out_dir / 'bias_flags.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()