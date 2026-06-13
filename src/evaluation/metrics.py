"""
src/evaluation/metrics.py

Métricas científicas para avaliação rigorosa de modelos de fake news.

Filosofia: accuracy é insuficiente e potencialmente enganosa.
  - PolitiFact 83% fake: accuracy de 83% é trivial (predizer sempre FAKE)
  - Datasets desbalanceados: F1 Macro e MCC são as métricas primárias
  - Calibração: ECE e Brier Score avaliam se as probabilidades são confiáveis
  - MCC: única métrica que considera todos os quadrantes da matriz de confusão

Referências:
  - Chicco & Jurman (2020): "The advantages of the Matthews correlation
    coefficient (MCC) over F1 score and accuracy in binary classification"
  - Guo et al. (2017): "On Calibration of Modern Neural Networks"
  - Ferro et al. (2019): "CLEF 2019 CheckThat! Lab"
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)


# ── Métricas de classificação ─────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    dataset_name: str = "",
) -> Dict[str, float]:
    """
    Computa conjunto completo de métricas de classificação.

    Parâmetros:
        y_true: labels verdadeiros (0 ou 1)
        y_pred: predições binárias
        y_prob: probabilidades da classe positiva (REAL=1)
                Necessário para AUC-ROC, AUC-PR, Brier Score
        dataset_name: para logging

    Retorna dict com todas as métricas — formato pronto para
    inserção em tabela do TCC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "dataset":   dataset_name,
        "n_samples": len(y_true),
        "n_fake":    int((y_true == 0).sum()),
        "n_real":    int((y_true == 1).sum()),

        # Métricas primárias
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc":        float(matthews_corrcoef(y_true, y_pred)),

        # Por classe
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_fake":    float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fake":        float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_real":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_real":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    # Métricas baseadas em probabilidade (requerem y_prob)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["roc_auc"]  = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"]   = float(average_precision_score(y_true, y_prob))
            metrics["brier"]    = float(brier_score_loss(y_true, y_prob))
        except ValueError as e:
            # Pode ocorrer se y_true tem apenas uma classe (dataset muito pequeno)
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"]  = float("nan")
            metrics["brier"]   = float("nan")

    return metrics


def compute_confusion_matrix_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Retorna TN, FP, FN, TP para análise detalhada de erros."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# ── Métricas de calibração ────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict]:
    """
    Expected Calibration Error (ECE).

    Interpretação: um modelo bem calibrado com probabilidade 0.7
    deve estar correto ~70% do tempo. ECE mede o desvio médio.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    onde B_b é o bin b, acc é accuracy no bin, conf é confiança média.

    ECE < 0.05: boa calibração
    ECE > 0.10: calibração problemática (comum em transformers não calibrados)

    Retorna:
        ece: float (valor escalar)
        bin_data: dict com detalhes por bin (para Reliability Diagram)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers     = bin_boundaries[:-1]
    bin_uppers     = bin_boundaries[1:]

    ece = 0.0
    bin_data = {
        "bin_centers":     [],
        "bin_accuracies":  [],
        "bin_confidences": [],
        "bin_counts":      [],
    }

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (y_prob >= lower) & (y_prob < upper)
        if mask.sum() == 0:
            continue

        bin_acc   = float(y_true[mask].mean())
        bin_conf  = float(y_prob[mask].mean())
        bin_count = int(mask.sum())

        ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

        bin_data["bin_centers"].append(float((lower + upper) / 2))
        bin_data["bin_accuracies"].append(bin_acc)
        bin_data["bin_confidences"].append(bin_conf)
        bin_data["bin_counts"].append(bin_count)

    return float(ece), bin_data


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Métricas completas de calibração.

    Retorna ECE, Brier Score e dados para Reliability Diagram.
    """
    ece, bin_data = compute_ece(y_true, y_prob, n_bins=n_bins)
    brier         = float(brier_score_loss(y_true, y_prob))

    # Overconfidence: modelo diz 0.9+ mas acerta <90%
    high_conf_mask = y_prob >= 0.9
    if high_conf_mask.sum() > 0:
        overconf_acc  = float(y_true[high_conf_mask].mean())
        overconf_rate = float(high_conf_mask.mean())
    else:
        overconf_acc  = float("nan")
        overconf_rate = 0.0

    return {
        "ece":            ece,
        "brier":          brier,
        "overconf_rate":  overconf_rate,   # % de predições com conf >= 0.9
        "overconf_acc":   overconf_acc,    # accuracy entre essas predições
        "bin_data":       bin_data,
    }


# ── Formatação para TCC ────────────────────────────────────────────────────

def format_metrics_table_row(metrics: Dict, include_calibration: bool = True) -> Dict:
    """
    Formata métricas para uma linha da tabela do TCC.
    Arredonda para 4 casas decimais e formata percentuais.
    """
    row = {
        "Dataset":    metrics.get("dataset", ""),
        "N":          metrics.get("n_samples", ""),
        "Accuracy":   f"{metrics.get('accuracy', 0):.4f}",
        "Macro F1":   f"{metrics.get('macro_f1', 0):.4f}",
        "MCC":        f"{metrics.get('mcc', 0):.4f}",
        "ROC-AUC":    f"{metrics.get('roc_auc', float('nan')):.4f}",
        "PR-AUC":     f"{metrics.get('pr_auc', float('nan')):.4f}",
        "F1 (Fake)":  f"{metrics.get('f1_fake', 0):.4f}",
        "F1 (Real)":  f"{metrics.get('f1_real', 0):.4f}",
    }
    if include_calibration:
        row["ECE"]   = f"{metrics.get('ece', float('nan')):.4f}"
        row["Brier"] = f"{metrics.get('brier', float('nan')):.4f}"
    return row


def generalization_gap(
    in_domain_metrics: Dict,
    cross_domain_metrics: Dict,
    metric: str = "macro_f1",
) -> float:
    """
    Calcula o generalization gap entre avaliação in-domain e cross-domain.

    gap = F1_in_domain - F1_cross_domain

    Interpretação:
        gap < 0.05: boa generalização
        gap 0.05-0.15: generalização moderada
        gap > 0.15: overfitting ao domínio de treino (shortcut learning)
        gap > 0.30: falha severa de generalização

    Este é o número mais importante do projeto.
    """
    id_val  = in_domain_metrics.get(metric, 0)
    cd_val  = cross_domain_metrics.get(metric, 0)
    return float(id_val - cd_val)