"""
src/evaluation/calibration.py

Calibração de modelos: Platt Scaling e análise de confiabilidade.

Por que calibração importa para fake news detection?
  Um modelo não calibrado que diz "95% de confiança que é fake"
  pode estar correto apenas 60% das vezes nessa faixa. Em um
  sistema de verificação de fatos, isso é perigoso — o usuário
  confia em uma probabilidade que não reflete a realidade.

  Referência: Guo et al. (2017) mostram que redes neurais modernas
  são sistematicamente overconfident. TF-IDF + LR também pode ser,
  especialmente quando treinado em datasets com shortcut features.

Métodos:
  - Platt Scaling: calibração logística post-hoc (Platt, 1999)
    Ajusta uma regressão logística sobre os scores brutos do modelo.
    Simples e eficaz para classificação binária.

  - Isotonic Regression: alternativa não-paramétrica
    Mais flexível que Platt, mas requer mais dados de calibração.
    Risco de overfitting em datasets pequenos.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class PlattScaler:
    """
    Platt Scaling: calibração post-hoc via regressão logística
    ajustada sobre os scores/probabilidades do modelo original.

    Uso:
        # Após treinar o modelo principal
        scaler = PlattScaler()
        scaler.fit(y_cal_true, y_cal_prob_uncalibrated)

        # Na inferência
        y_prob_calibrated = scaler.transform(y_test_prob_uncalibrated)

    IMPORTANTE: y_cal deve ser um conjunto SEPARADO do conjunto de treino.
    Usar os dados de validação para calibração é metodologicamente correto.
    Usar os dados de treino para calibração cria overfit na calibração.
    """

    def __init__(self):
        self._scaler = None
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "PlattScaler":
        """
        Ajusta o calibrador no conjunto de validação.

        y_true: labels verdadeiros (0 ou 1)
        y_prob: probabilidades brutas do modelo (não calibradas)
        """
        from sklearn.linear_model import LogisticRegression

        y_true = np.asarray(y_true).reshape(-1)
        y_prob = np.asarray(y_prob).reshape(-1, 1)

        # Logistic Regression sobre as probabilidades brutas
        # C grande = pouca regularização (Platt scaling padrão)
        self._scaler = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._scaler.fit(y_prob, y_true)
        self._fitted = True

        logger.info(
            f"PlattScaler ajustado: "
            f"coef={self._scaler.coef_[0][0]:.4f}, "
            f"intercept={self._scaler.intercept_[0]:.4f}"
        )
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Aplica calibração às probabilidades."""
        if not self._fitted:
            raise RuntimeError("PlattScaler não foi ajustado. Chame .fit() primeiro.")
        y_prob = np.asarray(y_prob).reshape(-1, 1)
        return self._scaler.predict_proba(y_prob)[:, 1]

    def fit_transform(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """Ajusta e transforma em uma operação."""
        self.fit(y_true, y_prob)
        return self.transform(y_prob)


def compare_calibration(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
) -> Dict:
    """
    Compara ECE e Brier Score antes e depois da calibração.
    Resultado para tabela do TCC.
    """
    from .metrics import compute_ece
    from sklearn.metrics import brier_score_loss

    ece_raw,  _ = compute_ece(y_true, y_prob_raw)
    ece_cal,  _ = compute_ece(y_true, y_prob_calibrated)
    brier_raw   = brier_score_loss(y_true, y_prob_raw)
    brier_cal   = brier_score_loss(y_true, y_prob_calibrated)

    return {
        "ece_before":    ece_raw,
        "ece_after":     ece_cal,
        "ece_reduction": ece_raw - ece_cal,
        "brier_before":  brier_raw,
        "brier_after":   brier_cal,
        "calibration_improved": ece_cal < ece_raw,
    }