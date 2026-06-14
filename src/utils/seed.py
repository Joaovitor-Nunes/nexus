# src/utils/seed.py

import os
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Reprodutibilidade global.
    deterministic=True: necessário para pesquisa, reduz ~10% de performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:
        pass

    logger.info(f"Seed global configurado: {seed} | deterministic={deterministic}")