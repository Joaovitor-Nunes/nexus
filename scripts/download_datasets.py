# scripts/download_datasets.py
"""
Download de datasets do Kaggle (robusto para TCC/ML pipeline).

Pré-requisito:
  pip install kaggle
  ~/.kaggle/kaggle.json configurado
  chmod 600 ~/.kaggle/kaggle.json

Execução:
  python scripts/download_datasets.py
"""

from pathlib import Path
import subprocess
import logging
import shutil
import sys
import time

BASE_DIR = Path("data/raw")

DATASETS = {
    "kaggle_fakenews": "clmentbisaillon/fake-and-real-news-dataset",
    "liar":            "doanquanvietnamca/liar-dataset",
    "politifact":      "muhammadaqeelkabir/politifact-dataset",
    "gossipcop":      "subodh7300/gossipcop",
}

MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def run_cmd(cmd, check=True):
    """
    Executa comando capturando stdout/stderr para debug real.
    """
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.stdout:
        logger.debug(result.stdout.strip())

    if result.stderr:
        logger.debug(result.stderr.strip())

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return result


def check_kaggle_cli():
    """
    Verifica se CLI existe e se autenticação funciona de verdade.
    """

    if shutil.which("kaggle") is None:
        logger.error("Kaggle CLI não encontrado.")
        logger.error("Instale com: pip install kaggle")
        sys.exit(1)

    # teste REAL de autenticação
    try:
        run_cmd(["kaggle", "datasets", "list", "-s", "fake news"])
        logger.info("Autenticação Kaggle OK.")
    except subprocess.CalledProcessError as e:
        logger.error("Falha na autenticação do Kaggle.")
        logger.error("Verifique ~/.kaggle/kaggle.json")
        logger.error(e.stderr)
        sys.exit(1)


def dataset_already_exists(path: Path) -> bool:
    """
    Evita re-download se já houver arquivos de dados.
    """

    patterns = ["*.csv", "*.tsv", "*.json", "*.parquet"]

    for p in patterns:
        if list(path.rglob(p)):
            return True

    return False


def download_dataset(folder: str, kaggle_id: str):

    target = BASE_DIR / folder
    target.mkdir(parents=True, exist_ok=True)

    if dataset_already_exists(target):
        logger.info(f"[SKIP] {folder} já existe.")
        return

    logger.info("=" * 70)
    logger.info(f"Dataset: {folder}")
    logger.info(f"Kaggle ID: {kaggle_id}")

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        kaggle_id,
        "-p",
        str(target),
        "--unzip"
    ]

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            result = run_cmd(cmd)
            logger.info(result.stdout.strip() or f"[OK] {folder}")
            return

        except subprocess.CalledProcessError as e:

            logger.warning(
                f"Falha tentativa {attempt}/{MAX_RETRIES} -> {folder}"
            )

            logger.warning(f"Erro Kaggle: {e.stderr.strip()}")

            # erro final
            if attempt == MAX_RETRIES:
                logger.error(f"[ERRO FINAL] {folder}")
                return

            time.sleep(2 * attempt)  # backoff simples


def main():

    logger.info("Verificando Kaggle CLI e autenticação...")
    check_kaggle_cli()

    logger.info(f"{len(DATASETS)} datasets configurados.")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for folder, kaggle_id in DATASETS.items():
        download_dataset(folder, kaggle_id)

    logger.info("=" * 70)
    logger.info("Downloads finalizados.")


if __name__ == "__main__":
    main()