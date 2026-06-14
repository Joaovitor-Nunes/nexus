"""
src/preprocessing/pipeline.py

Orquestrador do pipeline de pré-processamento.

Responsabilidades:
  1. Aplicar TextCleaner em todos os artigos
  2. Opcionalmente aplicar EntityAnonymizer
  3. Serializar artigos processados em Parquet (eficiente e tipado)
  4. Logar estatísticas de qualidade do processamento

Decisão de design — por que Parquet?
  - Preserva tipos de dados sem perda
  - Compressão eficiente (~4x menor que CSV para texto)
  - Leitura rápida com pandas/pyarrow
  - Suporta schema explícito
  - Padrão industrial para datasets de NLP
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

from ..data.schema import Article, Label
from .cleaner import TextCleaner, cleaner_standard
from .anonymizer import EntityAnonymizer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline completo: limpeza → (anonimização) → serialização.

    O campo `text_cleaned` sempre é preenchido.
    O campo `text_anonymized` é preenchido apenas se anonymizer não for None.
    Isso permite comparar modelos com/sem anonimização no mesmo arquivo.
    """

    def __init__(
        self,
        cleaner:    TextCleaner              = None,
        anonymizer: Optional[EntityAnonymizer] = None,
        batch_size: int                        = 128,
    ):
        self.cleaner    = cleaner or cleaner_standard()
        self.anonymizer = anonymizer
        self.batch_size = batch_size

    def process(self, articles: List[Article]) -> List[Article]:
        """
        Aplica pipeline a uma lista de artigos.
        Modifica os artigos in-place E retorna a lista.
        Artigos com texto vazio após limpeza são mantidos mas sinalizados.
        """
        logger.info(
            f"Iniciando pipeline: {len(articles):,} artigos | "
            f"cleaner={self.cleaner} | "
            f"anonymizer={'sim (' + self.anonymizer.spacy_model + ')' if self.anonymizer else 'não'}"
        )
        t0 = time.time()

        # ── Etapa 1: Limpeza ────────────────────────────────────────────
        texts = [a.text or "" for a in articles]
        cleaned = self.cleaner.clean_batch(texts)

        n_empty_after_clean = sum(1 for c in cleaned if not c)
        for article, clean_text in zip(articles, cleaned):
            article.text_cleaned = clean_text

        logger.info(
            f"  Limpeza concluída: {n_empty_after_clean} textos esvaziados "
            f"({n_empty_after_clean/len(articles):.1%})"
        )

        # ── Etapa 2: Anonimização (opcional) ────────────────────────────
        if self.anonymizer is not None:
            logger.info(
                f"  Anonimizando entidades: {sorted(self.anonymizer.entity_types)} "
                f"(batch_size={self.batch_size})"
            )
            # Processar em batches para logar progresso
            n_batches = (len(cleaned) + self.batch_size - 1) // self.batch_size

            all_anonymized = []
            for i in range(n_batches):
                batch = cleaned[i * self.batch_size:(i + 1) * self.batch_size]
                anon_batch = self.anonymizer.anonymize_batch(batch)
                all_anonymized.extend(anon_batch)

                if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                    pct = (i + 1) / n_batches * 100
                    logger.info(f"    Anonimização: {pct:.0f}% ({i+1}/{n_batches} batches)")

            for article, anon_text in zip(articles, all_anonymized):
                article.text_anonymized = anon_text

            logger.info("  Anonimização concluída.")

        elapsed = time.time() - t0
        logger.info(
            f"Pipeline concluído em {elapsed:.1f}s "
            f"({len(articles)/elapsed:.0f} artigos/s)"
        )

        return articles

    def get_model_input(self, article: Article, use_anonymized: bool = False) -> str:
        """
        Retorna o texto correto para o modelo dependendo do modo.
        Fallback gracioso: texto original se versão processada não existir.
        """
        if use_anonymized and article.text_anonymized:
            return article.text_anonymized
        if article.text_cleaned:
            return article.text_cleaned
        return article.text or ""


def articles_to_dataframe(articles: List[Article]) -> pd.DataFrame:
    """
    Converte lista de Article para DataFrame.
    Preserva todos os campos relevantes para treino e análise.
    """
    rows = []
    for a in articles:
        rows.append({
            "id":               a.id,
            "source_dataset":   a.source_dataset,
            "label":            int(a.label) if a.label is not None else None,
            "label_original":   a.label_original,
            "text":             a.text or "",
            "title":            a.title or "",
            "text_cleaned":     a.text_cleaned or "",
            "text_anonymized":  a.text_anonymized or "",
            "subject":          a.subject or "",
            "speaker":          a.speaker or "",
            "date":             a.date or "",
            "split":            a.split or "",
            "word_count":       a.word_count(),
        })
    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Salva DataFrame processado em Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Salvo: {path} ({len(df):,} linhas, {size_mb:.1f} MB)")


def load_processed(path: Path) -> pd.DataFrame:
    """Carrega DataFrame processado de Parquet."""
    df = pd.read_parquet(path)
    logger.info(f"Carregado: {path} ({len(df):,} linhas)")
    return df


def compute_processing_stats(
    original: List[Article],
    processed: List[Article],
) -> Dict[str, Any]:
    """
    Compara artigos antes e depois do processamento.
    Métricas de qualidade para documentar no TCC.
    """
    orig_lengths = [a.word_count() for a in original]
    proc_lengths = [
        len((a.text_cleaned or "").split()) for a in processed
    ]

    n_dropped = sum(1 for a in processed if not a.text_cleaned)

    return {
        "n_original":          len(original),
        "n_dropped":           n_dropped,
        "drop_rate":           n_dropped / len(original) if original else 0,
        "avg_length_before":   float(np.mean(orig_lengths)),
        "avg_length_after":    float(np.mean([l for l in proc_lengths if l > 0])),
        "length_reduction_pct": float(
            (np.mean(orig_lengths) - np.mean([l for l in proc_lengths if l > 0]))
            / np.mean(orig_lengths) * 100
            if orig_lengths else 0
        ),
        "has_anonymization": any(a.text_anonymized for a in processed),
    }