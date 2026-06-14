"""
src/preprocessing/cleaner.py

Pipeline de limpeza de texto configurável por experimento.

Design: cada operação é um parâmetro booleano independente.
Isso é essencial para ablation studies — você pode isolar o
impacto de cada etapa de limpeza sem modificar código.

Ordem das operações importa:
  1. Unicode normalization (antes de tudo)
  2. HTML removal (antes de URL — captura links em tags)
  3. URL removal/replacement
  4. Email/username removal
  5. Punctuation normalization
  6. Whitespace normalization
  7. Lowercasing (ÚLTIMO — features estilométricas precisam do case original)
"""

import re
import unicodedata
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Pipeline de limpeza configurável.

    NOTA METODOLÓGICA: lowercase=False por padrão.
    Capitalização excessiva é um sinal estilométrico de fake news.
    Se você aplicar lowercase antes de extrair features estilométricas,
    perde essa informação. O pipeline correto é:
        1. extrair features estilométricas do texto original
        2. aplicar lowercase apenas para o input do modelo de linguagem
    """

    # Padrões compilados uma vez — reutilizados em todo o batch
    _HTML       = re.compile(r'<[^>]+>')
    _URL        = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    _EMAIL      = re.compile(r'\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b')
    _USERNAME   = re.compile(r'@\w+')
    _REP_PUNCT  = re.compile(r'([!?.,-])\1{2,}')      # !!! → !
    _REP_SPACE  = re.compile(r'[ \t]+')                # múltiplos espaços
    _REP_LINES  = re.compile(r'\n{3,}')               # 3+ quebras de linha
    _BYLINE     = re.compile(                         # "CITY (REUTERS) —"
        r'^[A-Z\s,]+\([A-Z]+\)\s*[-—]\s*', re.MULTILINE
    )

    def __init__(
        self,
        remove_html:            bool = True,
        remove_urls:            bool = True,
        url_replacement:        str  = "[URL]",   # "" para remover completamente
        remove_emails:          bool = True,
        remove_usernames:       bool = True,
        remove_bylines:         bool = True,      # "WASHINGTON (Reuters) —"
        normalize_punctuation:  bool = True,
        normalize_whitespace:   bool = True,
        lowercase:              bool = False,
        min_length:             int  = 10,        # descarta textos muito curtos
    ):
        self.cfg = {
            "remove_html":           remove_html,
            "remove_urls":           remove_urls,
            "url_replacement":       url_replacement,
            "remove_emails":         remove_emails,
            "remove_usernames":      remove_usernames,
            "remove_bylines":        remove_bylines,
            "normalize_punctuation": normalize_punctuation,
            "normalize_whitespace":  normalize_whitespace,
            "lowercase":             lowercase,
            "min_length":            min_length,
        }

    # ── Limpeza individual ────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Limpa um único texto. Retorna string vazia se inválido."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Normalização Unicode (NFKC: decompõe + recompõe canonicamente)
        #    Remove caracteres invisíveis, normaliza espaço em branco Unicode
        text = unicodedata.normalize("NFKC", text)

        # 2. HTML
        if self.cfg["remove_html"]:
            text = self._HTML.sub(" ", text)

        # 3. Bylines de agência (ex: "WASHINGTON (Reuters) —")
        #    CRÍTICO para o Kaggle dataset — bylines são source leakage direto
        if self.cfg["remove_bylines"]:
            text = self._BYLINE.sub("", text)

        # 4. URLs
        if self.cfg["remove_urls"]:
            repl = self.cfg["url_replacement"]
            text = self._URL.sub(repl if repl else " ", text)

        # 5. Emails
        if self.cfg["remove_emails"]:
            text = self._EMAIL.sub("[EMAIL]", text)

        # 6. @usernames (Twitter/social)
        if self.cfg["remove_usernames"]:
            text = self._USERNAME.sub("", text)

        # 7. Pontuação repetida: !!! → !
        if self.cfg["normalize_punctuation"]:
            text = self._REP_PUNCT.sub(r'\1', text)

        # 8. Whitespace
        if self.cfg["normalize_whitespace"]:
            text = self._REP_SPACE.sub(" ", text)
            text = self._REP_LINES.sub("\n\n", text)
            text = text.strip()

        # 9. Lowercase (ÚLTIMO)
        if self.cfg["lowercase"]:
            text = text.lower()

        # Validação de comprimento mínimo
        if len(text.split()) < self.cfg["min_length"]:
            return ""

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Limpa lista de textos. Preserva índices (textos inválidos → '')."""
        return [self.clean(t) for t in texts]

    def get_config(self) -> dict:
        """Retorna configuração para logging no MLflow."""
        return self.cfg.copy()

    def __repr__(self) -> str:
        active = [k for k, v in self.cfg.items() if v is True]
        return f"TextCleaner(active={active})"


# ── Configurações pré-definidas para experimentos ─────────────────────────

def cleaner_minimal() -> TextCleaner:
    """Limpeza mínima: apenas HTML e URLs. Para ablation de impacto de limpeza."""
    return TextCleaner(
        remove_html=True, remove_urls=True,
        remove_emails=False, remove_usernames=False,
        remove_bylines=False, normalize_punctuation=False,
    )


def cleaner_standard() -> TextCleaner:
    """Limpeza padrão recomendada para todos os experimentos."""
    return TextCleaner()   # defaults são a configuração padrão


def cleaner_aggressive() -> TextCleaner:
    """
    Limpeza agressiva: remove bylines, lowercase.
    Máxima remoção de source leakage — testar se melhora cross-domain F1.
    """
    return TextCleaner(
        remove_bylines=True,
        lowercase=True,
        url_replacement="",   # Remove URL completamente em vez de placeholder
    )