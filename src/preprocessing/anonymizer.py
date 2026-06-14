"""
src/preprocessing/anonymizer.py

Anonimização de entidades nomeadas via spaCy NER.

═══════════════════════════════════════════════════════════════
ANÁLISE CRÍTICA: QUANDO ANONIMIZAR AJUDA VS PREJUDICA
═══════════════════════════════════════════════════════════════

AJUDA — casos em que anonimização melhora generalização:

  1. Source memorization no Kaggle dataset:
     "WASHINGTON (Reuters) — The government said..."
     → "[GPE] ([ORG]) — The government said..."
     O modelo deixa de associar "Reuters" com "real".
     SEM anonimização: accuracy artificial alta in-domain.
     COM anonimização: generalização real para fontes não vistas.

  2. Temporal drift: figuras políticas de 2016 mudam.
     "Hillary Clinton stated..." em treino de 2016 não generaliza
     para dados de 2024. "[PERSON] stated..." é temporalmente neutro.

  3. Cross-dataset generalization: LIAR fala de políticos americanos,
     GossipCop fala de celebridades. Entidades específicas são
     artefatos de domínio, não sinais de factualidade.

  4. Fairness: sem anonimização, o modelo pode aprender associações
     espúrias entre grupos (étnicos, políticos, religiosos) e fake news.
     Isso é um risco real de discriminação algorítmica.

PREJUDICA — casos em que anonimização perde informação real:

  1. Credibilidade de fonte é um sinal legítimo:
     "According to a Harvard study..." vs "According to my neighbor..."
     Anonimizar [ORG] perde essa distinção válida.

  2. Fact verification baseada em speaker:
     Em LIAR, o speaker (e seu histórico de mentiras) é feature válida.
     Anonymizar PERSON pode degradar performance nesse dataset.

  3. Coerência de co-referência: em artigos longos, entidades
     sustentam a narrativa. "[PERSON] said [PERSON] believed [PERSON]
     would..." perde rastreabilidade de quem diz o quê.

  4. Political alignment detection: em fake news partidária,
     QUEM é mencionado e com que valência é o sinal principal.

CONCLUSÃO METODOLÓGICA:
  Tratar como variável de experimento binária.
  Hipótese H1: anonymization_gain = cross_domain_F1(anon) - cross_domain_F1(no_anon) > 0
  Hipótese H2: o ganho varia por domínio — maior em Kaggle, menor em LIAR
  Hipótese H3: transformers são mais robustos à anonimização que TF-IDF+LR
═══════════════════════════════════════════════════════════════
"""

import logging
from typing import List, Optional, Dict, Set

logger = logging.getLogger(__name__)

# Mapeamento entidade → token de substituição
ENTITY_REPLACEMENT = {
    "PERSON":  "[PERSON]",
    "ORG":     "[ORG]",
    "GPE":     "[GPE]",        # Geo-Political Entity: países, cidades, estados
    "NORP":    "[NORP]",       # Nationalities, Religious, Political groups
    "DATE":    "[DATE]",
    "MONEY":   "[MONEY]",
    "PERCENT": "[PERCENT]",
    "PRODUCT": "[PRODUCT]",
    "EVENT":   "[EVENT]",
    "LOC":     "[LOC]",        # Localizações não-GPE (montanhas, rios)
    "FAC":     "[FAC]",        # Facilities (aeroportos, prédios)
}

# Subconjuntos por estratégia
ENTITIES_SOURCE_ONLY  = {"ORG"}                                    # só remove fontes
ENTITIES_POLITICAL    = {"PERSON", "ORG", "GPE", "NORP"}           # padrão político
ENTITIES_FULL         = set(ENTITY_REPLACEMENT.keys())             # tudo


class EntityAnonymizer:
    """
    Anonimizador de entidades via spaCy NER.

    Lazy loading: spaCy é carregado apenas na primeira chamada.
    Isso evita overhead de importação quando anonimização está desabilitada.
    """

    def __init__(
        self,
        entity_types: Optional[Set[str]] = None,
        spacy_model:  str = "en_core_web_sm",   # sm: rápido; trf: mais preciso
        batch_size:   int = 64,
    ):
        """
        entity_types: conjunto de tipos de entidade a anonimizar.
                      None → usa ENTITIES_POLITICAL (padrão recomendado).
        spacy_model:  "en_core_web_sm" para CPU rápida,
                      "en_core_web_trf" para máxima precisão (GPU recomendada).
        batch_size:   número de textos por batch no pipe do spaCy.
        """
        self.entity_types = entity_types or ENTITIES_POLITICAL
        self.spacy_model  = spacy_model
        self.batch_size   = batch_size
        self._nlp         = None

    def _load_spacy(self) -> None:
        """Carrega spaCy com apenas os componentes necessários."""
        if self._nlp is not None:
            return

        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy não instalado. Execute: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )

        try:
            # Desabilitar componentes não necessários para NER puro
            self._nlp = spacy.load(
                self.spacy_model,
                disable=["parser", "lemmatizer", "attribute_ruler"]
            )
        except OSError:
            logger.warning(
                f"Modelo '{self.spacy_model}' não encontrado. "
                f"Execute: python -m spacy download {self.spacy_model}"
            )
            raise

        logger.info(
            f"spaCy carregado: {self.spacy_model} | "
            f"entidades alvo: {sorted(self.entity_types)}"
        )

    def _replace_entities(self, doc) -> str:
        """
        Substitui entidades em um Doc spaCy.
        Substituição reversa (de trás para frente) preserva offsets de caractere.
        """
        text = doc.text
        replacements = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                token = ENTITY_REPLACEMENT.get(ent.label_, f"[{ent.label_}]")
                replacements.append((ent.start_char, ent.end_char, token))

        # Ordem reversa: substituir de trás para frente evita offset drift
        for start, end, token in sorted(replacements, key=lambda x: x[0], reverse=True):
            text = text[:start] + token + text[end:]

        return text

    def anonymize(self, text: str) -> str:
        """Anonimiza um único texto."""
        if not text:
            return text

        self._load_spacy()
        doc = self._nlp(text)
        return self._replace_entities(doc)

    def anonymize_batch(self, texts: List[str]) -> List[str]:
        """
        Anonimiza lista de textos em batch.
        Muito mais eficiente que loop de chamadas individuais.
        O pipe do spaCy processa em paralelo internamente.
        """
        self._load_spacy()
        results = []

        # nlp.pipe: processa em batch, muito mais eficiente que nlp() em loop
        for doc in self._nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=1,   # >1 só funciona com modelos não-GPU
        ):
            results.append(self._replace_entities(doc))

        return results

    def analyze_entity_distribution(self, texts: List[str]) -> Dict:
        """
        Analisa quais entidades aparecem e com que frequência.
        
        Útil para:
        - Verificar que anonimização está funcionando
        - Identificar entidades mais discriminativas (source leakage)
        - Documentar o que foi anonimizado no TCC
        """
        from collections import Counter
        self._load_spacy()

        entity_counts  = Counter()
        entity_surface = {et: Counter() for et in self.entity_types}

        for doc in self._nlp.pipe(texts, batch_size=self.batch_size):
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    entity_counts[ent.label_] += 1
                    entity_surface[ent.label_][ent.text.lower()] += 1

        return {
            "total_by_type": dict(entity_counts),
            "top_20_by_type": {
                et: dict(counter.most_common(20))
                for et, counter in entity_surface.items()
            },
        }

    def get_config(self) -> dict:
        return {
            "entity_types": sorted(self.entity_types),
            "spacy_model":  self.spacy_model,
            "batch_size":   self.batch_size,
        }


# ── Configurações pré-definidas ──────────────────────────────────────────

def anonymizer_none() -> None:
    """Sem anonimização — baseline."""
    return None


def anonymizer_source_only() -> EntityAnonymizer:
    """
    Anonimiza apenas ORG (fontes como Reuters, AP).
    Hipótese: remove source memorization sem perder sinais políticos.
    """
    return EntityAnonymizer(entity_types=ENTITIES_SOURCE_ONLY)


def anonymizer_political() -> EntityAnonymizer:
    """
    Anonimiza entidades políticas: PERSON, ORG, GPE, NORP.
    Configuração padrão recomendada para o experimento principal.
    """
    return EntityAnonymizer(entity_types=ENTITIES_POLITICAL)


def anonymizer_full() -> EntityAnonymizer:
    """
    Anonimização máxima: todos os tipos de entidade.
    Mais agressivo — pode perder sinais legítimos de credibilidade.
    """
    return EntityAnonymizer(entity_types=ENTITIES_FULL)