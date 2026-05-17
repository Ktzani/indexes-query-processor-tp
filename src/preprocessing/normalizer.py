"""
Normalizer: converte tokens brutos em termos prontos para indexacao.

Pipeline:
    tokens [ja em lowercase, sem pontuacao pura]
      -> remove stopwords
      -> filtra por comprimento [MIN, MAX]
      -> opcionalmente descarta puros numericos
      -> aplica stemming Snowball
      -> termos finais

A ordem importa: removemos stopwords ANTES de stemming porque alguns
stems de stopwords podem coincidir com termos uteis ("the" -> "the"
nao tem problema, mas evitar trabalho desnecessario eh bom). E
filtramos por tamanho ANTES de stemming porque a checagem eh barata.

CRITICO: indexer.py e processor.py DEVEM usar a mesma instancia
(mesma config) deste Normalizer, ou queries nao casam com documentos.
"""

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config.preprocessing import (
    DROP_PURE_NUMERIC,
    MAX_TOKEN_LENGTH,
    MIN_TOKEN_LENGTH,
    STEMMER_LANGUAGE,
)


class Normalizer:
    """Aplica stopword removal + stemming + filtros."""

    def __init__(
        self,
        language: str = STEMMER_LANGUAGE,
        min_length: int = MIN_TOKEN_LENGTH,
        max_length: int = MAX_TOKEN_LENGTH,
        drop_numeric: bool = DROP_PURE_NUMERIC,
    ):
        self._stemmer = SnowballStemmer(language)
        # frozenset eh ~2x mais rapido que set para 'in' em lookups
        # repetidos. ~180 stopwords em ingles.
        self._stopwords = frozenset(stopwords.words(language))
        self._min_length = min_length
        self._max_length = max_length
        self._drop_numeric = drop_numeric

    def normalize(self, tokens: list[str]) -> list[str]:
        """
        Recebe tokens (esperados em lowercase) e retorna termos
        normalizados prontos para indexacao.
        """
        result = []
        for tok in tokens:
            # Filtro 1: stopword
            if tok in self._stopwords:
                continue
            # Filtro 2: comprimento (antes de stemmer, pra economizar)
            if len(tok) < self._min_length or len(tok) > self._max_length:
                continue
            # Filtro 3: numero puro
            if self._drop_numeric and tok.isdigit():
                continue
            # Stemming
            stem = self._stemmer.stem(tok)
            # Recheca tamanho apos stem (stems podem ficar curtos demais)
            if len(stem) < self._min_length:
                continue
            result.append(stem)
        return result
