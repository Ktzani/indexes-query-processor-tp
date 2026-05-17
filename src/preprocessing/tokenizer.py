"""
Tokenizer: converte texto bruto em lista de tokens.

Estrategia:
1. NLTK word_tokenize (Treebank) faz o trabalho pesado: separa
   contracoes ('don't' -> ['do', 'n't']), trata pontuacao anexa,
   preserva apostrofes internos.
2. Lowercase e descarte de tokens puramente pontuacao.

Nao faz: stopword removal, stemming, filtro por tamanho.
Isso eh responsabilidade do Normalizer.

A separacao tokenizer/normalizer eh intencional: facilita testes e
permite trocar implementacoes de forma independente.
"""

import re

from nltk.tokenize import word_tokenize


# Token "lixo" se tiver SO pontuacao/simbolos. Mantemos tudo que tem
# pelo menos uma letra ou digito.
_HAS_ALNUM = re.compile(r"[a-z0-9]", re.IGNORECASE)

# Contracoes residuais do tokenizer Treebank (e.g., "don't" -> "do" + "n't").
# Esses pedacos nao tem valor semantico isolado e geram ruido no indice.
_CONTRACTION_RESIDUALS = frozenset([
    "n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'t",
])


class Tokenizer:
    """Tokenizador baseado em NLTK + filtros minimos."""

    def __init__(self, language: str = "english"):
        self._language = language

    def tokenize(self, text: str) -> list[str]:
        """
        Retorna a lista de tokens em lowercase, descartando pontuacao
        pura e residuos de contracoes ('n't, 's, etc.).
        """
        if not text:
            return []

        # word_tokenize precisa de texto str (nao bytes).
        raw_tokens = word_tokenize(text, language=self._language)

        result = []
        for tok in raw_tokens:
            tok = tok.lower()
            # Descarta se nao tem alfanumerico (pontuacao pura).
            if not _HAS_ALNUM.search(tok):
                continue
            # Descarta residuos de contracoes (n't, 's, etc.).
            if tok in _CONTRACTION_RESIDUALS:
                continue
            # Descarta tokens que comecam com apostrofe (ex: 'em, 'cause).
            if tok.startswith("'"):
                continue
            result.append(tok)
        return result
