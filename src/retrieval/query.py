"""
Query: representa uma query de busca apos pre-processamento.

CRITICO: usa EXATAMENTE o mesmo Tokenizer + Normalizer que o indexer.
Sem isso, queries nao casariam com documentos (por exemplo, se o
indexer faz stemming mas o processor nao, 'running' no doc viraria
'run' no indice mas a query 'running' continuaria 'running' e nao
acharia nada).

Decisao de design: a query mantem dois estados:
    raw_text: a string original (para o JSON de output)
    terms:    a lista de termos apos preprocessing (para o DAAT)

Termos duplicados na query: pre-processamos a lista PRESERVANDO
duplicatas? Nao. Removemos duplicatas (preservando ordem). Razao:
o DAAT conjunctive faz interseccao das postings, e termos repetidos
nao mudam o conjunto resultante; alem disso o TF do termo no
documento eh o que importa para o score, nao no quanto ele aparece
na query.
"""

from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer


class Query:
    """Uma query de busca, com texto cru e termos preprocessados."""

    def __init__(
        self,
        raw_text: str,
        tokenizer: Tokenizer,
        normalizer: Normalizer,
    ):
        self._raw_text = raw_text.strip()
        # Aplica o MESMO pipeline do indexer.
        tokens = tokenizer.tokenize(self._raw_text)
        normalized = normalizer.normalize(tokens)
        # Remove duplicatas preservando ordem (importante para outputs
        # determinasticos e para evitar trabalho duplicado no DAAT).
        seen = set()
        self._terms: list[str] = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                self._terms.append(t)

    @property
    def raw_text(self) -> str:
        """Texto original da query (para o JSON de output)."""
        return self._raw_text

    @property
    def terms(self) -> list[str]:
        """Termos preprocessados, sem duplicatas."""
        return self._terms

    def is_empty(self) -> bool:
        """
        True se a query nao gerou termos uteis (vazia, so stopwords,
        ou termos descartados por tamanho).
        """
        return len(self._terms) == 0

    def __repr__(self) -> str:
        return f"Query(raw={self._raw_text!r}, terms={self._terms})"