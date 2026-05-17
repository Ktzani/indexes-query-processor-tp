"""
TermLexicon: API de leitura do lexicon term -> (offset, df).

Carrega lexicon.pkl em memoria. Para 4.6M docs com ~3-5M termos
unicos esperados, gasta ~100-150 MB. Aceitavel.

A presenca de um termo no lexicon eh CRUCIAL no DAAT: se um termo da
query nao esta no lexicon, a intersecao conjunctiva eh vazia
imediatamente (DAAT pode retornar 0 resultados sem ler o disco).
"""

import os
import pickle

from src.config.indexer import TERM_LEXICON_FILENAME


class TermLexicon:
    """API de leitura do lexicon."""

    def __init__(self, index_dir: str):
        """Carrega lexicon.pkl em RAM."""
        path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
        with open(path, "rb") as f:
            self._lex: dict[str, tuple[int, int]] = pickle.load(f)

    def has_term(self, term: str) -> bool:
        """True se o termo existe no indice."""
        return term in self._lex

    def get_entry(self, term: str) -> tuple[int, int] | None:
        """
        Retorna (offset, df) do termo no inverted index, ou None se
        o termo nao existe no indice.
        """
        return self._lex.get(term)

    def get_df(self, term: str) -> int:
        """Document frequency do termo (0 se nao existe)."""
        entry = self._lex.get(term)
        return entry[1] if entry else 0

    def num_terms(self) -> int:
        """Numero de termos unicos no lexicon."""
        return len(self._lex)

    def __contains__(self, term: str) -> bool:
        return term in self._lex
