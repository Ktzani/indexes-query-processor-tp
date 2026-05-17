"""
Scorers: TF-IDF e BM25.

Recebem o DAATResult (com matched_postings) e o DocumentIndex, e
calculam um score escalar para cada documento que casou.

Decisoes de design:

TF-IDF variant: ltn (query) x lnc-sem-cosine (doc)
    l (log TF):       1 + log(tf)
    n / t (idf):      query usa log(N/df); doc nao tem idf
    c (cosine):       SIMPLIFICADO. Nao temos norma do doc armazenada;
                      ranking continua valido (so muda scale).

    score(doc, q) = sum_{t em q} (1 + log(tf_doc)) * log(N / df_t)

    Como assumimos query sem duplicatas (Query.terms eh deduplicado),
    o tf_query eh sempre 1 e nao precisa ser multiplicado.

BM25 (Robertson, classico):
    score(doc, q) = sum_{t em q}
                    idf(t) * (tf * (k1+1)) /
                    (tf + k1 * (1 - b + b * |doc|/avgdl))

    idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
    (forma "BM25+" do Lucene, evita IDF negativo)

    Hiperparametros: k1=1.2, b=0.75 (em config/processor.py)

Performance: usa math.log (Python stdlib). numpy seria overkill para
scoring pontual de 1 doc por vez. Para batch, vetorizaria com numpy.
"""

import math
from abc import ABC, abstractmethod

from src.config.processor import BM25_B, BM25_K1
from src.index.document_index import DocumentIndex
from src.index.posting import Posting
from src.index.term_lexicon import TermLexicon


class Scorer(ABC):
    """Interface comum para scorers."""

    def __init__(
        self,
        doc_index: DocumentIndex,
        lexicon: TermLexicon,
    ):
        self._doc_index = doc_index
        self._lexicon = lexicon
        self._num_docs = doc_index.num_docs()

    @abstractmethod
    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        """
        Calcula o score do doc_id dado as postings dos termos da query
        que existem no doc.

        Parametros:
            doc_id: id interno do documento
            query_postings: dict term -> Posting (do DAATResult)

        Retorna:
            score escalar (maior = mais relevante)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do scorer (para debug/log)."""
        ...


class TFIDFScorer(Scorer):
    """
    TF-IDF com variante lnc.ltn (sem cosine no doc).

    Pre-calcula IDF de cada termo apenas uma vez no init para evitar
    recalculo a cada documento.
    """

    @property
    def name(self) -> str:
        return "TFIDF"

    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        score = 0.0
        for term, posting in query_postings.items():
            tf = posting.tf
            df = self._lexicon.get_df(term)
            if df == 0:
                # Defesa: teoricamente nao deveria acontecer (o DAAT
                # ja garante que o termo existe), mas seguro ignorar.
                continue
            # log TF + IDF (sem cosine; ranking eh preservado)
            log_tf = 1.0 + math.log(tf)
            idf = math.log(self._num_docs / df)
            score += log_tf * idf
        return score


class BM25Scorer(Scorer):
    """
    BM25 (Robertson) com IDF "BM25+" (sem negativos).

    Hiperparametros lidos de config: BM25_K1=1.2, BM25_B=0.75.
    """

    def __init__(
        self,
        doc_index: DocumentIndex,
        lexicon: TermLexicon,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        super().__init__(doc_index, lexicon)
        self._k1 = k1
        self._b = b
        self._avgdl = doc_index.avg_doc_length()

    @property
    def name(self) -> str:
        return "BM25"

    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        doc_length = self._doc_index.get_length(doc_id)
        # Fator de normalizacao por tamanho do doc (constante para o doc)
        norm = 1.0 - self._b + self._b * (doc_length / self._avgdl)

        score = 0.0
        for term, posting in query_postings.items():
            tf = posting.tf
            df = self._lexicon.get_df(term)
            if df == 0:
                continue
            # IDF "BM25+" do Lucene
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)
            # TF saturado com normalizacao por tamanho
            tf_component = (tf * (self._k1 + 1.0)) / (tf + self._k1 * norm)
            score += idf * tf_component
        return score


def get_scorer(
    name: str,
    doc_index: DocumentIndex,
    lexicon: TermLexicon,
) -> Scorer:
    """
    Factory: retorna o scorer apropriado pelo nome.
    Aceita 'TFIDF' ou 'BM25' (case-insensitive).
    """
    name_normalized = name.strip().upper()
    if name_normalized == "TFIDF":
        return TFIDFScorer(doc_index, lexicon)
    if name_normalized == "BM25":
        return BM25Scorer(doc_index, lexicon)
    raise ValueError(f"Scorer desconhecido: {name!r}. Use 'TFIDF' ou 'BM25'.")
