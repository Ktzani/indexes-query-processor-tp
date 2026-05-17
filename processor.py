"""
processor.py - Entry point do query processor.

Uso (conforme enunciado):
    python3 processor.py -i <INDEX> -q <QUERIES> -r <RANKER>

Argumentos:
    -i <INDEX>: caminho do diretorio com o indice
    -q <QUERIES>: arquivo com queries (uma por linha)
    -r <RANKER>: TFIDF ou BM25

Output:
    Um JSON por linha no stdout, no formato:
        {"Query": "...", "Results": [{"ID": "...", "Score": ...}, ...]}

Logs de progresso vao para STDERR.

Fluxo por query:
    Query(raw) -> Tokenizer -> Normalizer -> terms
    terms -> ConjunctiveDAAT -> DAATResult (matched_doc_ids + postings)
    matched docs -> Scorer (TFIDF | BM25) -> [(doc_id, score)]
    candidates -> Ranker -> top-K
    top-K -> JSON output
"""

import argparse
import json
import os
import sys
import time

from src.config.processor import TEXT_ENCODING, TOP_K
from src.index.document_index import DocumentIndex
from src.index.inverted_index import InvertedIndex
from src.index.term_lexicon import TermLexicon
from src.preprocessing.nltk_setup import ensure_nltk_data
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.retrieval.daat import ConjunctiveDAAT
from src.retrieval.query import Query
from src.retrieval.ranker import Ranker
from src.retrieval.scorer import get_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query processor with conjunctive DAAT matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        metavar="INDEX",
        help="Path to the index directory",
    )
    parser.add_argument(
        "-q",
        type=str,
        required=True,
        metavar="QUERIES",
        help="Path to the queries file (one query per line)",
    )
    parser.add_argument(
        "-r",
        type=str,
        required=True,
        metavar="RANKER",
        choices=["TFIDF", "BM25", "tfidf", "bm25"],
        help="Ranking function: TFIDF or BM25",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if not os.path.isdir(args.i):
        print(f"erro: diretorio de indice nao encontrado: {args.i}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.q):
        print(f"erro: arquivo de queries nao encontrado: {args.q}", file=sys.stderr)
        sys.exit(1)


def read_queries(queries_path: str) -> list[str]:
    """Le todas as queries do arquivo, uma por linha. Ignora linhas vazias."""
    queries = []
    with open(queries_path, "r", encoding=TEXT_ENCODING) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def process_query(
    raw_query: str,
    tokenizer: Tokenizer,
    normalizer: Normalizer,
    daat: ConjunctiveDAAT,
    scorer,
    ranker: Ranker,
    doc_index: DocumentIndex,
) -> dict:
    """
    Processa uma query end-to-end. Retorna o dict pronto para serializar
    como JSON.
    """
    # Preprocessing
    q = Query(raw_query, tokenizer, normalizer)

    # Query vazia (so stopwords, ou texto vazio) -> sem resultados
    if q.is_empty():
        return {"Query": q.raw_text, "Results": []}

    # DAAT: encontra docs candidatos
    daat_result = daat.intersect(q.terms)

    # Sem matches -> sem resultados
    if not daat_result.matched_doc_ids:
        return {"Query": q.raw_text, "Results": []}

    # Score cada candidato
    candidates: list[tuple[int, float]] = []
    for doc_id in daat_result.matched_doc_ids:
        postings = daat_result.matched_postings[doc_id]
        score = scorer.score(doc_id, postings)
        candidates.append((doc_id, score))

    # Top-K
    top = ranker.top_k(candidates)

    # Traduz doc_id interno para original_id (string), formata
    results = [
        {"ID": doc_index.get_original_id(doc_id), "Score": round(score, 4)}
        for doc_id, score in top
    ]

    return {"Query": q.raw_text, "Results": results}


def main():
    args = parse_args()
    validate_args(args)
    ranker_name = args.r.upper()

    # Setup NLTK (idempotente, rapido se ja baixado)
    ensure_nltk_data()

    # === Carrega indice (uma vez) ===
    print(f"[processor] carregando indice de {args.i}", file=sys.stderr)
    t0 = time.perf_counter()
    lexicon = TermLexicon(args.i)
    doc_index = DocumentIndex(args.i)
    elapsed_load = time.perf_counter() - t0
    print(
        f"[processor] indice carregado em {elapsed_load:.2f}s: "
        f"{lexicon.num_terms()} termos, {doc_index.num_docs()} docs, "
        f"avgdl={doc_index.avg_doc_length():.2f}",
        file=sys.stderr,
    )

    # === Setup das estruturas de processamento ===
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    ranker = Ranker(k=TOP_K)
    scorer = get_scorer(ranker_name, doc_index, lexicon)

    # === Le queries ===
    queries = read_queries(args.q)
    print(f"[processor] {len(queries)} queries lidas", file=sys.stderr)

    # === Processa todas as queries ===
    # O InvertedIndex eh aberto uma unica vez, em context manager.
    t_start = time.perf_counter()
    with InvertedIndex(args.i) as ii:
        daat = ConjunctiveDAAT(lexicon, ii)

        for i, raw_query in enumerate(queries, start=1):
            t_query_start = time.perf_counter()
            output = process_query(
                raw_query, tokenizer, normalizer, daat, scorer, ranker, doc_index,
            )
            t_query_elapsed = time.perf_counter() - t_query_start

            # Imprime JSON da query no stdout (uma linha por query)
            print(json.dumps(output, ensure_ascii=False))

            # Log de progresso no stderr
            print(
                f"[processor] q{i}/{len(queries)}: {raw_query!r} "
                f"-> {len(output['Results'])} results in {t_query_elapsed*1000:.1f}ms",
                file=sys.stderr,
            )

    elapsed_total = time.perf_counter() - t_start
    print(
        f"[processor] {len(queries)} queries processadas em "
        f"{elapsed_total:.2f}s ({elapsed_total/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
