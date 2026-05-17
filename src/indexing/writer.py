"""
Writer: persiste lexicon e document index em disco (formato pickle)
e calcula as estatisticas finais do indice.

O arquivo inverted.idx eh produzido diretamente pelo merger; aqui
apenas tomamos suas estatisticas (tamanho).

Por que pickle:
    Lexicon e doc_index sao dicts Python. Pickle eh a forma idiomatica
    e mais rapida de persisti-los, dado que estamos lock-in em Python
    (o enunciado pede Python 3.14 estritamente).

Por que protocolo 5:
    Protocolo 5 (Python 3.8+) suporta out-of-band buffers, importante
    para dicts grandes (4.6M docs). E mais compacto e mais rapido que
    protocolos legados.

Saida do write_index:
    {
        "Index Size": float,           # MB, total dos 3 arquivos
        "Elapsed Time": float,         # segundos, preenchido externamente
        "Number of Lists": int,        # numero de termos
        "Average List Size": float,    # total_postings / num_lists
    }
"""

import os
import pickle

from src.config.indexer import (
    DOCUMENT_INDEX_FILENAME,
    TERM_LEXICON_FILENAME,
    INVERTED_INDEX_FILENAME,
)


# Protocolo 5: out-of-band buffers, mais rapido para dicts grandes.
_PICKLE_PROTOCOL = 5


def write_lexicon(
    lexicon: dict[str, tuple[int, int]],
    index_dir: str,
) -> str:
    """
    Persiste o lexicon como pickle. Retorna o path do arquivo.

    Formato:
        dict[term: str -> (offset_no_inverted_idx: int, df: int)]
    """
    path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
    with open(path, "wb") as f:
        pickle.dump(lexicon, f, protocol=_PICKLE_PROTOCOL)
    return path


def write_doc_index(
    doc_index: dict[int, dict],
    index_dir: str,
) -> str:
    """
    Persiste o document index como pickle. Retorna o path do arquivo.

    Formato:
        dict[internal_id: int -> {
            "original_id": str,
            "length": int,    # numero de tokens apos preprocessing
        }]
    """
    path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)
    with open(path, "wb") as f:
        pickle.dump(doc_index, f, protocol=_PICKLE_PROTOCOL)
    return path


def compute_statistics(
    index_dir: str,
    lexicon: dict[str, tuple[int, int]],
) -> dict:
    """
    Computa as estatisticas exigidas pelo enunciado.

    Retorna dict com:
        - Index Size: tamanho total dos 3 arquivos em MB
        - Number of Lists: numero de termos no lexicon
        - Average List Size: media de postings por lista

    Nao inclui Elapsed Time: eh responsabilidade do entry point
    indexer.py medir e preencher.
    """
    inverted_path = os.path.join(index_dir, INVERTED_INDEX_FILENAME)
    lexicon_path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
    doc_index_path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)

    total_bytes = 0
    for p in (inverted_path, lexicon_path, doc_index_path):
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)

    num_lists = len(lexicon)
    if num_lists > 0:
        total_postings = sum(df for _, df in lexicon.values())
        avg_list_size = total_postings / num_lists
    else:
        avg_list_size = 0.0

    return {
        "Index Size": round(total_bytes / (1024.0 * 1024.0), 3),
        "Number of Lists": num_lists,
        "Average List Size": round(avg_list_size, 2),
    }
