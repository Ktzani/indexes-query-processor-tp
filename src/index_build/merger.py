"""
K-way merge externo dos blocks parciais produzidos pelo SPIMI.

Algoritmo:
    1. Cada block eh aberto como um iterator de (term, postings) em
       ordem alfabetica de termo.
    2. Mantemos uma heap minima com (term, block_idx) onde term eh o
       proximo termo nao consumido daquele block.
    3. A cada iteracao, removemos da heap todos os elementos que
       tem o termo do topo. Combinamos suas postings, escrevemos no
       arquivo final e atualizamos o lexicon.
    4. Avancamos esses blocks (le proximo termo) e reinserimos na heap.

Complexidade:
    Tempo: O(N log K), N = total de postings, K = numero de blocks
    Espaco: O(K) para a heap + buffer da inverted list atual

Saida:
    - inverted_index_path: arquivo binario com as listas finais
    - lexicon: dict term -> (offset, df) que sera serializado a parte

Formato da inverted list no arquivo final:
    [num_postings: uint32][posting_1][posting_2]...
    (sem term_length/term: o lexicon ja resolve isso)

O offset que vai no lexicon eh o offset do uint32 num_postings.
"""

import heapq
import os
import sys

from src.utils.io_utils import (
    read_posting,
    read_uint16,
    read_uint32,
    write_posting,
    write_uint32,
)


class _BlockReader:
    """
    Iterator sobre um block binario. A cada chamada de next(),
    retorna (term: str, postings: list[(doc_id, tf)]) em ordem
    alfabetica de termo, ou None ao EOF.

    Mantem o arquivo aberto enquanto vivo; close() libera.
    """

    def __init__(self, path: str):
        self._path = path
        self._f = open(path, "rb")
        self._eof = False

    def next_entry(self) -> tuple[str, list[tuple[int, int]]] | None:
        """Le proxima entrada (term, postings) ou retorna None ao EOF."""
        if self._eof:
            return None
        term_len = read_uint16(self._f)
        if term_len == -1:
            self._eof = True
            return None
        term_bytes = self._f.read(term_len)
        if len(term_bytes) < term_len:
            self._eof = True
            return None
        term = term_bytes.decode("utf-8")
        num_postings = read_uint32(self._f)
        if num_postings == -1:
            self._eof = True
            return None
        postings = []
        for _ in range(num_postings):
            p = read_posting(self._f)
            if p is None:
                self._eof = True
                return None
            postings.append(p)
        return term, postings

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def merge_blocks(
    block_paths: list[str],
    inverted_index_path: str,
) -> dict[str, tuple[int, int]]:
    """
    Faz o k-way merge dos blocks parciais.

    Parametros:
        block_paths: lista de paths de blocks binarios (gerados pelo SPIMI)
        inverted_index_path: arquivo de saida (sera sobrescrito)

    Retorna:
        lexicon: dict term -> (offset_no_arquivo, document_frequency)
    """
    if not block_paths:
        # Edge case: nenhum block. Cria arquivo vazio e lexicon vazio.
        open(inverted_index_path, "wb").close()
        return {}

    # Abre todos os blocks
    readers = [_BlockReader(p) for p in block_paths]

    # Inicializa heap com primeira entrada de cada block
    # heap: (term, block_idx) -- block_idx eh tie-breaker
    heap: list[tuple[str, int]] = []
    # Buffer da entrada corrente de cada block (term, postings)
    current: list[tuple[str, list[tuple[int, int]]] | None] = []

    for i, reader in enumerate(readers):
        entry = reader.next_entry()
        if entry is not None:
            term, postings = entry
            current.append(entry)
            heapq.heappush(heap, (term, i))
        else:
            current.append(None)

    lexicon: dict[str, tuple[int, int]] = {}

    try:
        with open(inverted_index_path, "wb") as out:
            while heap:
                # Pega menor termo
                top_term, _ = heap[0]

                # Coleta todas as postings de todos os blocks que tem
                # este termo no topo
                combined: dict[int, int] = {}  # doc_id -> tf (failsafe)

                # Remove e processa todos os elementos com top_term
                while heap and heap[0][0] == top_term:
                    _, block_idx = heapq.heappop(heap)
                    entry = current[block_idx]
                    if entry is None:
                        # Patologico: nao deveria acontecer
                        continue
                    _, postings = entry
                    for doc_id, tf in postings:
                        # Failsafe: se um doc_id aparecer em multiplos
                        # blocks (nao deveria), somamos as TFs.
                        if doc_id in combined:
                            combined[doc_id] += tf
                        else:
                            combined[doc_id] = tf

                    # Avanca o iterator deste block
                    next_entry = readers[block_idx].next_entry()
                    current[block_idx] = next_entry
                    if next_entry is not None:
                        next_term, _ = next_entry
                        heapq.heappush(heap, (next_term, block_idx))

                # Postings finais ordenadas por doc_id
                final_postings = sorted(combined.items())
                df = len(final_postings)

                # Anota offset ANTES de escrever
                offset = out.tell()
                lexicon[top_term] = (offset, df)

                # Escreve a inverted list final
                write_uint32(out, df)
                for doc_id, tf in final_postings:
                    write_posting(out, doc_id, tf)
    finally:
        for r in readers:
            r.close()

    return lexicon


def cleanup_blocks(block_paths: list[str]):
    """Remove os arquivos de blocks parciais apos o merge."""
    for p in block_paths:
        try:
            os.remove(p)
        except OSError as e:
            print(f"[merger] aviso: nao removi {p}: {e}", file=sys.stderr)
