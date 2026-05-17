"""
Indice invertido parcial em memoria.

Acumula postings ate atingir o orcamento de memoria; entao eh
despejado em disco como um "block" no formato binario padrao do
projeto. O orquestrador (SPIMI) decide quando flushar com base no
MemoryMonitor.

Formato do block em disco:
    Para cada termo (em ordem alfabetica):
        [term_length: uint16][term: utf-8 bytes][num_postings: uint32]
        [posting_1 = (doc_id, tf) = 8 bytes]
        [posting_2 = (doc_id, tf) = 8 bytes]
        ...

Por que ordenado:
    O merger executa um k-way merge sobre os blocks. Se cada block ja
    esta ordenado, o merge final eh linear no numero total de
    postings (heap de k tamanho constante).

Por que postings dentro do termo nao precisam ser ordenadas aqui:
    Documentos sao processados sequencialmente (a thread reader
    entrega na ordem do JSONL e atribui doc_ids monotonicamente). Logo
    as postings de cada termo ja saem em ordem crescente de doc_id.
"""

from collections import Counter, defaultdict

from src.utils.io_utils import write_posting, write_uint16, write_uint32


class PartialIndex:
    """Indice invertido parcial em memoria. Nao eh thread-safe; usar
    com lock externo no SPIMI."""

    def __init__(self):
        # term -> list of (doc_id, tf)
        self._index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        # Contadores expostos
        self._num_postings = 0
        self._num_docs = 0

    def add_document(self, doc_id: int, terms: list[str]):
        """
        Adiciona um documento ao indice. Recebe a lista de termos JA
        normalizados (com repeticoes, na ordem do texto).

        Calcula TF localmente e adiciona uma posting por termo unico.
        Postings dentro de cada termo ficam em ordem crescente de
        doc_id, pois docs sao processados sequencialmente.
        """
        if not terms:
            return
        # Counter eh O(n) numa passada.
        tfs = Counter(terms)
        for term, tf in tfs.items():
            self._index[term].append((doc_id, tf))
            self._num_postings += 1
        self._num_docs += 1

    def num_terms(self) -> int:
        """Quantidade de termos unicos atualmente no indice."""
        return len(self._index)

    def num_postings(self) -> int:
        """Quantidade total de postings (sum over terms of len(postings))."""
        return self._num_postings

    def num_docs(self) -> int:
        """Quantidade de documentos contribuidos a este indice parcial."""
        return self._num_docs

    def is_empty(self) -> bool:
        return self._num_postings == 0

    def clear(self):
        """
        Zera o indice apos um flush. CRITICO: tambem zera o dict
        internamente para que o garbage collector libere a memoria
        imediatamente (em vez de manter strings de termos em memoria).
        """
        self._index.clear()
        # Cria um defaultdict novo para garantir liberacao de overhead
        # interno do dict anterior (rehash buckets etc.).
        self._index = defaultdict(list)
        self._num_postings = 0
        self._num_docs = 0

    def dump_to_disk(self, path: str) -> dict:
        """
        Escreve o block em disco no formato binario padrao, ordenado
        por termo (string compare).

        Retorna metadados do block:
            {
                "path": path,
                "num_terms": int,
                "num_postings": int,
            }
        """
        if self.is_empty():
            raise ValueError("PartialIndex vazio; nao ha o que despejar")

        sorted_terms = sorted(self._index.keys())

        with open(path, "wb") as f:
            for term in sorted_terms:
                postings = self._index[term]
                term_bytes = term.encode("utf-8")
                # Cabecalho do termo
                write_uint16(f, len(term_bytes))
                f.write(term_bytes)
                write_uint32(f, len(postings))
                # Postings: o ordenamento por doc_id eh garantido pelo
                # invariante de insercao (docs processados em ordem).
                for doc_id, tf in postings:
                    write_posting(f, doc_id, tf)

        return {
            "path": path,
            "num_terms": len(sorted_terms),
            "num_postings": self._num_postings,
        }
