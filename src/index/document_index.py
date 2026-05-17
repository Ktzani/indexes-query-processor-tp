"""
DocumentIndex: API de leitura do mapping internal_doc_id -> metadados.

Carrega doc_index.pkl em memoria e expoe lookups O(1) para:
    - original_id (string original do corpus)
    - length (numero de tokens apos preprocessing)
    - avg_doc_length (media usada pelo BM25)

Decisao: carregar tudo em RAM. Para 4.6M docs:
    ~50 bytes por doc (dict overhead + 2 keys) -> ~230 MB
    Cabe no orcamento de memoria de 1 GB do processor.

Alternativa rejeitada: lookup em disco com mmap. Adicionaria
complexidade sem ganho material (a query precisa ler o length de
todos os docs candidatos, gerando muitos seeks aleatorios).
"""

import os
import pickle

from src.config.indexer import DOCUMENT_INDEX_FILENAME


class DocumentIndex:
    """API de leitura do document index."""

    def __init__(self, index_dir: str):
        """Carrega doc_index.pkl em RAM e pre-calcula avg_doc_length."""
        path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)
        with open(path, "rb") as f:
            self._docs: dict[int, dict] = pickle.load(f)

        # Pre-calculo do avg_doc_length (usado pelo BM25)
        if self._docs:
            total = sum(d["length"] for d in self._docs.values())
            self._avg_length = total / len(self._docs)
        else:
            self._avg_length = 0.0

    def get_length(self, doc_id: int) -> int:
        """Retorna o numero de tokens (apos preprocessing) do documento."""
        return self._docs[doc_id]["length"]

    def get_original_id(self, doc_id: int) -> str:
        """Retorna o id original do documento (string do JSONL)."""
        return self._docs[doc_id]["original_id"]

    def avg_doc_length(self) -> float:
        """Comprimento medio dos documentos. Usado pelo BM25."""
        return self._avg_length

    def num_docs(self) -> int:
        """Numero total de documentos no indice (= N para TF-IDF/BM25)."""
        return len(self._docs)

    def __contains__(self, doc_id: int) -> bool:
        return doc_id in self._docs
