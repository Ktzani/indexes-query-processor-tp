"""
Le o corpus em formato JSONL (uma linha = um JSON com um documento)
e produz objetos Document.

Implementado como generator para evitar carregar o corpus inteiro em
memoria. Para um corpus de 4.6M documentos, isso eh essencial.

Tolerancia a erros: linhas malformadas (JSON invalido ou campos
faltando) sao puladas com warning no stderr. NAO interrompemos a
indexacao por causa de uma linha corrompida.
"""

import json
import sys
from typing import Iterator

from src.config.indexer import TEXT_ENCODING
from src.index_build.document import Document


class CorpusReader:
    """Leitura streaming de um JSONL produzindo Documents."""

    def __init__(self, corpus_path: str, max_docs: int | None = None):
        """
        Parametros:
            corpus_path: caminho para o arquivo JSONL
            max_docs: se nao-None, limita a leitura aos primeiros N
                      docs (util em testes).
        """
        self._corpus_path = corpus_path
        self._max_docs = max_docs

        # Contadores expostos pos-iteracao (para logs/relatorio).
        self.docs_read = 0
        self.docs_skipped = 0

    def __iter__(self) -> Iterator[Document]:
        """Itera sobre o corpus produzindo objetos Document."""
        self.docs_read = 0
        self.docs_skipped = 0

        with open(self._corpus_path, "r", encoding=TEXT_ENCODING) as f:
            for line_num, line in enumerate(f, start=1):
                if self._max_docs and self.docs_read >= self._max_docs:
                    break

                line = line.strip()
                if not line:
                    continue

                doc = self._parse_line(line, line_num)
                if doc is None:
                    self.docs_skipped += 1
                    continue

                self.docs_read += 1
                yield doc

    def _parse_line(self, line: str, line_num: int) -> Document | None:
        """
        Parseia uma linha JSON. Retorna None se a linha eh invalida.
        Loga warning no stderr para linhas puladas (sem matar o proc).
        """
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(
                f"[corpus_reader] linha {line_num}: JSON invalido ({e})",
                file=sys.stderr,
            )
            return None

        # Verifica que eh um dict com pelo menos 'id'.
        # title/text/keywords sao opcionais (corpus pode ter docs com
        # campos faltando; preferimos indexar o que tem a pular).
        if not isinstance(obj, dict):
            print(
                f"[corpus_reader] linha {line_num}: nao eh JSON object",
                file=sys.stderr,
            )
            return None

        doc_id = obj.get("id")
        if doc_id is None:
            print(
                f"[corpus_reader] linha {line_num}: doc sem 'id'",
                file=sys.stderr,
            )
            return None

        # Coercao defensiva: alguns datasets podem ter id como int.
        doc_id = str(doc_id)

        title = obj.get("title") or ""
        text = obj.get("text") or ""
        keywords = obj.get("keywords") or []

        # Defesa contra estruturas inesperadas.
        if not isinstance(keywords, list):
            keywords = []

        return Document(
            id=doc_id,
            title=str(title),
            text=str(text),
            keywords=[str(k) for k in keywords],
        )
