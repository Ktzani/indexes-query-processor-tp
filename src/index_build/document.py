"""
Representa um documento do corpus, ja parseado e pronto para
indexacao.

Decisao: combinamos title, text e keywords em um unico campo textual
('content') durante a indexacao. Isso simplifica o pipeline e eh
suficiente para o escopo deste trabalho (sem boost de campo).

Mantemos os 4 campos originais (id, title, text, keywords) no
dataclass por dois motivos:
1. Facilita debug e inspecao do corpus.
2. Permite mudar a estrategia depois (ex: dar peso maior pro title)
   sem mexer no CorpusReader.
"""

from dataclasses import dataclass, field


@dataclass
class Document:
    """Documento do corpus apos parsing do JSON."""

    id: str                          # ex: "0000001"
    title: str                       # ex: "!!!"
    text: str                        # corpo descritivo
    keywords: list[str] = field(default_factory=list)

    def full_content(self) -> str:
        """
        Retorna a concatenacao de title + text + keywords como uma
        unica string. Eh o conteudo que sera tokenizado e indexado.

        Junta com espaco entre campos para garantir que o tokenizer
        nao concatene tokens que pertencem a campos diferentes.
        """
        parts = []
        if self.title:
            parts.append(self.title)
        if self.text:
            parts.append(self.text)
        if self.keywords:
            parts.extend(self.keywords)
        return " ".join(parts)
