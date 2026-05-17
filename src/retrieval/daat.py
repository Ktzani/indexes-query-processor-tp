"""
Conjunctive Document-at-a-Time (DAAT) matching com galloping search.

Algoritmo:
    1. Para cada termo da query, carrega a inverted list e cria um
       Cursor.
    2. Se algum termo nao existe no indice -> retorna [] imediatamente
       (interseccao conjuntiva eh vazia).
    3. Ordena cursors por df crescente (menor lista primeiro). Isso
       minimiza o numero medio de advances necessarios.
    4. Loop principal: usa o doc_id maximo entre os cursors como
       'pivot' e avanca todos os outros ate >= pivot. Se todos
       coincidem no pivot, eh um match.

Otimizacoes implementadas:
    - Curto-circuito quando algum termo nao existe (df=0)
    - Ordenacao das listas por df crescente
    - Galloping search (exponential + binary) para avancos rapidos
      quando o gap entre cursors eh grande
    - Metricas de quanto se economizou em postings lidas (saving ratio)

Por que galloping em vez de linear advance:
    Listas curtas em queries conjuntivas geralmente forcam saltos
    grandes na lista longa (poucos matches significam grandes gaps).
    Linear scan = O(gap); galloping = O(log gap). Em queries com
    listas de tamanhos muito diferentes (caso comum), o ganho eh
    substancial.

Thread-safety: nao eh. Cada thread deve criar sua propria instancia.
"""

from dataclasses import dataclass

from src.index_store.inverted_index import InvertedIndex
from src.index_store.posting import Posting
from src.index_store.term_lexicon import TermLexicon


@dataclass
class DAATResult:
    """Resultado de uma operacao de matching conjuntivo."""

    matched_doc_ids: list[int]
    # Postings retornadas (todas as postings dos termos que matcheiam,
    # uma por (termo, doc_match)). O scorer precisa delas pro TF.
    matched_postings: dict[int, dict[str, Posting]]  # doc_id -> {term -> Posting}

    # Metricas
    postings_scanned: int        # postings efetivamente examinadas
    postings_in_lists: int       # tamanho total das listas (sum of df)
    early_terminated: bool       # true se abortou cedo (curto-circuito)

    @property
    def skip_ratio(self) -> float:
        """Fracao de postings 'economizadas' vs linear scan completo."""
        if self.postings_in_lists == 0:
            return 0.0
        return 1.0 - (self.postings_scanned / self.postings_in_lists)


class _Cursor:
    """
    Cursor sobre uma inverted list (lista de postings ja carregadas
    em memoria). Mantem posicao corrente e suporta galloping search
    para avanco rapido.

    A metrica 'scanned' conta quantas POSICOES DISTINTAS foram
    examinadas durante a vida do cursor. Cada posicao eh contada
    uma unica vez, mesmo se o cursor passar varias vezes por ela
    (durante o binary search). Esse eh o numero relevante para
    comparar com 'postings_in_lists' (tamanho total das listas).
    """

    def __init__(self, term: str, postings: list[Posting]):
        self.term = term
        self.postings = postings
        self.pos = 0
        # Conjunto de posicoes visitadas (para metrica). Usamos set
        # para deduplicar: binary search pode revisitar posicoes ja
        # contadas durante o galloping.
        self._visited: set[int] = set()
        if postings:
            self._visited.add(0)

    @property
    def current_doc_id(self) -> int:
        """doc_id na posicao corrente. Assume nao exhausted."""
        return self.postings[self.pos].doc_id

    @property
    def current_posting(self) -> Posting:
        """Posting na posicao corrente."""
        return self.postings[self.pos]

    @property
    def scanned(self) -> int:
        """Numero de posicoes distintas visitadas."""
        return len(self._visited)

    def is_exhausted(self) -> bool:
        return self.pos >= len(self.postings)

    def advance_to(self, target: int):
        """
        Avanca a posicao ate o primeiro doc_id >= target. Usa
        galloping search (exponential probing + binary search).

        Se nenhuma posicao tem doc_id >= target, o cursor fica
        exhausted (pos = len(postings)).
        """
        if self.is_exhausted():
            return

        # Ja esta no target ou alem? Nada a fazer.
        if self.postings[self.pos].doc_id >= target:
            return

        # Fase 1: galloping (exponencial). Comeca com salto 1 e dobra.
        lo = self.pos
        jump = 1
        hi = lo + jump
        n = len(self.postings)

        # Avanca enquanto nao passou do target e nao estourou o final
        while hi < n and self.postings[hi].doc_id < target:
            self._visited.add(hi)
            lo = hi
            jump *= 2
            hi = lo + jump

        # hi pode ter passado do final da lista
        hi = min(hi, n - 1)
        self._visited.add(hi)

        # Se ate o final da lista nao alcancou o target, exhausted
        if self.postings[hi].doc_id < target:
            self.pos = n
            return

        # Fase 2: binary search no intervalo [lo+1, hi]
        # (lo tem doc_id < target; hi tem doc_id >= target)
        left = lo + 1
        right = hi
        while left < right:
            mid = (left + right) // 2
            self._visited.add(mid)
            if self.postings[mid].doc_id < target:
                left = mid + 1
            else:
                right = mid
        # left == right e postings[left].doc_id >= target
        self.pos = left
        self._visited.add(self.pos)

    def advance_one(self):
        """Avanca uma posicao (usado apos match para evitar reprocess)."""
        self.pos += 1
        if not self.is_exhausted():
            self._visited.add(self.pos)


class ConjunctiveDAAT:
    """
    Executa matching conjuntivo (AND) entre as listas de varios termos.

    Pre-carrega todas as listas em memoria antes de processar. Para o
    corpus deste trabalho (4.6M docs), mesmo termos populares teriam
    listas de poucos MB, entao o trade-off favorece simplicidade.
    """

    def __init__(self, lexicon: TermLexicon, inverted_index: InvertedIndex):
        self._lexicon = lexicon
        self._ii = inverted_index

    def intersect(self, terms: list[str]) -> DAATResult:
        """
        Encontra todos os documentos que contem TODOS os termos.

        Retorna um DAATResult com:
            matched_doc_ids: docs que casam, em ordem crescente
            matched_postings: para cada doc_match, dict term -> Posting
                              (necessario para o scorer calcular TF*IDF)
            postings_scanned: numero de postings examinadas
            postings_in_lists: numero total de postings nas listas
                               (= soma dos df dos termos)
            early_terminated: true se abortou cedo
        """
        # Caso trivial: query vazia
        if not terms:
            return DAATResult(
                matched_doc_ids=[],
                matched_postings={},
                postings_scanned=0,
                postings_in_lists=0,
                early_terminated=False,
            )

        # Verifica que TODOS os termos existem no lexicon. Se algum
        # nao existe, interseccao eh vazia (curto-circuito).
        term_entries: list[tuple[str, int, int]] = []  # (term, offset, df)
        for term in terms:
            entry = self._lexicon.get_entry(term)
            if entry is None:
                # Termo nao existe -> sem matches possiveis
                return DAATResult(
                    matched_doc_ids=[],
                    matched_postings={},
                    postings_scanned=0,
                    postings_in_lists=0,
                    early_terminated=True,
                )
            offset, df = entry
            term_entries.append((term, offset, df))

        # Ordena por df crescente para minimizar trabalho.
        # A menor lista controla o numero de iteracoes do loop principal.
        term_entries.sort(key=lambda x: x[2])

        # Carrega as listas e cria cursors.
        cursors: list[_Cursor] = []
        total_postings_in_lists = 0
        for term, offset, df in term_entries:
            postings = self._ii.read_postings(offset, df)
            cursors.append(_Cursor(term, postings))
            total_postings_in_lists += df

        matched_doc_ids: list[int] = []
        matched_postings: dict[int, dict[str, Posting]] = {}

        # Loop principal
        while True:
            # Verifica exhausted: se qualquer cursor esgotou, fim
            if any(c.is_exhausted() for c in cursors):
                break

            # pivot = maior doc_id entre todos os cursors
            pivot = max(c.current_doc_id for c in cursors)

            # Avanca cada cursor ate >= pivot
            for c in cursors:
                if c.current_doc_id < pivot:
                    c.advance_to(pivot)

            # Se algum cursor esgotou apos os avancos, fim
            if any(c.is_exhausted() for c in cursors):
                break

            # Se todos os cursors agora estao no pivot, eh match
            if all(c.current_doc_id == pivot for c in cursors):
                matched_doc_ids.append(pivot)
                # Coleta as postings de cada termo para o doc match
                postings_for_doc = {}
                for c in cursors:
                    postings_for_doc[c.term] = c.current_posting
                matched_postings[pivot] = postings_for_doc
                # Avanca todos
                for c in cursors:
                    c.advance_one()

        # Soma de postings_scanned de todos os cursors
        total_scanned = sum(c.scanned for c in cursors)

        return DAATResult(
            matched_doc_ids=matched_doc_ids,
            matched_postings=matched_postings,
            postings_scanned=total_scanned,
            postings_in_lists=total_postings_in_lists,
            early_terminated=False,
        )
