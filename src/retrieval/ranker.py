"""
Ranker: seleciona os top-K documentos por score usando min-heap.

Algoritmo:
    Mantem uma min-heap de tamanho fixo K com (score, doc_id).
    - Se a heap tem menos de K elementos, adiciona.
    - Se ja tem K e o novo score eh maior que o menor da heap,
      substitui o menor pelo novo (heappushpop).
    - Senao, ignora (esse doc nao entra no top-K).

    Ao final, extrai os K elementos em ordem decrescente de score.

Complexidade:
    Inserir N candidatos: O(N log K)
    Extrair top-K ordenado: O(K log K)
    Memoria: O(K)

Por que min-heap (e nao max-heap)?
    Queremos os MAIORES K scores. O 'gate' pro top-K eh o menor
    score atualmente na heap. Se o menor sobe, candidatos abaixo
    sao descartados. Min-heap eh perfeita para isso: heap[0] eh
    o menor elemento, comparacao O(1) e remocao O(log K).

Decisao: ranker eh stateless e generico. Recebe lista de
(doc_id, score) e retorna a lista ordenada de top-K. Quem chama
(processor.py) faz a traducao para o JSON de output com original_id.

Empate de scores: docs com mesmo score sao ordenados por doc_id
crescente (estavel, deterministico). Importante para reprodutibilidade
nos testes.
"""

import heapq

from src.config.processor import TOP_K


class Ranker:
    """Seleciona o top-K de candidatos por score usando min-heap."""

    def __init__(self, k: int = TOP_K):
        if k <= 0:
            raise ValueError(f"k deve ser positivo, recebido {k}")
        self._k = k

    def top_k(
        self,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Seleciona os k candidatos com maior score.

        Parametros:
            candidates: lista de (doc_id, score)

        Retorna:
            lista de (doc_id, score) ordenada por score DECRESCENTE.
            Em caso de empate, ordena por doc_id crescente.
        """
        if not candidates:
            return []

        # Min-heap de (score, -doc_id). Negamos doc_id para que, em
        # empate de score, a heap considere doc_id MAIOR como "menor"
        # (e portanto mais facil de descartar). Resultado: em empate,
        # mantemos doc_ids MENORES no top-K, comportamento determi-
        # nistico desejavel.
        heap: list[tuple[float, int]] = []

        for doc_id, score in candidates:
            if len(heap) < self._k:
                heapq.heappush(heap, (score, -doc_id))
            elif score > heap[0][0] or (
                score == heap[0][0] and -doc_id > heap[0][1]
            ):
                # Score maior, OU score igual mas doc_id menor (=> -doc_id maior).
                # Na segunda condicao, queremos MANTER o doc_id menor, entao
                # substituimos o atual (que tem doc_id maior) se -doc_id_novo
                # > -doc_id_atual (i.e., doc_id_novo < doc_id_atual).
                heapq.heappushpop(heap, (score, -doc_id))

        # Extrai e ordena: score desc, doc_id asc
        result = [(-neg_doc_id, score) for score, neg_doc_id in heap]
        result.sort(key=lambda x: (-x[1], x[0]))
        return result
