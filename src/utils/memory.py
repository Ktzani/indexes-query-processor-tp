"""
Monitoramento de memoria via psutil.

CRITICO: o enunciado exige que o indexer respeite o orcamento de
memoria informado via -m. Excesso pode ser killado por OOM. Este modulo
centraliza a leitura do RSS (Resident Set Size) do processo, que eh a
medida que o SO usa para decidir OOM.

API publica:
    MemoryMonitor(budget_mb): monitor com orcamento configurado
    .used_mb()            -> float, RSS atual em MB
    .used_ratio()         -> float em [0, ∞], fracao do orcamento usada
    .should_flush()       -> bool, se passou do threshold de flush
    .reset_peak()         -> reseta o pico observado (apos flush)
    .peak_mb()            -> pico observado desde o ultimo reset
"""

import os
import psutil

from src.config.indexer import MEMORY_FLUSH_THRESHOLD


class MemoryMonitor:
    """Wrapper de psutil com helpers especificos do indexer."""

    def __init__(self, budget_mb: int):
        if budget_mb <= 0:
            raise ValueError(f"budget_mb deve ser positivo, recebido {budget_mb}")
        self._budget_mb = budget_mb
        self._process = psutil.Process(os.getpid())
        self._peak_mb = 0.0

    @property
    def budget_mb(self) -> int:
        return self._budget_mb

    def used_mb(self) -> float:
        """Retorna o RSS atual do processo em MB."""
        rss = self._process.memory_info().rss
        mb = rss / (1024.0 * 1024.0)
        if mb > self._peak_mb:
            self._peak_mb = mb
        return mb

    def used_ratio(self) -> float:
        """Fracao do orcamento atualmente em uso (0.0 a 1.0+)."""
        return self.used_mb() / self._budget_mb

    def should_flush(self) -> bool:
        """True se o uso passou do threshold configurado de flush."""
        return self.used_ratio() >= MEMORY_FLUSH_THRESHOLD

    def peak_mb(self) -> float:
        """Pico observado desde o ultimo reset_peak()."""
        # Garante atualizacao do pico antes de reportar
        _ = self.used_mb()
        return self._peak_mb

    def reset_peak(self):
        """Zera o pico (chamado apos um flush, para iniciar nova janela)."""
        self._peak_mb = self.used_mb()
