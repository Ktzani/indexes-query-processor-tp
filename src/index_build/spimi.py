"""
SPIMI Orchestrator.

Orquestra o pipeline de indexacao parcial:
- 1 thread Reader le o JSONL em batches e atribui doc_ids sequenciais
- N threads Processor consomem batches: tokenize -> normalize -> add ao PartialIndex
- A thread principal monitora memoria e dispara flushes do PartialIndex

Saida: lista de paths de blocks parciais ordenados por termo, e um
mapping doc_id_int -> {original_id, length} (vira o document index).

Sincronizacao:
    _pi_lock        serializa acesso ao PartialIndex (writes) e ao
                    contador _docs_since_check.
    _flush_event    quando setado, processors pausam apos terminar o
                    batch corrente. Main faz flush, depois reseta o
                    event.
    _doc_index_lock serializa adicao ao mapping doc_id_int -> metadados.
    _done_event     setado pelo Reader quando termina; processors
                    saem quando queue.empty() e este event setado.
"""

import os
import queue
import sys
import threading
import time

from src.config.indexer import (
    BLOCKS_DIR_NAME,
    MEMORY_CHECK_EVERY_DOCS,
    NUM_PROCESSOR_THREADS,
    QUEUE_MAX_BATCHES,
    READER_BATCH_SIZE,
)
from src.index_build.corpus_reader import CorpusReader
from src.index_build.partial_index import PartialIndex
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer
from src.utils.memory import MemoryMonitor


# Sentinela colocada na queue quando o Reader termina, para que
# Processors saibam que nao havera mais batches.
_END_OF_INPUT = object()


class SPIMIOrchestrator:
    """Coordena o pipeline de indexacao parcial."""

    def __init__(
        self,
        corpus_path: str,
        blocks_dir: str,
        memory: MemoryMonitor,
        num_threads: int = NUM_PROCESSOR_THREADS,
        max_docs: int | None = None,
    ):
        self._corpus_path = corpus_path
        self._blocks_dir = blocks_dir
        self._memory = memory
        self._num_threads = num_threads
        self._max_docs = max_docs

        os.makedirs(self._blocks_dir, exist_ok=True)

        # Estruturas compartilhadas
        self._pi = PartialIndex()
        self._pi_lock = threading.Lock()
        self._doc_index: dict[int, dict] = {}
        self._doc_index_lock = threading.Lock()

        # Sincronizacao
        self._batch_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_BATCHES)
        self._reader_done = threading.Event()
        self._flush_event = threading.Event()
        self._flush_done = threading.Event()
        self._stop_event = threading.Event()  # erro fatal: aborta tudo

        # Estado do flush controlado pela main
        self._block_paths: list[str] = []
        self._next_block_idx = 0

        # Contadores para logs
        self._docs_processed = 0
        self._docs_processed_lock = threading.Lock()

    # =====================================================================
    # API publica
    # =====================================================================

    def run(self) -> tuple[list[str], dict[int, dict]]:
        """
        Executa o pipeline ate o fim. Retorna (block_paths, doc_index).
        """
        # Inicia Reader
        reader_thread = threading.Thread(
            target=self._reader_loop, name="reader", daemon=True
        )
        reader_thread.start()

        # Inicia Processors
        processors = []
        for i in range(self._num_threads):
            t = threading.Thread(
                target=self._processor_loop, name=f"processor-{i}", daemon=True
            )
            t.start()
            processors.append(t)

        # Main loop: monitora memoria e coordena flushes
        self._main_loop(reader_thread, processors)

        # Aguarda todos terminarem
        reader_thread.join(timeout=5.0)
        for t in processors:
            t.join(timeout=5.0)

        # Flush final se ainda houver dados em memoria
        if not self._pi.is_empty():
            self._do_flush(final=True)

        return self._block_paths, self._doc_index

    # =====================================================================
    # Threads
    # =====================================================================

    def _reader_loop(self):
        """
        Le o JSONL e enfileira batches. Atribui doc_ids monotonicos a
        cada documento, na ordem em que aparecem no JSONL.
        """
        reader = CorpusReader(self._corpus_path, max_docs=self._max_docs)
        batch: list[tuple[int, object]] = []
        next_id = 0

        try:
            for doc in reader:
                if self._stop_event.is_set():
                    break
                batch.append((next_id, doc))
                next_id += 1
                if len(batch) >= READER_BATCH_SIZE:
                    self._enqueue_batch_blocking(batch)
                    batch = []
            # Sobra
            if batch and not self._stop_event.is_set():
                self._enqueue_batch_blocking(batch)
        except Exception as e:
            print(f"[reader] erro fatal: {e}", file=sys.stderr)
            self._stop_event.set()
        finally:
            # Sinaliza fim para os processors
            self._reader_done.set()
            # Enche a queue com sentinelas (uma por thread) para que
            # processors bloqueados em queue.get() acordem.
            for _ in range(self._num_threads):
                try:
                    self._batch_queue.put(_END_OF_INPUT, timeout=1.0)
                except queue.Full:
                    # Em caso patologico, processors veem reader_done +
                    # queue.empty() e saem.
                    pass

    def _enqueue_batch_blocking(self, batch: list[tuple[int, object]]):
        """Coloca um batch na queue, bloqueando se cheia."""
        while not self._stop_event.is_set():
            try:
                self._batch_queue.put(batch, timeout=0.5)
                return
            except queue.Full:
                continue

    def _processor_loop(self):
        """
        Consome batches da queue: tokeniza, normaliza, adiciona ao PI.
        Cada processor tem seu proprio Tokenizer/Normalizer (sao
        stateless apos init, mas evitamos contencao).
        """
        tokenizer = Tokenizer()
        normalizer = Normalizer()

        while not self._stop_event.is_set():
            # Aguarda flush se houver
            if self._flush_event.is_set():
                self._flush_done.wait(timeout=0.5)
                continue

            try:
                item = self._batch_queue.get(timeout=0.5)
            except queue.Empty:
                # Se o reader terminou e a queue esvaziou, sai
                if self._reader_done.is_set() and self._batch_queue.empty():
                    return
                continue

            if item is _END_OF_INPUT:
                return

            batch = item
            for internal_id, doc in batch:
                if self._stop_event.is_set():
                    return

                # Tokeniza/normaliza FORA do lock (parte cara)
                content = doc.full_content()
                tokens = tokenizer.tokenize(content)
                terms = normalizer.normalize(tokens)
                length = len(terms)

                # Registra no document index
                with self._doc_index_lock:
                    self._doc_index[internal_id] = {
                        "original_id": doc.id,
                        "length": length,
                    }

                # Atualiza PI sob lock
                with self._pi_lock:
                    self._pi.add_document(internal_id, terms)

                # Contador de docs processados (para checagens de mem)
                with self._docs_processed_lock:
                    self._docs_processed += 1

    def _main_loop(
        self,
        reader_thread: threading.Thread,
        processors: list[threading.Thread],
    ):
        """
        Thread principal: roda em loop verificando memoria e disparando
        flushes. Sai quando todos os processors terminam.
        """
        last_log_time = time.time()
        last_logged_docs = 0

        while True:
            # Log periodico de progresso (a cada ~10s)
            now = time.time()
            if now - last_log_time >= 10.0:
                with self._docs_processed_lock:
                    cur = self._docs_processed
                rate = (cur - last_logged_docs) / (now - last_log_time)
                used = self._memory.used_mb()
                print(
                    f"[spimi] {cur} docs, {rate:.0f} docs/s, "
                    f"mem={used:.0f}MB ({self._memory.used_ratio()*100:.0f}%), "
                    f"blocks={len(self._block_paths)}",
                    file=sys.stderr,
                )
                last_log_time = now
                last_logged_docs = cur

            # Verifica memoria
            if self._memory.should_flush():
                self._do_flush()

            # Condicao de saida: todas as threads terminaram
            all_done = (
                not reader_thread.is_alive()
                and all(not t.is_alive() for t in processors)
            )
            if all_done:
                break

            time.sleep(0.5)

    def _do_flush(self, final: bool = False):
        """
        Despeja o PartialIndex em disco como um novo block.

        Estrategia:
        1. Sinaliza flush_event para pausar processors.
        2. Aguarda processors pararem (drena trabalho em curso).
        3. Dumpa PartialIndex em disco.
        4. Limpa PartialIndex.
        5. Reseta flush_event e libera processors.
        """
        # Sinaliza pausa (processors ja em meio a um batch terminam ele)
        self._flush_event.set()
        self._flush_done.clear()

        # Aguarda processors pararem brevemente. Como add_document eh
        # rapido, threshold de 100ms eh largo o suficiente para a maioria
        # das threads liberarem o lock.
        # (Nao precisamos de barreira rigorosa: o PI lock serializa
        # writes de qualquer forma.)
        time.sleep(0.1)

        with self._pi_lock:
            if self._pi.is_empty():
                # Nada a dumpar; libera processors e retorna.
                self._flush_done.set()
                self._flush_event.clear()
                return

            block_path = os.path.join(
                self._blocks_dir, f"block_{self._next_block_idx:05d}.bin"
            )
            metadata = self._pi.dump_to_disk(block_path)
            self._next_block_idx += 1
            self._block_paths.append(block_path)

            tag = "final" if final else "flush"
            used = self._memory.used_mb()
            print(
                f"[spimi] {tag} -> {os.path.basename(block_path)}: "
                f"{metadata['num_terms']} termos, "
                f"{metadata['num_postings']} postings, "
                f"mem={used:.0f}MB",
                file=sys.stderr,
            )

            self._pi.clear()
            # Forca GC para devolver memoria ao SO/heap o quanto antes.
            # Sem isso, o RSS pode demorar para baixar e disparar novo
            # flush prematuramente.
            import gc
            gc.collect()
            self._memory.reset_peak()

        # Libera processors
        self._flush_done.set()
        self._flush_event.clear()