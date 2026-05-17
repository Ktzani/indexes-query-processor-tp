"""
Constantes do processo de indexacao.
"""

# Numero de threads do pool de processadores (parsing + normalizacao).
# Valor refinado empiricamente. Sera ajustado apos experimento.
NUM_PROCESSOR_THREADS = 4

# Tamanho do batch de documentos lidos do JSONL antes de serem
# colocados na fila do pipeline. Batches maiores reduzem overhead de
# sincronizacao da queue, mas aumentam pico de memoria.
READER_BATCH_SIZE = 500

# Tamanho maximo da fila entre Reader e Processors (em batches). Limita
# o pico de memoria caso processors fiquem mais lentos que reader.
QUEUE_MAX_BATCHES = 8

# Fracao do orcamento de memoria que pode ser consumida ANTES de
# disparar flush do PartialIndex. Deixamos margem de seguranca para
# overheads (buffers de I/O, garbage collector, etc.).
MEMORY_FLUSH_THRESHOLD = 0.75

# Periodicidade (em documentos processados) com que a memoria eh
# verificada. Verificar a cada documento eh caro (psutil tem custo);
# verificar a cada N=500 eh suficiente.
MEMORY_CHECK_EVERY_DOCS = 500

# Diretorio de blocos parciais (sob o INDEX dir).
BLOCKS_DIR_NAME = "blocks"

# Nomes dos arquivos finais do indice.
INVERTED_INDEX_FILENAME = "inverted.idx"
TERM_LEXICON_FILENAME = "lexicon.pkl"
DOCUMENT_INDEX_FILENAME = "doc_index.pkl"

# Encoding usado em arquivos de texto (jsonl, queries).
TEXT_ENCODING = "utf-8"
