from src.preprocessing.nltk_setup import ensure_nltk_data
ensure_nltk_data()

from src.indexing.spimi import SPIMIOrchestrator
from src.utils.memory import MemoryMonitor

memory = MemoryMonitor(budget_mb=512)
orchestrator = SPIMIOrchestrator(
    corpus_path='samples/sample_corpus.jsonl',
    blocks_dir='./test_blocks/',
    memory=memory,
    num_threads=2,
)
blocks, doc_index = orchestrator.run()
print(f'{len(blocks)} blocks, {len(doc_index)} docs')