from src.preprocessing.nltk_setup import ensure_nltk_data
ensure_nltk_data()

from src.indexing.corpus_reader import CorpusReader
from src.indexing.partial_index import PartialIndex
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer

reader = CorpusReader('samples/sample_corpus.jsonl')
tokenizer = Tokenizer()
normalizer = Normalizer()
pi = PartialIndex()

internal_id = 0
for doc in reader:
    tokens = tokenizer.tokenize(doc.full_content())
    terms = normalizer.normalize(tokens)
    pi.add_document(internal_id, terms)
    internal_id += 1

print(f'{pi.num_terms()} termos, {pi.num_postings()} postings, {pi.num_docs()} docs')
pi.dump_to_disk('test_block.bin')
import os
print(f'Block size: {os.path.getsize("test_block.bin")} bytes')
os.remove('test_block.bin')
