from src.preprocessing.nltk_setup import ensure_nltk_data
ensure_nltk_data()

from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer

text = "!!! is a dance-punk band that formed in Sacramento, California, in 1996."

tokenizer = Tokenizer()
normalizer = Normalizer()

tokens = tokenizer.tokenize(text)
terms = normalizer.normalize(tokens)

print("TOKENS:", tokens)
print("TERMS:", terms)