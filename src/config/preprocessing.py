"""
Constantes de pre-processamento usadas tanto pelo indexer quanto pelo
processor. CRITICO: indexer e processor DEVEM usar os mesmos valores
aqui, senao queries nao casam com documentos.
"""

# Idioma do corpus (Wikipedia inglesa). Usado pelo SnowballStemmer e
# pela lista de stopwords do NLTK.
STEMMER_LANGUAGE = "english"

# Comprimento minimo de token apos normalizacao. Filtra ruido (letras
# soltas, restos de tokenizacao).
MIN_TOKEN_LENGTH = 2

# Comprimento maximo de token. Tokens absurdamente longos (e.g.,
# strings de hash, lixo de HTML que escapou) sao descartados.
MAX_TOKEN_LENGTH = 40

# Se True, descarta tokens que sao puramente numericos.
# Indexar numeros gigantescos polui o lexicon sem benefico para o
# tipo de queries esperadas em entidades.
DROP_PURE_NUMERIC = True
