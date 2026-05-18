# Programming Assignment #2 - Indexer and Query Processor

Information Retrieval - UFMG  
Gabriel Catizani Faria Oliveira

## Descricao

Implementacao de um indexador invertido SPIMI com merge externo e um query
processor com DAAT conjunctive matching, suportando rankings TF-IDF e BM25.

## Setup

```bash
python3 -m venv pa2
# Windows
.\pa2\Scripts\Activate.ps1
# Linux/Mac
source pa2/bin/activate

pip install -r requirements.txt
```

Na primeira execucao, dados do NLTK (punkt e stopwords) sao baixados
automaticamente.

## Uso

### Indexador

```bash
python indexer.py -m 1024 -c corpus.jsonl -i index/ 2>&1 | Tee-Object -FilePath indexer_log.txt
```

O que rodei:
```bash
python indexer.py -m 1024 -c data\corpus\entities.jsonl -i data\indexes\index_full3\ 2>&1 | Tee-Object -FilePath data\indexes\index_full3_log.txt
```

Argumentos:
- `-m <MEMORY>`: memoria maxima disponivel em MB
- `-c <CORPUS>`: caminho do arquivo JSONL do corpus
- `-i <INDEX>`: diretorio onde os indices serao salvos

Output: JSON em stdout com estatisticas do indice.

### Query Processor

```bash
python processor.py -i index/ -q queries.txt -r BM25 2>&1 | Tee-Object -FilePath processor_log.txt        
```

O que rodei: 
```bash
python processor.py -i data\indexes\index_full2\ -q data\queries\real.txt -r BM25 > data\results\real_tfidf.jsonl 2> data\results\real_tfidf.log   
```  

Argumentos:
- `-i <INDEX>`: caminho do diretorio com os indices
- `-q <QUERIES>`: arquivo com as queries (uma por linha)
- `-r <RANKER>`: TFIDF ou BM25

Output: um JSON em stdout por query, com os top 10 resultados.

## Estrutura do Projeto

```
pa2-indexer/
├── indexer.py              # entry point indexer
├── processor.py            # entry point query processor
├── src/
│   ├── config/             # constantes
│   ├── preprocessing/      # tokenizer, normalizer
│   ├── index_build/        # SPIMI, partial index, merger
│   ├── index_store/        # estruturas em disco
│   ├── retrieval/          # query, DAAT, scorer, ranker
│   └── utils/              # memoria, I/O
└── samples/                # corpus/queries de exemplo
```

ARQUITETURA:
                   ┌──────────────┐
                   │    Main      │
                   │ (memoria,    │
                   │  flush)      │
                   └──────┬───────┘
                          │
       ┌─────────┐        │        ┌────────────────┐
       │ Reader  │──queue▶│        │ PartialIndex   │
       │ thread  │        │        │ (lock externo) │
       └─────────┘        │        └────────────────┘
                          │              ▲
              ┌───────────┴─────────────┐│
              ▼           ▼             ▼│
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │Processor │ │Processor │ │Processor │
       │  #0      │ │  #1      │ │  #N-1    │
       └──────────┘ └──────────┘ └──────────┘