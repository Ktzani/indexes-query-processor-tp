# Programming Assignment #2 - Indexer and Query Processor

Information Retrieval - UFMG
Gabriel Catizani Faria Oliveira

## Descricao

Implementacao de um indexador invertido SPIMI com merge externo e um query
processor com DAAT conjunctive matching, suportando rankings TF-IDF e BM25.

## Setup

```bash
python3 -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Na primeira execucao, dados do NLTK (punkt e stopwords) sao baixados
automaticamente.

## Uso

### Indexador

```bash
python indexer.py -m 1024 -c corpus.jsonl -i index/
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
python processor.py -i data\indexes\index_full3\ -q data\queries\real.txt -r BM25 > data\results\real_bm25.jsonl 2> data\results\real_bm25.log   
```

e

```bash
python processor.py -i data\indexes\index_full3\ -q data\queries\real.txt -r TFIDF > data\results\real_tfidf.jsonl 2> data\results\real_tfidf.log   
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
├── script/
└── tests/
```

---

## Como funciona o codigo (passo a passo)

### Indexer

**Entrada:** JSONL no formato:
```json
{"id": "3442552", "title": "OK Computer", "text": "OK Computer is the third album by Radiohead..."}
```

**Saida:** tres arquivos no diretorio `-i`:

| Arquivo         | Formato | Conteudo                                              |
|-----------------|---------|-------------------------------------------------------|
| `inverted.idx`  | binario | `[num_postings: u32][(doc_id, tf) x N]` por termo     |
| `lexicon.pkl`   | pickle  | `dict[term -> (offset, df)]`                          |
| `doc_index.pkl` | pickle  | `(original_ids: list[str], lengths: array)`           |

**Pipeline:**

```
       ┌──────────┐
       │ Reader   │  (1 thread)
       │ thread   │  Lê JSONL e atribui internal_id
       └────┬─────┘
            │
            ▼  enfileira batches de 500 docs
       ┌─────────┐
       │  Queue  │  ← capacidade limitada (8 batches)
       └────┬────┘
            │
   ┌────────┼────────┬────────┐
   ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Proc │ │ Proc │ │ Proc │ │ Proc │  (4 threads)
│  #0  │ │  #1  │ │  #2  │ │  #3  │
└───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘
    └────────┼────────┼────────┘
             ▼        ▼
        ┌──────────────────┐
        │   PartialIndex   │  ← compartilhado, protegido por lock
        └──────────────────┘
             ▲
             │  monitora RSS
        ┌────┴─────┐
        │ Main /   │  (1 thread)
        │ Monitor  │  Decide quando flush
        └──────────┘
```

1. **Reader** le o JSONL em batches de 500 documentos, atribui um
   `internal_id` monotonico (0, 1, 2, ...) e enfileira numa Queue.

2. **Processors (4 threads)** consomem batches em paralelo:
   - Tokenizam com `nltk.word_tokenize` e normalizam (stopwords + Snowball
     stemmer + filtros de tamanho/numericos).
   - Exemplo: `"OK Computer is the third album..."` ->
     `["ok", "comput", "third", "album", "radiohead", ...]`
   - Atualizam o `PartialIndex` em memoria com `(termo, doc_id, tf)`.

3. **MemoryMonitor** verifica RSS a cada 500 docs. Ao atingir
   **75% do orcamento**, sinaliza flush.

4. **Flush:** o PartialIndex é ordenado e gravado como `block_NNN.bin`
   (binario little-endian, 8 bytes por posting). `gc.collect()` libera
   memoria e o pipeline continua.

5. **K-way merge externo:** depois de processar todos os docs, abre N
   cursores (um por bloco), mantem min-heap por termo e consolida tudo
   no `inverted.idx` final, registrando offsets no lexicon.

6. **Persiste** lexicon e doc_index como pickles.

**Exemplo de saida (stdout):**
```json
{
  "Index Size": 1135.794,
  "Elapsed Time": 3349.83,
  "Number of Lists": 3888938,
  "Average List Size": 33.0
}
```

### Processor

**Entrada:** indice + arquivo de queries (uma por linha) + ranker.

**Saida:** JSON em stdout, um por linha, com top-10 por query.

**Pipeline:**

1. **Carrega lexicon e doc_index na RAM** (uma vez). `inverted.idx` fica em
   disco, acessado sob demanda via `seek`.

2. **Pre-processa a query** com o mesmo pipeline do indexer.
   - Exemplo: `"radiohead albums"` -> `["radiohead", "album"]`

3. **Curto-circuito:** se algum termo nao esta no lexicon, retorna
   `Results: []` sem tocar o `inverted.idx`.

4. **Carrega postings** via offset:
   - `lexicon["radiohead"]` -> `(offset=812445930, df=5)`
   - `seek` no arquivo e le 5 pares `(doc_id, tf)`.

5. **DAAT conjuntivo com galloping search**:
   - Ordena cursores por `df` crescente (termo raro guia o avanco).
   - Para cada candidato, avanca os outros cursors com saltos
     `1, 2, 4, 8, ...` ate superar o alvo, depois bissecao
     (`O(log d)` por avanco).
   - Coleta doc_ids presentes em **todas** as listas.

6. **Score** (TF-IDF lnc.ltn ou BM25 com IDF BM25+ Lucene,
   `k1=1.2`, `b=0.75`).

7. **Top-K** via min-heap de tamanho 10 (`O(N log K)`).
   Tie-break por `doc_id` menor (determinismo).

8. **Traduz `internal_id` -> `original_id`** via doc_index e emite JSON.

**Exemplo de saida:**
```json
{
  "Query": "radiohead albums",
  "Results": [
    {"ID": "3442552", "Score": 22.6312},
    {"ID": "3442549", "Score": 21.6043},
    {"ID": "4112215", "Score": 21.4521}
  ]
}
```

---

## Configuracoes

Constantes em `src/config/`:
- `indexer.py`: `NUM_PROCESSOR_THREADS=4`, `READER_BATCH_SIZE=500`,
  `QUEUE_MAX_BATCHES=8`, `MEMORY_FLUSH_THRESHOLD=0.75`
- `preprocessing.py`: `STEMMER_LANGUAGE="english"`, `MIN_TOKEN_LENGTH=2`,
  `MAX_TOKEN_LENGTH=40`, `DROP_PURE_NUMERIC=True`
- `processor.py`: `TOP_K=10`, `BM25_K1=1.2`, `BM25_B=0.75`

**CRITICO:** indexer e processor compartilham `preprocessing.py` - qualquer
divergencia de parametro faz queries deixarem de casar com os documentos.

## Estatisticas tipicas (corpus completo)

- Documentos indexados: 4.641.784
- Termos unicos no lexicon: 3.888.938
- Tamanho do `inverted.idx`: 1135.79 MB
- Tempo de indexacao: 55.8 min (4 threads, `-m 1024`)
- Memoria pico: 770 MB (75% do orcamento)
- Latencia media por query: ~313 ms