# contact-center-ai

A Retrieval-Augmented Generation (RAG) pipeline that answers natural-language
questions about operational incidents by retrieving relevant historical records
and generating a grounded response. Rather than letting a language model answer
from its general training alone — where it may fabricate plausible-sounding
details — the pipeline first retrieves real matching documents and then generates
an answer constrained to that evidence.

The project is self-contained: it generates a synthetic corpus of incident logs,
indexes them in a local vector store, and answers questions against them, with an
offline mode that requires no API key and an optional language-model mode for
fully generated responses.

---

## Table of contents

1. [Background: what RAG is and why it matters](#1-background)
2. [Core concepts](#2-core-concepts)
3. [Architecture](#3-architecture)
4. [Repository contents](#4-repository-contents)
5. [Procedure: running the pipeline](#5-procedure)
6. [Configuring the language-model mode](#6-configuring-the-language-model-mode)
7. [A limitation this project surfaces](#7-a-limitation-this-project-surfaces)
8. [Design notes and limitations](#8-design-notes-and-limitations)
9. [Requirements](#9-requirements)
10. [Possible extensions](#10-possible-extensions)

---

## 1. Background

### The problem RAG solves

A large language model's knowledge is fixed at training time and generalized
across a broad corpus. It has no access to a specific organization's private,
current data — a set of incident logs, for example — and when asked about it
directly, the model will often produce a fluent but fabricated answer. This is the
well-known hallucination problem.

**Retrieval-Augmented Generation (RAG)** addresses this by inserting a retrieval
step before generation. Given a question, the system first finds the most relevant
real documents, then asks the language model to answer *using only those documents
as context*. The answer is therefore grounded in actual data rather than in the
model's general priors. This is the standard architecture behind systems that
answer questions over a private document corpus, and is widely adopted in
regulated domains precisely because its answers can be traced back to source
documents.

### What this project is

An end-to-end RAG pipeline over a synthetic corpus of contact-center incident
summaries. It demonstrates the full retrieve-then-generate flow — embedding,
vector storage, similarity search, and grounded answer generation — using
lightweight, inspectable components so that each stage of the architecture is
visible rather than hidden behind a framework.

---

## 2. Core concepts

These concepts underpin the pipeline; reading them first makes the architecture in
section 3 and the code straightforward to follow.

### Vector-space semantics and embeddings

An *embedding* converts text into a list of numbers (a vector) positioned in a
high-dimensional space such that texts with similar meaning sit close together,
measured by cosine similarity or distance. Two families of embedding exist:

- **Statistical embeddings** such as **TF-IDF**, whose notion of similarity is
  based on shared words weighted by rarity. They are simple, fully local, and
  require no model download — but they are blind to meaning beyond exact tokens.
- **Neural embeddings** (for example from sentence-transformer or hosted models),
  trained so that *semantic* similarity, not just word overlap, determines
  closeness. These capture that "outage" and "outages," or "spike" and "surge,"
  are related — something TF-IDF cannot.

This project uses TF-IDF so the whole pipeline runs offline and every step is
transparent; the architecture (section 3) is designed so a neural embedding model
can be substituted with no other changes.

### TF-IDF, precisely

TF-IDF combines **Term Frequency** (how often a word appears in a document) with
**Inverse Document Frequency** (a penalty for words appearing in many documents,
since common words carry little distinguishing information). The result gives the
highest weight to words that are frequent in one document but rare across the
whole collection — which is why this decades-old technique still performs
reasonably for keyword-driven retrieval.

### Vector stores and similarity search

Once documents are embedded as vectors, answering a question means finding the
document vectors closest to the question's vector. A **vector store** (also called
a vector database) is a data store built specifically for this operation: it holds
each document's vector alongside its original text and metadata, and answers
"nearest-neighbour" queries — *return the k vectors most similar to this one* —
efficiently. A conventional database is built for exact matches (find the row where
id = 42); a vector store is built for approximate similarity (find the rows whose
meaning is closest to this query), which is the operation retrieval depends on.

This project uses **ChromaDB**, a local, persistent vector store. At index time
each incident record's TF-IDF vector, its text, and an identifier are added to a
ChromaDB collection. At query time the question is embedded with the same
vectorizer and passed to ChromaDB, which returns the k closest records along with a
distance score for each — a lower distance meaning a closer match. Those retrieved
records become the grounding context for the generation step.

### The retriever/generator split

RAG deliberately separates *finding relevant information* from *phrasing an
answer*. This is the classic software principle of separation of concerns applied
to language systems: the embedding model, the vector store, and the generation
model are independent components, each replaceable without touching the others.
This is why upgrading from TF-IDF to a neural embedding model in this project
requires no change to the vector-store or generation code.

### Grounding and attribution

Because a RAG answer is produced from specific retrieved documents, it can be
traced back to them. This auditability — the ability to show *which* records
informed an answer — is what distinguishes RAG from unconstrained generation, and
is a primary reason regulated fields favor it.

---

## 3. Architecture

```
question
   │
   ▼
[ embed question ]  ──TF-IDF──►  vector
   │
   ▼
[ search vector store ]  ──ChromaDB──►  top-k most similar incident records
   │
   ▼
[ generate answer ]  ──►  grounded response
        │
        ├── extractive mode  (offline, no API key)
        └── language-model mode  (hosted LLM, requires token)
```

| Stage | Component | Role |
|---|---|---|
| Corpus | `generate_data.py` | Produces synthetic incident-log summaries |
| Embedding | `TfidfVectorizer` (scikit-learn) | Converts text to weighted-term vectors |
| Vector store | ChromaDB (local, persistent) | Indexes vectors for fast similarity search |
| Retrieval | ChromaDB query | Returns the k most similar records to a question |
| Generation | extractive or hosted LLM | Produces the final grounded answer |

The two generation modes serve different needs. **Extractive mode** returns the
retrieved records directly and runs fully offline with no credentials — useful for
demonstrating and testing the retrieval half. **Language-model mode** sends the
retrieved context and the question to a hosted model for a fully generated answer,
and requires an API token (section 6).

---

## 4. Repository contents

```
contact-center-ai/
├── generate_data.py      # Generates the synthetic incident-log corpus
├── rag_pipeline.py       # Embedding, vector store, retrieval, generation
├── incident_logs.csv     # Generated corpus (produced by generate_data.py)
├── requirements.txt      # Python dependencies
├── README.md
└── .gitignore
```

A vector-store directory (`chroma_store/`) and a Python virtual environment
(`venv/`) are created at runtime and are excluded from version control. A local
`.env` file, if used for the language-model mode, is also excluded (section 6).

---

## 5. Procedure

The pipeline runs in a Python environment (including WSL on Windows).

### Step 1 — Create an isolated environment and install dependencies

```bash
cd contact-center-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A virtual environment keeps this project's dependencies isolated from other
projects, so package versions cannot conflict.

### Step 2 — Generate the synthetic corpus

```bash
python3 generate_data.py
```

This writes `incident_logs.csv` — a set of synthetic incident summaries (volume
spikes, outages, escalation clusters, staffing gaps) with no real customer or
operational data.

### Step 3 — Run the pipeline

```bash
python3 rag_pipeline.py
```

This embeds the corpus with TF-IDF, indexes it in ChromaDB, and runs a set of
example questions through the retrieve-then-answer flow. In its default extractive
mode it requires no API key and runs fully offline, printing the retrieved
incidents most relevant to each question along with their retrieval distances.

**Sample output (extractive mode).** For the question *"Why did call volume spike
recently?"*, extractive mode returns the retrieved records directly, ordered by
similarity:

```
QUERY: Why did call volume spike recently?
Found 3 related historical incidents:
1. Call volume spiked 64% above forecast following seasonal demand;
   wait times rose to 8 minutes; customers notified via SMS to reduce inbound.
2. Call volume spiked 41% above forecast following a product outage ...
3. Call volume spiked 20% above forecast following a policy change announcement ...

(retrieval distances: [0.71, 0.78, 0.82] — lower is a closer match)
```

Extractive mode performs no generation: it surfaces the evidence the retrieval step
found. This is the fully offline, reproducible path, and it makes the retrieval
quality directly visible through the distance scores.

To use the language-model generation mode instead, configure a token first
(section 6).

---

## 6. Configuring the language-model mode

The language-model generation path sends retrieved context and the question to a
hosted model (via an OpenAI-compatible chat-completions endpoint) and requires an
API token supplied as the `HF_TOKEN` environment variable. Two ways to provide it:

### Option A — `.env` file (persists across sessions)

Create a file named `.env` in the project root containing the token:

```bash
echo 'HF_TOKEN=REPLACE_WITH_TOKEN' > .env
```

The pipeline loads this automatically at startup (via `python-dotenv`), so no
manual export is needed. **The `.env` file is excluded by `.gitignore` and must
never be committed** — it holds a live credential.

### Option B — shell export (session-only)

```bash
export HF_TOKEN="REPLACE_WITH_TOKEN"
python3 rag_pipeline.py
```

This persists only for the current terminal session, leaving no token file on
disk — useful for a one-off run or in a CI environment where the token is injected
as an environment variable.

### Running a generated answer

With a token configured, the language-model mode can be invoked for a single
question:

```bash
python3 -c "
from rag_pipeline import retrieve, generate_with_llm
docs, metas, dist = retrieve('Why did call volume spike recently?', k=3)
print(generate_with_llm('Why did call volume spike recently?', docs))
"
```

**Sample output (language-model mode).** Instead of returning the raw records,
this path sends them to the model as context and produces a synthesized answer
grounded in them:

```
Call volume spiked recently due to a combination of factors including seasonal
demand, a product outage, and a policy change announcement.
```

The distinction between the two modes is the essence of RAG. Extractive mode shows
*what was retrieved*; language-model mode *phrases an answer* from it. Critically,
the generated sentence draws only on the three retrieved incidents — seasonal
demand, a product outage, and a policy change — rather than inventing causes, which
is the grounding property that separates RAG from unconstrained generation.

### A note on secret handling

Whichever method is used, the token is a secret and should be treated as one: kept
out of version control, out of command-line arguments that get logged, and rotated
if ever exposed. Environment variables and ignored `.env` files are the standard
mechanisms for exactly this reason.

---

## 7. A limitation this project surfaces

Because the pipeline uses TF-IDF rather than a neural embedding model, retrieval is
based on shared words rather than shared meaning. A query for "recent system
outages" retrieves weakly against documents that use the singular "outage," because
TF-IDF treats "outage" and "outages" as unrelated tokens — there is no stemming or
semantic understanding. A neural embedding model would recognize these as the same
concept and retrieve correctly.

This is a genuine, reported limitation rather than a hidden flaw, and it makes
concrete *why* production RAG systems use neural embeddings: the retrieval quality
ceiling of a statistical embedding is lower. Because of the retriever/generator
split (section 2), addressing it requires swapping only the embedding step — the
vector store and generation logic are unaffected.

---

## 8. Design notes and limitations

This project runs on synthetic data and uses a lightweight statistical embedding by
default. It demonstrates the RAG architecture end to end — embedding, vector
storage, retrieval, and grounded generation — with each stage inspectable rather
than abstracted away. It is not a production retrieval system: it does not include
neural embeddings, re-ranking, chunking strategies for long documents, or
evaluation of answer faithfulness, each of which a production deployment would add.

The clean separation of embedding, storage, and generation is the central design
choice, and it mirrors how production RAG systems are structured so that each
component can be scaled or replaced independently.

---

## 9. Requirements

- Python 3.10+ (including WSL on Windows).
- Dependencies in `requirements.txt`: pandas, numpy, scikit-learn, chromadb,
  requests, python-dotenv.
- For the language-model mode only: an `HF_TOKEN` and internet access to the
  hosted model endpoint.

---

## 10. Possible extensions

- Replace TF-IDF with a neural sentence-embedding model to enable semantic
  retrieval; no change to the vector store or generation code is required.
- Add re-ranking of retrieved candidates to improve top-k precision.
- Add document chunking so longer records can be retrieved at passage granularity.
- Add citation of the specific retrieved records that informed each generated
  answer, making the grounding explicit.
- Add an evaluation harness measuring answer faithfulness against the retrieved
  context.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
