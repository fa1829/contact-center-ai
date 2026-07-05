# contact-center-ai

Phase 4 of the ML skills-build track (Manulife/RBC Borealis GenAI alignment).
A **Retrieval-Augmented Generation (RAG)** pipeline that answers questions
about contact-center operations by retrieving relevant historical incidents
and generating a grounded answer — instead of an LLM guessing from general
knowledge alone.

## Why RAG, and why this closes a specific job-posting gap
Both Manulife and RBC Borealis postings mention GenAI as a differentiator.
The naive approach — just calling an LLM with a raw question — produces
answers that sound confident but aren't grounded in your actual data (this is
the "hallucination" problem). **RAG fixes this** by first retrieving the most
relevant real documents, then asking the LLM to answer *using only those
documents as context*. This is the standard pattern behind every production
"chat with your data" system.

## What problem is being solved
Imagine an ops analyst asking: *"Why did call volume spike in December?"*
Instead of manually searching through months of incident reports, this
pipeline retrieves the handful of historical incidents most related to that
question and synthesizes a direct answer — grounded in what actually happened,
with the source incidents visible for verification.

## Architecture
```
question → embed → search vector store → top-k similar incidents → LLM (with those incidents as context) → grounded answer
```

| Stage | Tool used here | What it does |
|---|---|---|
| **Knowledge base** | `generate_data.py` | 150 synthetic incident-log summaries (volume spikes, outages, escalation clusters, staffing gaps) |
| **Embedding** | scikit-learn `TfidfVectorizer` | Converts each text document into a numeric vector based on word-importance weighting |
| **Vector store** | ChromaDB (local, persistent) | Stores embeddings + original text + metadata; supports similarity search |
| **Retrieval** | `collection.query()` | Given a new question's embedding, finds the k most similar stored vectors |
| **Generation** | `extractive_summary()` (offline) or `generate_with_llm()` (HuggingFace router) | Turns retrieved incidents into a direct answer |

## Term-by-term glossary
- **RAG (Retrieval-Augmented Generation)**: answer generation that first
  retrieves relevant real documents, then generates a response grounded in
  them — instead of the model answering from what it memorized during training.
- **Embedding**: converting text into a list of numbers (a vector) such that
  similar meanings end up as similar vectors. Real embedding models (like
  those on HuggingFace) capture semantic meaning; **TF-IDF**, used here,
  captures word-overlap importance — simpler, fully local, but blind to
  synonyms and word variants (see limitation below).
- **Vector store**: a database built specifically for fast "find the most
  similar vectors to this one" queries, rather than exact-match lookups like
  a normal database.
- **Cosine distance**: the similarity metric ChromaDB uses by default here —
  lower distance means more similar; a distance near 1.0 means almost no
  meaningful overlap between the query and the retrieved document.

## A real limitation this project surfaced (worth knowing for interviews)
Testing the query *"What caused recent system outages?"* returned weak
matches — all at distance 1.0 (essentially no similarity) — because
**TF-IDF has no stemming**: the query used "outages" (plural) while the
documents used "outage" (singular), and TF-IDF treats these as completely
unrelated tokens. A real embedding model (e.g. a HuggingFace sentence
transformer) would recognize these as semantically identical and retrieve
correctly. This is a genuine, honestly-reported limitation of the
lightweight local approach — not a bug to hide, but the exact reason
production RAG systems use neural embeddings instead of TF-IDF.

**The interview-ready answer**: *"I used TF-IDF for the embedding step so the
whole pipeline runs fully offline with no model download — but testing it
surfaced a real limitation: it missed a plural/singular match that a neural
embedding model would catch. That's the actual tradeoff between a
lightweight local demo and a production-grade embedding model, and I can
point to exactly where and why it failed."* That's a stronger answer than
claiming it worked perfectly.

## Two generation modes
1. **`extractive_summary()`** — offline, no API key, fully testable anywhere.
   Surfaces the retrieved incidents directly with a count. This is what ran
   in the test output below.
2. **`generate_with_llm()`** — calls HuggingFace's router
   (`router.huggingface.co`, OpenAI-compatible chat completions), the same
   pattern used in `fleet-qa-copilot`. Requires an `HF_TOKEN` environment
   variable and internet access — run this in your WSL environment.

## Sample output (from an actual run)
```
QUERY: Why did call volume spike recently?
Found 3 related historical incidents:
1. Call volume spiked 64% above forecast on 2025-06-09 following seasonal
   demand. Wait times increased to 8 minutes. Resolution: Customers were
   proactively notified via SMS to reduce inbound volume.
2. Call volume spiked 41% above forecast on 2025-05-07 following a product
   outage...
3. Call volume spiked 20% above forecast on 2025-06-18 following a policy
   change announcement...
```

## How to run (WSL/Ubuntu)
```bash
cd ~/projects
mkdir contact-center-ai && cd contact-center-ai
# copy in the files from this delivery
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 generate_data.py
python3 rag_pipeline.py
```

### To test real LLM generation (requires internet + HF token)
```bash
export HF_TOKEN="your_huggingface_token_here"
python3 -c "
from rag_pipeline import retrieve, generate_with_llm
docs, metas, dist = retrieve('Why did call volume spike recently?', k=3)
print(generate_with_llm('Why did call volume spike recently?', docs))
"
```

## Upgrading to real embeddings (the natural next step)
Replace the TF-IDF block with a HuggingFace embedding call via
`router.huggingface.co`, or a local `sentence-transformers` model if network
access to the HF Hub is available. The rest of the pipeline (ChromaDB
storage, retrieval, generation) needs no changes — this is exactly why RAG
architectures separate "embedding model" from "vector store" from
"generation model": each piece is swappable independently.

## How this connects to your other projects
- **fleet-qa-copilot**: this project's `generate_with_llm()` reuses the exact
  `router.huggingface.co` calling pattern already validated there.
- **timeseries-forecasting-lab**: the *question* this RAG system answers
  ("why did volume spike?") is the natural companion to that project's
  *forecast* of volume — one predicts the number, the other explains it.
- **banking-intent-classifier**: together, these three projects cover the
  three most common applied-ML task types financial-services/GenAI roles
  ask about: time-series forecasting, tabular classification with
  explainability, and retrieval-augmented generation.

## What this demonstrates (interview framing)
- Understanding of RAG architecture end-to-end, not just "I called an LLM API"
- Honest, specific awareness of the embedding-quality limitation and its fix
- Clean separation of concerns (embedding / storage / retrieval / generation)
  that maps directly onto how production RAG systems are actually built
- Consistency with your existing `fleet-qa-copilot` HuggingFace integration pattern

## Repo convention
Standalone repo: `github.com/fa1829/contact-center-ai` — consistent with
`timeseries-forecasting-lab` and `banking-intent-classifier`.
