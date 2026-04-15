# Paper 1: How Chunking, Embedding, and Reranking Interact in Long-Document RAG

**Status:** Experiments in progress  
**Target:** EMNLP 2026 / ACL ARR

## Research Question
Do chunking strategy, embedding model, and reranking interact in predictable ways across document lengths — and which factor dominates for long documents?

## Experimental Design
- **3 chunking strategies:** fixed-256, fixed-512, sentence-boundary
- **3 retrieval configs:** BM25, BGE-small dense, hybrid BM25+BGE (RRF)
- **2 reranking modes:** with/without cross-encoder (ms-marco-MiniLM-L-6-v2)
- **2 datasets:** QuALITY (avg 5K tokens), NarrativeQA (avg 64K tokens)
- **Total:** 18 configurations × 2 datasets

## Files
- `experiment.ipynb` — Full Colab experiment (run on free T4 GPU, ~4 hrs)
- `chunking.py` — Chunking strategy implementations
- `retrieval.py` — BM25, dense, hybrid retrieval
- `evaluation.py` — Recall@k, MRR metrics
- `paper_draft.docx` — Current paper draft

## How to Run
1. Open `experiment.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom (~4 hrs)
4. Download `rag_experiment_results.csv` at end
