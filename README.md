# Advanced Offline RAG — Privacy-first (LLM Summarizer Included)

## Overview
This project is an advanced privacy-first Retrieval-Augmented system with an optional local LLM summarizer for higher-quality abstractive summaries. It is designed to run on Hugging Face Spaces (Streamlit) or locally. The LLM summarizer is optional and disabled by default — enable it only if your environment has sufficient RAM.

## Features included
- PDF and TXT ingestion with chunking and overlap
- Sentence-transformer embeddings (MiniLM / MPNet)
- FAISS vector index and retrieval
- Extractive summarization (select sentences by semantic centrality)
- Keyword extraction (TF-IDF)
- Evidence visualization (retrieval strength bar chart)
- Knowledge-graph (file-level similarity) view
- Index persistence and export/import (.zip)
- Optional **local LLM summarizer** (default model: `sshleifer/distilbart-cnn-12-6`) for abstractive rewriting of extractive summaries

## How to run locally
1. Create virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Run streamlit:
```bash
streamlit run streamlit_app.py
```
3. Upload PDFs or TXT files and click **Build Index**.
4. Use **Summarize All** or **Question Answering**. You can enable the local summarizer from the sidebar (will download model).

## Deploy to Hugging Face Spaces (Streamlit)
- Create a new Space (Streamlit).
- Add these files to the repo and push.
- The Space will install requirements and models download on first use.
- Do NOT initialize the local summarizer in Spaces unless you know the Space has enough memory; it may fail for large models.

## Notes on the LLM summarizer
- Default model `sshleifer/distilbart-cnn-12-6` is a distilled BART fine-tuned for summarization (smaller and faster than full BART).
- For instruction-style rewriting you may choose other seq2seq models, but they might be larger.
- The summarizer is used only for rewriting extractive summaries into cleaner abstractive summaries. Core retrieval and evidence remain extractive and transparent.

## Files
- `core.py` — core logic with summarizer interface
- `streamlit_app.py` — Streamlit UI
- `requirements.txt` — dependencies
- `README.md` — this file

