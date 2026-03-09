📄 **Technical Report**: [Advanced Offline RAG — Privacy-First Knowledge Summarizer (PDF)](./Technical_Report.pdf)

---
# Advanced Offline RAG — Privacy-First Knowledge Summarizer

An advanced **Retrieval-Augmented Generation (RAG)** system designed for **offline, privacy-first document intelligence**.  
It integrates local embeddings, FAISS-based retrieval, extractive + abstractive summarization, and document-level knowledge-graph visualization.

---

## Research Summary

This project implements a complete **offline Retrieval-Augmented Generation pipeline**.  
It combines semantic embeddings (Sentence-Transformers), vector search (FAISS), and hybrid summarization (extractive + transformer-based abstractive) to perform context-aware retrieval and reasoning over unstructured text.  

---
## New Features

- **Hybrid Retrieval (Dense + Sparse)**  
  Combines FAISS vector similarity search with BM25 keyword search to improve retrieval accuracy.

- **Duplicate Result Filtering**  
  Automatically removes duplicate chunks when merging dense and sparse retrieval results.

- **Semantic Topic Clustering**  
  Uses **KMeans clustering on document embeddings** to group related research topics automatically.

- **Research Theme Detection**  
  Extracts dominant research themes from clustered document groups using TF-IDF keyword extraction.

- **Research Gap Detection**  
  Identifies underrepresented topic clusters (clusters significantly smaller than others) as potential research gaps.

- **Automated Hypothesis Generation**  
  Generates research ideas by analyzing detected themes, gaps, and cross-topic relationships.

- **Research Analysis Mode (UI)**  
  A dedicated Streamlit mode that allows users to:
  - analyze corpus themes
  - detect research gaps
  - generate research hypotheses interactively.

  ---

## ✨ Key Features

### Modular RAG Core
- Multi-document ingestion (`.pdf` & `.txt`)  
- Sentence-Transformer embeddings (`MiniLM`, `MPNet`)  
- FAISS-based semantic retrieval  
- Extractive + optional abstractive summarization  
- Keyword extraction via TF-IDF  

### Knowledge Graph Visualization
- Builds document-to-document similarity graph  
- Uses cosine similarity between file embeddings  
- Rendered interactively with `networkx` + `matplotlib`

### Persistent Indexing
- Save / load / export / import FAISS indices (`.zip`)  
- Reuse embeddings and metadata across sessions  

---

## Tech Stack

| Component | Library |
|------------|----------|
| Embeddings | Sentence-Transformers |
| Vector Store | FAISS |
| Summarization | Hugging Face Transformers |
| UI | Streamlit |
| Graph Visualization | NetworkX + Matplotlib |
| PDF Parsing | PyPDF |

---
## Interface mode
| Mode                   | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| **Question Answering** | Ask natural questions → retrieves relevant chunks + summarized answer |
| **Summarize All**      | Generates summaries for uploaded documents                            |
| **Explore Index**      | Visualizes document relationships as a knowledge graph                |

---
## Tech detail
| Component              | Implementation Details                                                     |
| ---------------------- | -------------------------------------------------------------------------- |
| **Embedding Model**    | Uses `SentenceTransformer` (`MiniLM` or `MPNet`) → normalized L2 vectors   |
| **Indexing**           | FAISS `IndexFlatIP` for cosine similarity retrieval                        |
| **Chunking Strategy**  | Word-based segmentation (default 400 words, 50 overlap)                    |
| **Summarization**      | Hybrid = extractive sentence scoring + optional abstractive LLM rewrite    |
| **Keyword Extraction** | TF-IDF vectorization with top-k token selection                            |
| **Knowledge Graph**    | Cosine similarity between file centroids → weighted edges → NetworkX graph |
| **Persistence**        | Save / load / export (index + embeddings + metadata) as ZIP                |
| **Interface**          | Streamlit UI integrating retrieval, summarization, and graph exploration   |

---

       

