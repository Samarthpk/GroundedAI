# GroundedAI

A local-first multi-agent Retrieval-Augmented Generation (RAG) system designed to answer questions from a private document knowledge base while providing source-backed responses and automated citation validation.

## Overview

GroundedAI combines semantic retrieval, local LLM inference, agent orchestration, and citation validation into a single research workflow.

The system follows this pipeline:

**User Question → Retriever → Analyst → Validator → Final Answer**

If citation validation fails, the workflow automatically routes the response back to the Analyst for one retry before returning a failure response.

## Architecture

### Retriever Agent
- Loads documents from the local knowledge base
- Splits documents into overlapping text chunks
- Generates embeddings using Sentence-Transformers
- Stores embeddings in ChromaDB
- Retrieves the top 5 most semantically relevant chunks for each question

### Analyst Agent
- Uses Llama 3.2 3B through Ollama
- Generates answers strictly from retrieved document context
- Requires source citations using retrieved chunk IDs
- Prevents intentional use of external knowledge through grounding instructions

### Validator Agent
- Extracts citations from generated responses
- Checks citations against retrieved chunk IDs
- Rejects responses containing invalid or missing citations
- Routes failed responses back to the Analyst for regeneration

## RAG Configuration

| Component | Configuration |
|---|---|
| Embedding Model | Sentence-Transformers |
| Vector Database | ChromaDB |
| Retrieval | Semantic similarity |
| Top-K | 5 |
| Selected Chunk Size | 800 characters |
| Chunk Overlap | 150 characters |
| LLM | Llama 3.2 3B |
| LLM Runtime | Ollama |
| Agent Orchestration | LangGraph |
| Web Interface | Flask |

## Retrieval Experiment

Retrieval quality was evaluated using multiple chunk configurations:

| Chunk Size | Overlap |
|---:|---:|
| 300 | 50 |
| 800 | 150 |
| 1500 | 200 |

Testing showed that chunk size materially affected retrieval behavior. Smaller chunks produced highly focused matches but fragmented surrounding context, while larger chunks introduced additional unrelated information.

The 800-character / 150-character overlap configuration was retained as the working configuration to balance semantic relevance and contextual completeness.

## Example

**Question**

> Why should sensitive information use local AI systems?

**Retrieved Source**

`ai_policy.txt-0`

**Answer**

> Sensitive information should use local AI systems because data can remain within organizational infrastructure [ai_policy.txt-0].

The Validator confirms that the cited chunk exists in the retrieved context before the response is accepted.

## Project Structure

```text
GroundedAI/
│
├── agents/
│   └── graph.py
│
├── ingestion/
│   ├── document_loader.py
│   └── vector_store.py
│
├── data/
│   └── sample_documents/
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_chunk_sizes.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

**Python** — application logic and data processing  
**LangGraph** — multi-agent workflow orchestration  
**ChromaDB** — persistent vector storage and semantic retrieval  
**Sentence-Transformers** — local document embeddings  
**Ollama** — local model runtime  
**Llama 3.2 3B** — answer generation  
**Flask** — browser-based user interface

## Installation

Clone the repository:

```bash
git clone https://github.com/Samarthpk/GroundedAI.git
cd GroundedAI
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

## Running the Application

Make sure Ollama is running, then start the Flask application:

```bash
python app.py
```

Open the local Flask URL displayed in the terminal and submit a question through the GroundedAI interface.

## Design Goals

GroundedAI was designed around four principles:

1. **Grounded generation** — answers should come from retrieved documents rather than unsupported model knowledge.
2. **Citation traceability** — generated claims should reference retrieved source chunks.
3. **Local inference** — the architecture supports running inference through Ollama without requiring an external LLM API.
4. **Validation guardrails** — responses failing citation checks are automatically regenerated before being returned.

## Current Scope

The repository demonstrates a working RAG architecture using a small sample knowledge base. It is intended as an engineering prototype rather than a production deployment.

Potential extensions include PDF ingestion, hybrid retrieval, reranking, retrieval evaluation datasets, document upload support, and stronger claim-level citation validation.

## Author

**Samarth Kulkarni**

from pathlib import Path
from typing import Any


def load_text_document(file_path: str) -> dict[str, Any]:
    """
    Load a plain-text document and return its content with basic metadata.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    text = path.read_text(encoding="utf-8")

    return {
        "content": text,
        "metadata": {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lower(),
        },
    }


if __name__ == "__main__":
    document = load_text_document("sample_data/sample_document.txt")

    print("Filename:", document["metadata"]["filename"])
    print("File type:", document["metadata"]["file_type"])
    print("\nContent:\n")
    print(document["content"])
    
