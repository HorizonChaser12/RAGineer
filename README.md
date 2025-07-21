# Enhanced Azure RAG System

**Advanced Retrieval-Augmented Generation (RAG) System with Hybrid Retrieval, AI Re-Ranking, and Analytical Insights**

## Overview

The Enhanced Azure RAG System is a powerful, production-ready AI platform for enterprise knowledge retrieval, document search, and analytics. It combines state-of-the-art OpenAI models (via Azure), Sentence Transformers, FAISS vector search, Cross-Encoder re-ranking, and direct data analytics to deliver accurate, explainable answers for business users and technical support scenarios.

**Key Features:**
- Hybrid retrieval using Azure OpenAI embeddings and Sentence Transformers.
- Cross-Encoder-based re-ranking for highly accurate semantic matches.
- Analytical thinking mode for direct Pandas-based data analysis.
- Fast and extensible FastAPI backend with CORS support.
- Modern web interface (TailwindCSS) for seamless user experience and detailed insight visualization.
- Robust system logging, error handling, and status monitoring.

## Demo

 <!-- Add an actual screenshot for best results -->

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

## Features

- **Multi-Modal Retrieval:** Combine Azure OpenAI and Sentence Transformers for hybrid semantic query matching.
- **AI Re-Ranking:** Cross-Encoder refines relevant results for precise context retrieval.
- **Pattern Analysis:** Automatic detection of trends, ranges, categories, and metrics in retrieved documents.
- **Direct Analytics:** Bypass Q&A to perform Pandas-based statistical analysis with one click.
- **Interactive Frontend:** Responsive UI with TailwindCSS, analytics dashboards, and real-time system status.
- **Cloud-Ready:** Compatible with Azure, Docker, and scalable for enterprise deployments.

## Architecture

text
                 User (Web/App)
                      │
           ┌──────────▼──────────┐
           │      FastAPI        │
           ├─────────┬───────────┤
      Hybrid Retrieval   Analytics Mode
  (OpenAI, FAISS, ST)        (Pandas)
           │                  │
Retriever → Re-Ranker Data Analyzer
│
│ Pattern De
ection LLM/Structu
text

## Setup & Installation

### 1. Clone the Repository

git clone https://github.com/yourusername/azure-rag-enhanced.git
cd azure-rag-enhance

text

### 2. Install Dependencies

Ensure Python 3.8+ is available.

pip install -r requirements.txt

text

### 3. Configure Environment

- Copy `.env.example` to `.env` and fill in your Azure OpenAI keys and endpoints.
- Place any source Excel files needed for retrieval queries in a valid path.

### 4. Start the Backend

uvicorn main:app --reload

text

### 5. Launch the Frontend

Open `index.html` (in your web browser).

## Usage

1. **Initialize the System:** Point to your data file via `/initialize`.
2. **Query:** Use the web form to ask questions or use `/query` API.
3. **Switch Modes:** Toggle between retrieval and analytical thinking as needed.
4. **Monitor Status:** Check `/status` for system health and readiness.

## API Endpoints

| Endpoint              | Method | Description                                  |
|-----------------------|--------|----------------------------------------------|
| `/initialize`         | POST   | Load data, set configs, and start system     |
| `/query`              | POST   | Ask a question and get a detailed answer     |
| `/retrieve`           | POST   | Retrieve relevant documents only             |
| `/rebuild-index`      | POST   | Rebuild search indices from current data     |
| `/status`             | GET    | Get system's current status and availability |
| `/health`             | GET    | Simple health check                          |

**See `main.py` for payload examples.**

## Configuration

All sensitive keys and connection details are managed by environment variables:

AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_VERSION=2023-03-15-preview
AZURE_OPENAI_DEPLOYMENT=your-deployment
text

## Tech Stack

- **Python**, **FastAPI**, **Uvicorn**
- **Pandas**, **FAISS**, **Sentence Transformers**, **Cross-Encoder**
- **Azure OpenAI API**
- **TailwindCSS** (Frontend)

## Contributing

Contributions, issues, and feature requests are welcome!  
Please open a pull request or file an issue for discussion.

## Acknowledgments

- TCS Aviva for giving me opportunity to work on this project.
- OpenAI, Huggingface, and the open-source community for models, APIs, and libraries.