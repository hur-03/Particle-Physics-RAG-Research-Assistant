# Particle Physics Research Assistant

A Retrieval-Augmented Generation (RAG) application that combines particle physics literature, vector search, live web search, and Large Language Models to answer physics questions using both local documents and recent online information.

## Features

- PDF document ingestion
- Semantic search using FAISS
- OpenAI embeddings
- Groq-hosted LLM responses
- Tavily web search integration
- Multi-agent workflow:
  - Researcher
  - Writer
  - Reviewer
- Streamlit web interface
- Permanent PDF uploads
- Docker support

## Project Architecture

```text
Particle Physics PDFs
        ↓
Document Loading
        ↓
Text Chunking
        ↓
OpenAI Embeddings
        ↓
FAISS Vector Database
        ↓
Retriever
        ↓
Groq LLM

Researcher Agent
        ↓
Writer Agent
        ↓
Reviewer Agent
        ↓
Final Answer
```

## Technologies Used

- Python
- LangChain
- FAISS
- OpenAI Embeddings
- Groq
- Tavily
- Streamlit
- Docker
- Scikit-Learn
- Pandas

## Example Questions

- What is unfolding in particle physics?
- How does ATLAS reconstruct particles?
- What is flavour tagging?
- What are parton distribution functions?
- How was the Higgs boson discovered?

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/particle-physics-rag-assistant.git
cd particle-physics-rag-assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```
### Add Research Files

Create a `crew_data` file to store research papers and other data

```crew_data
Upload files here
```

### Run Application

```bash
streamlit run app.py
```

## Docker

Build image:

```bash
docker build -t particle-physics-rag-app .
```

Run container:

```bash
docker run -p 8501:8501 --env-file .env particle-physics-rag-app
```

Open:

```text
http://localhost:8501
```

## Future Improvements

- PDF report export
- Citation generation
- Cloud deployment
- Additional specialised physics agents


## Visualisation of Embedding using PCA
<img width="987" height="690" alt="image" src="https://github.com/user-attachments/assets/3bb524f2-f0d9-4897-811d-0cacd272ee7a" />

## Screenshot of Application

<img width="1265" height="704" alt="image" src="https://github.com/user-attachments/assets/e5fe9a71-38fb-4ecc-87e8-2c6bc058a4d3" />

