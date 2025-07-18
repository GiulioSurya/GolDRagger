# GolDRagger

## Project Overview

I'm a lazy person who wants to create an AI agent to help me with specific tasks during work. Why do everything manually when you can automate it?

The first step was creating a RAG (Retrieval-Augmented Generation) system to allow the AI to access my personal information and documents. But this is just the beginning - many other features will be added to make my work life simpler.

## Current Features

- **RAG System** with ChromaDB for document embedding and retrieval
- **Chatbot** interface for interacting with indexed documents

## Usage

### 1. Create embeddings from terminal

To index your documents and create embeddings:

```bash
python chroma_embender.py your_document.pdf
```

Available options:
```bash
python chroma_embender.py document.pdf --chunk-size 600 --collection "my_docs"
```

This script:
- Reads PDF documents
- Creates embeddings using ChromaDB with Ollama
- Saves everything in the vector database for retrieval

### 2. Use the chatbot

To start the chatbot and ask questions about your documents:

```bash
python Gol_D_Ragger.py
```

Once started, you can:
- Ask questions about the documents you've indexed
- Get contextualized responses based on your data
- Exit by typing `quit` or `exit`
- Use commands like `/help` and `/clear`

## Next Steps

This is just the beginning. Future features will include:
- Automation of repetitive tasks
- Integration with work tools
- Automatic scheduling
- Other useful things for lazy people like me

## Installation

```bash
pip install -r requirements.txt
```

Make sure you have Ollama running locally with the required models:
- `nomic-embed-text` for embeddings
- `llama3.1:8b` for the language model

## Requirements

- Python 3.8+
- Ollama running on localhost:11434
- Required models downloaded in Ollama