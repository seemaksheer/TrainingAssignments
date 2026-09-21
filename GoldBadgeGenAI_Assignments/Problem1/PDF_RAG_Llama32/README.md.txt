# RAG_WITH_GPT4All_Desktop

## Retrieval-Augmented Generation (RAG) using Gemini Embeddings, ChromaDB and GPT4All Desktop

This project demonstrates a complete local Retrieval-Augmented Generation (RAG) application using:

- Google Gemini Embeddings
- ChromaDB Vector Database
- GPT4All Desktop
- Local Phi-3 Mini Instruct LLM
- Python
- Jupyter Notebook
- PDF documents as the knowledge base

The application retrieves relevant information from PDF documents and provides grounded answers using a locally running Large Language Model through the GPT4All Desktop API.

---

## 1. Project Objective

The objective of this project is to build a RAG-based question-answering system that can:

1. Load information from PDF documents.
2. Extract and split the document content into smaller chunks.
3. Generate vector embeddings using Google Gemini.
4. Store the embeddings in ChromaDB.
5. Perform semantic similarity search for a user question.
6. Retrieve the most relevant document chunks.
7. Send the retrieved context to a local LLM running through GPT4All Desktop.
8. Generate an answer based on the retrieved information.
9. Display the source document and page information used for the answer.

---

## 2. Architecture

The application follows this RAG pipeline:

```text
                 PDF Documents
                       |
                       v
              PDF Text Extraction
                       |
                       v
                 Text Chunking
                       |
                       v
             Gemini Embeddings
                       |
                       v
                  ChromaDB
              Vector Database
                       |
                       |
User Question --------+
                       |
                       v
              Gemini Embedding
                       |
                       v
             Semantic Retrieval
                       |
                       v
             Relevant PDF Chunks
                       |
                       v
             Context Construction
                       |
                       v
             GPT4All Desktop API
                       |
                       v
          Phi-3 Mini Instruct Model
                       |
                       v
                Final Answer
                       |
                       v
             Verified Source Info