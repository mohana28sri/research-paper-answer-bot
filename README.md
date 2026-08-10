# Research Paper Answer Bot 🤖📄

An AI-powered system that allows users to upload research papers (PDFs) and ask questions. 
The application uses Retrieval Augmented Generation (RAG) to retrieve relevant information from documents and generate accurate answers.

## 🚀 Features

- Upload research paper PDFs
- Extract and process document content
- Split documents into meaningful chunks
- Generate embeddings using AI models
- Store vectors using ChromaDB
- Retrieve relevant context using similarity search
- Generate answers using LLM
- FastAPI backend with API testing

## 🏗️ Architecture

User
 ↓
PDF Upload
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embedding Generation
 ↓
ChromaDB Vector Store
 ↓
Retriever
 ↓
LLM
 ↓
Answer Generation


## 🛠️ Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- OpenAI / LLM APIs
- PyPDF
- Pytest

## 📂 Project Structure
