#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn qdrant-client langchain langchain-community langchain-qdrant langchain-huggingface pypdf python-docx python-multipart