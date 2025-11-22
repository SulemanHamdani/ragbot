# RAG Chatbot Implementation Plan

## Goals
- Build an asynchronous, end-to-end RAG pipeline that ingests PDFs and audio files, stores embeddings in Qdrant, and answers user questions with GPT 5-mini.
- Provide modular, testable components and runnable scripts for ingestion and querying.

## Architecture Overview
1. **Configuration**
   - Centralize environment configuration (API keys, model names, chunk sizes, Qdrant settings) using `pydantic`-style dataclasses.
2. **Data Ingestion**
   - PDF extraction via `pypdf` (threaded to avoid blocking the event loop).
   - Audio transcription via OpenAI Whisper using `AsyncOpenAI`.
3. **Pre-processing**
   - Normalize and clean text (whitespace trimming, optional lowercasing).
   - Chunk text using a token-aware splitter (`tiktoken`) with overlaps for context preservation.
4. **Embeddings**
   - Generate embeddings asynchronously with OpenAI embeddings API (`text-embedding-3-small` or configurable).
5. **Vector Store**
   - Qdrant client (`:memory:` by default; configurable URL/port).
   - Upsert documents with metadata (source type, file name, chunk index).
6. **Retrieval & QA**
   - Embed queries, retrieve top-k matches from Qdrant, and build a structured prompt including context.
   - Generate answers via GPT 5-mini using `AsyncOpenAI`.
7. **Orchestration Scripts**
   - `scripts/run_ingestion.py`: ingest PDFs and audio directories into Qdrant.
   - `scripts/ask_questions.py`: load Qdrant, ask questions, and log answers.

## Detailed Steps
1. **Config Module** (`src/config.py`)
   - Load `.env` using `python-dotenv`.
   - Define settings: OpenAI API key, model names, Qdrant location, chunking params, default collection name.
2. **Loaders** (`src/data_loader`)
   - `pdf_loader.py`: async wrappers around PDF text extraction with `asyncio.to_thread`.
   - `audio_transcriber.py`: async transcription via Whisper (`client.audio.transcriptions.create`).
3. **Text Processing** (`src/text_processing/chunker.py`)
   - Token-aware splitter with max tokens and overlap.
   - Simple normalization helper for whitespace.
4. **Embeddings** (`src/embeddings/openai_embeddings.py`)
   - Async embedding generation for lists of texts with retry logic (respecting best practices, no try/except around imports).
5. **Vector Store** (`src/vectorstore/qdrant_store.py`)
   - Initialize client and collection.
   - Upsert points with embeddings and metadata.
   - Query for nearest neighbors with filters for source or filename if needed.
6. **RAG Core** (`src/rag/pipeline.py` & `src/rag/chatbot.py`)
   - `Pipeline` to orchestrate ingestion from PDFs/audios, chunking, embedding, and storage.
   - `ChatBot` to retrieve and generate answers with a context-aware prompt.
7. **Scripts**
   - `run_ingestion.py`: CLI arguments for PDF/audio directories and collection name; logs progress.
   - `ask_questions.py`: CLI to ask questions (single or batch), emitting a timestamped log in `logs/`.
8. **Documentation**
   - Update README with setup, usage, and logging instructions.

## Testing Strategy
- Sanity check with `python -m compileall src` for syntax.
- Provide instructions in README for running ingestion and asking sample questions once data is available.

## Future Enhancements
- Add streaming responses, hybrid search (BM25 + vectors), and evaluation harness.
