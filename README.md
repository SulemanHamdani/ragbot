# RAGBot

Asynchronous Retrieval-Augmented Generation pipeline for ingesting PDFs and audio, storing embeddings in Qdrant, and answering questions with GPT 5-mini + Whisper.

## Setup
1. Create a `.env` file with your OpenAI credentials:
   ```bash
   echo "OPENAI_API_KEY=<your-key>" > .env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
- `docs/PLAN.md`: Detailed implementation plan.
- `src/`: Library code for ingestion, chunking, embeddings, vector store, and chatbot.
- `scripts/`: Command-line tools for ingestion and querying.
- `logs/`: Output folder for generated answer logs.

## Usage
### Ingest PDFs and Audio
```bash
python scripts/run_ingestion.py --pdf-dir /path/to/pdfs --audio-dir /path/to/audio --collection ragbot-collection
```

### Ask Questions
```bash
python scripts/ask_questions.py "What are the production Do's for RAG?" "Why is hybrid search better than vector-only search?"
```
Or provide a file with one question per line:
```bash
python scripts/ask_questions.py --file questions.txt --collection ragbot-collection
```
This writes a timestamped log to `logs/`.

## Notes
- Qdrant defaults to in-memory (`:memory:`). Provide `QDRANT_URL` and `QDRANT_API_KEY` in `src/config.py` if pointing to a remote instance.
- Models: `gpt-5.1-mini` for chat, `text-embedding-3-small` for embeddings, and `whisper-1` for transcription. Adjust in `src/config.py` as needed.

## Testing
Basic syntax check:
```bash
python -m compileall src
```
