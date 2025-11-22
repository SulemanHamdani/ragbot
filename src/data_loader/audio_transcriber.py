"""Asynchronous audio transcription using OpenAI Whisper."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List, Tuple

from openai import AsyncOpenAI

from src.config import settings


async def transcribe_audio(path: Path, client: AsyncOpenAI | None = None) -> str:
    """Transcribe an audio file to text asynchronously using Whisper."""
    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)
    with path.open("rb") as file_handle:
        response = await active_client.audio.transcriptions.create(
            file=file_handle,
            model=settings.openai.transcription_model,
            response_format="text",
        )
    return response


async def transcribe_audios(paths: Iterable[Path], client: AsyncOpenAI | None = None) -> List[Tuple[Path, str]]:
    """Transcribe multiple audio files asynchronously."""
    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)
    tasks = [transcribe_audio(path, client=active_client) for path in paths]
    transcripts = await asyncio.gather(*tasks)
    return list(zip(paths, transcripts))
