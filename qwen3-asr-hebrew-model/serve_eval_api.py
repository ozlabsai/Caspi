#!/usr/bin/env python3
"""
FastAPI server for ASR evaluation UI.

Provides REST API for:
- Listing datasets and models
- Running evaluations with real-time progress via SSE
- Viewing completed evaluation results
- Managing spelling equivalence rules
- Resumable evaluation runs

Usage:
    uv run uvicorn serve_eval_api:app --host 0.0.0.0 --port 8001 --reload
"""

import asyncio
import csv
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# File paths for persistent storage
DATA_DIR = Path(__file__).parent / "eval_data"
DATA_DIR.mkdir(exist_ok=True)
EQUIVALENCES_FILE = DATA_DIR / "spelling_equivalences.json"
IGNORED_WORDS_FILE = DATA_DIR / "ignored_words.json"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def load_ignored_words() -> set[str]:
    """Load list of words to ignore in WER calculation."""
    if IGNORED_WORDS_FILE.exists():
        with open(IGNORED_WORDS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def save_ignored_words(words: set[str]):
    """Save ignored words list."""
    with open(IGNORED_WORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(words), f, ensure_ascii=False, indent=2)


def load_equivalences() -> dict[str, str]:
    """Load spelling equivalences from file. Maps word -> canonical form."""
    if EQUIVALENCES_FILE.exists():
        with open(EQUIVALENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_equivalences(equivalences: dict[str, str]):
    """Save spelling equivalences to file."""
    with open(EQUIVALENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(equivalences, f, ensure_ascii=False, indent=2)


def load_checkpoint(dataset: str, model: str) -> Optional[dict]:
    """Load checkpoint for a dataset/model combination."""
    checkpoint_file = CHECKPOINTS_DIR / f"{dataset}_{model.replace('/', '_')}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_checkpoint(dataset: str, model: str, data: dict):
    """Save checkpoint for resumable runs."""
    checkpoint_file = CHECKPOINTS_DIR / f"{dataset}_{model.replace('/', '_')}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def clear_checkpoint(dataset: str, model: str):
    """Clear checkpoint after successful completion."""
    checkpoint_file = CHECKPOINTS_DIR / f"{dataset}_{model.replace('/', '_')}.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()

# Available datasets
DATASETS = {
    "eval-d1": {
        "name": "ivrit-ai/eval-d1",
        "split": "test",
        "text_col": "text",
        "description": "ivrit.ai evaluation set D1"
    },
    "eval-whatsapp": {
        "name": "ivrit-ai/eval-whatsapp",
        "split": "test",
        "text_col": "text",
        "description": "ivrit.ai WhatsApp conversations"
    },
    "saspeech": {
        "name": "upai-inc/saspeech",
        "split": "test",
        "text_col": "text",
        "description": "South African Speech corpus"
    },
    "hebrew-speech-kan": {
        "name": "imvladikon/hebrew_speech_kan",
        "split": "validation",
        "text_col": "sentence",
        "description": "Hebrew speech from Kan news"
    },
    "caspi": {
        "name": "OzLabs/caspi",
        "split": "train",
        "text_col": "text",
        "description": "OzLabs Caspi Hebrew evaluation set"
    },
}

# Available models
MODELS = [
    {"id": "OzLabs/Qwen3-ASR-Hebrew-1.7B", "name": "OzLabs Hebrew (Fine-tuned)", "default": True},
    {"id": "Qwen/Qwen3-ASR-1.7B", "name": "Qwen3-ASR Original", "default": False},
]

# Job storage (in-memory)
jobs: dict = {}

app = FastAPI(title="ASR Evaluation API", version="1.0.0")

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class DatasetInfo(BaseModel):
    id: str
    name: str
    split: str
    text_col: str
    description: str


class ModelInfo(BaseModel):
    id: str
    name: str
    default: bool


class EvaluationRequest(BaseModel):
    dataset: str
    model: str
    max_samples: Optional[int] = None
    device: str = "cuda:0"
    resume: bool = False  # Whether to resume from checkpoint


class EvaluationEntry(BaseModel):
    id: int
    reference_text: str
    predicted_text: str
    wer: float
    wil: float
    audio_duration: float
    transcription_time: float


class SpellingEquivalence(BaseModel):
    word1: str
    word2: str


class EquivalenceRule(BaseModel):
    words: List[str]  # All words that are equivalent
    canonical: str  # The canonical form


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, cancelled, error
    dataset: str
    model: str
    total_samples: int
    completed_samples: int
    current_wer: float
    start_time: Optional[str]
    end_time: Optional[str]
    output_file: Optional[str]
    error: Optional[str]


class ResultFile(BaseModel):
    filename: str
    path: str
    size: int
    modified: str
    dataset: Optional[str]
    model: Optional[str]
    wer: Optional[float]
    samples: Optional[int]


# ============== REST Endpoints ==============

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets():
    return [
        DatasetInfo(id=k, name=v["name"], split=v["split"], text_col=v["text_col"], description=v["description"])
        for k, v in DATASETS.items()
    ]


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    return [ModelInfo(**m) for m in MODELS]


@app.get("/api/results", response_model=list[ResultFile])
async def list_results():
    """List all evaluation result CSV files."""
    results_dir = Path(__file__).parent
    result_files = []

    for csv_file in results_dir.glob("*.csv"):
        if csv_file.name.startswith("benchmark") or csv_file.name.startswith("eval"):
            stat = csv_file.stat()

            # Try to extract metadata from CSV
            dataset = None
            model = None
            wer = None
            samples = None

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        samples = len(rows)
                        if 'dataset' in rows[0]:
                            dataset = rows[0]['dataset']
                        if 'model' in rows[0]:
                            model = rows[0]['model']
                        if 'wer' in rows[0]:
                            wers = [float(r['wer']) for r in rows if r.get('wer')]
                            if wers:
                                wer = sum(wers) / len(wers)
            except Exception:
                pass

            result_files.append(ResultFile(
                filename=csv_file.name,
                path=str(csv_file),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                dataset=dataset,
                model=model,
                wer=wer,
                samples=samples,
            ))

    return sorted(result_files, key=lambda x: x.modified, reverse=True)


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    """Get parsed CSV results as JSON."""
    results_dir = Path(__file__).parent
    csv_file = results_dir / filename

    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if not csv_file.suffix == '.csv':
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Convert numeric fields
            for row in rows:
                for key in ['wer', 'wil', 'audio_duration', 'transcription_time']:
                    if key in row and row[key]:
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                for key in ['id', 'substitutions', 'deletions', 'insertions', 'hits']:
                    if key in row and row[key]:
                        try:
                            row[key] = int(row[key])
                        except ValueError:
                            pass

            # Calculate summary stats
            wers = [r['wer'] for r in rows if isinstance(r.get('wer'), (int, float))]
            durations = [r['audio_duration'] for r in rows if isinstance(r.get('audio_duration'), (int, float))]
            times = [r['transcription_time'] for r in rows if isinstance(r.get('transcription_time'), (int, float))]

            summary = {
                "total_samples": len(rows),
                "avg_wer": sum(wers) / len(wers) if wers else None,
                "total_audio_duration": sum(durations) if durations else None,
                "total_transcription_time": sum(times) if times else None,
                "rtf": sum(times) / sum(durations) if durations and times and sum(durations) > 0 else None,
            }

            return {"rows": rows, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Equivalence Endpoints ==============

@app.get("/api/equivalences")
async def get_equivalences():
    """Get all spelling equivalences."""
    equivalences = load_equivalences()
    # Group by canonical form
    groups: dict[str, list[str]] = {}
    for word, canonical in equivalences.items():
        if canonical not in groups:
            groups[canonical] = [canonical]
        if word not in groups[canonical]:
            groups[canonical].append(word)
    return {"rules": [{"canonical": k, "words": v} for k, v in groups.items()]}


@app.post("/api/equivalences")
async def add_equivalence(equiv: SpellingEquivalence):
    """Add a spelling equivalence (word1 and word2 are equivalent)."""
    equivalences = load_equivalences()

    # Find existing canonical form for either word
    canonical1 = equivalences.get(equiv.word1, equiv.word1)
    canonical2 = equivalences.get(equiv.word2, equiv.word2)

    # Use word1 as canonical if both are new, otherwise use existing canonical
    if canonical1 == equiv.word1 and canonical2 == equiv.word2:
        canonical = equiv.word1
    elif canonical1 != equiv.word1:
        canonical = canonical1
    else:
        canonical = canonical2

    # Map both words to the canonical form
    equivalences[equiv.word1] = canonical
    equivalences[equiv.word2] = canonical

    # Update any words that pointed to word2's canonical to use the new canonical
    if canonical2 != canonical:
        for word, can in list(equivalences.items()):
            if can == canonical2:
                equivalences[word] = canonical

    save_equivalences(equivalences)
    return {"status": "ok", "canonical": canonical, "words": [equiv.word1, equiv.word2]}


@app.delete("/api/equivalences/{word}")
async def remove_equivalence(word: str):
    """Remove a word from equivalences."""
    equivalences = load_equivalences()
    if word in equivalences:
        del equivalences[word]
        save_equivalences(equivalences)
    return {"status": "ok"}


@app.get("/api/ignored-words")
async def get_ignored_words():
    """Get list of ignored words."""
    return {"words": list(load_ignored_words())}


@app.post("/api/ignored-words/{word}")
async def add_ignored_word(word: str):
    """Add a word to the ignore list."""
    words = load_ignored_words()
    words.add(word)
    save_ignored_words(words)
    return {"status": "ok", "word": word}


@app.delete("/api/ignored-words/{word}")
async def remove_ignored_word(word: str):
    """Remove a word from the ignore list."""
    words = load_ignored_words()
    words.discard(word)
    save_ignored_words(words)
    return {"status": "ok"}


class RecalculateWerRequest(BaseModel):
    entries: List[dict]
    temp_ignored_words: Optional[List[str]] = None  # Temporary ignored words (not saved)


@app.post("/api/recalculate-wer")
async def recalculate_wer(request: RecalculateWerRequest):
    """Recalculate WER for entries using current equivalences and ignored words.

    temp_ignored_words: optional list of words to ignore temporarily (not saved globally).
    """
    import jiwer
    import whisper.normalizers
    from hebrew import Hebrew

    equivalences = load_equivalences()
    ignored_words = load_ignored_words()
    # Add temporary ignored words (not saved to file)
    temp_ignored = set(request.temp_ignored_words or [])
    all_ignored = ignored_words | temp_ignored

    def clean_unicode(text):
        chars = "\u061C\u200B\u200C\u200D\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069\uFEFF"
        return text.translate({ord(c): None for c in chars})

    def apply_equivalences(text: str) -> str:
        if not equivalences:
            return text
        words = text.split()
        normalized_words = [equivalences.get(w, w) for w in words]
        return ' '.join(normalized_words)

    def remove_ignored(text: str) -> str:
        if not all_ignored:
            return text
        words = text.split()
        filtered = [w for w in words if w not in all_ignored]
        return ' '.join(filtered)

    def normalize(text):
        text = clean_unicode(text)
        text = Hebrew(text).no_niqqud().string
        text = text.replace('"', "").replace("'", "")
        text = whisper.normalizers.BasicTextNormalizer()(text)
        text = apply_equivalences(text)
        text = remove_ignored(text)
        return text

    updated_entries = []
    for entry in request.entries:
        norm_ref = normalize(entry.get("reference_text", ""))
        norm_pred = normalize(entry.get("predicted_text", ""))
        metrics = jiwer.process_words([norm_ref], [norm_pred])

        updated_entry = {
            **entry,
            "norm_reference_text": norm_ref,
            "norm_predicted_text": norm_pred,
            "wer": metrics.wer,
            "wil": metrics.wil,
        }
        updated_entries.append(updated_entry)

    # Calculate new average WER
    avg_wer = sum(e["wer"] for e in updated_entries) / len(updated_entries) if updated_entries else 0

    return {"entries": updated_entries, "avg_wer": avg_wer}


# ============== Audio Endpoint ==============

@app.get("/api/audio/{dataset}/{sample_id}")
async def get_audio(dataset: str, sample_id: int):
    """Get audio data for a specific sample."""
    import datasets as hf_datasets
    import io
    import soundfile as sf
    import librosa

    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")

    ds_config = DATASETS[dataset]

    try:
        # Load with decode=False to get raw audio bytes (avoids torchcodec dependency)
        ds = hf_datasets.load_dataset(ds_config["name"])[ds_config["split"]]
        ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))

        if sample_id >= len(ds):
            raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

        entry = ds[sample_id]
        audio_data = entry["audio"]

        # Audio is a dict with 'path' and 'bytes'
        audio_bytes = audio_data.get("bytes")
        if audio_bytes:
            # Decode audio bytes using librosa
            audio_buffer = io.BytesIO(audio_bytes)
            audio_array, sr = librosa.load(audio_buffer, sr=16000)
        else:
            # Fallback: load from path
            audio_path = audio_data.get("path")
            if not audio_path:
                raise HTTPException(status_code=500, detail="No audio data available")
            audio_array, sr = librosa.load(audio_path, sr=16000)

        # Write to BytesIO as WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sr, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"inline; filename=sample_{sample_id}.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Checkpoint Endpoints ==============

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints for resuming."""
    checkpoints = []
    for f in CHECKPOINTS_DIR.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                checkpoints.append({
                    "dataset": data.get("dataset"),
                    "model": data.get("model"),
                    "completed": data.get("completed_samples", 0),
                    "total": data.get("total_samples", 0),
                    "timestamp": data.get("timestamp"),
                })
        except Exception:
            pass
    return {"checkpoints": checkpoints}


@app.delete("/api/checkpoints/{dataset}/{model:path}")
async def delete_checkpoint(dataset: str, model: str):
    """Delete a checkpoint."""
    clear_checkpoint(dataset, model)
    return {"status": "ok"}


@app.post("/api/evaluate", response_model=JobStatus)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a new evaluation job."""
    # Handle "all" datasets
    if request.dataset == "all":
        datasets_to_run = list(DATASETS.keys())
    elif request.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}")
    else:
        datasets_to_run = [request.dataset]

    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "dataset": request.dataset,
        "datasets_to_run": datasets_to_run,
        "model": request.model,
        "device": request.device,
        "max_samples": request.max_samples,
        "resume": request.resume,
        "total_samples": 0,
        "completed_samples": 0,
        "current_wer": 0.0,
        "start_time": None,
        "end_time": None,
        "output_file": None,
        "error": None,
        "queue": asyncio.Queue(),
        "entries": [],
    }

    background_tasks.add_task(run_evaluation, job_id)

    return JobStatus(**{k: v for k, v in jobs[job_id].items() if k not in ['queue', 'entries', 'datasets_to_run']})


@app.get("/api/evaluate/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get current job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    return JobStatus(**{k: v for k, v in job.items() if k not in ['queue', 'entries']})


@app.post("/api/evaluate/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    if job["status"] == "running":
        job["status"] = "cancelled"
        job["end_time"] = datetime.now().isoformat()
        await job["queue"].put({"event": "cancelled"})

    return {"status": "cancelled", "job_id": job_id}


@app.get("/api/evaluate/stream/{job_id}")
async def stream_progress(job_id: str):
    """SSE endpoint for real-time progress updates."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        job = jobs[job_id]
        queue = job["queue"]

        # Send initial status
        yield f"data: {json.dumps({'event': 'connected', 'job_id': job_id})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"

                if event.get("event") in ["complete", "error", "cancelled"]:
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'event': 'keepalive'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============== Background Evaluation Task ==============

async def run_evaluation(job_id: str):
    """Run evaluation in background and emit progress events."""
    job = jobs[job_id]
    queue = job["queue"]

    try:
        job["status"] = "running"
        job["start_time"] = datetime.now().isoformat()

        await queue.put({"event": "started", "job_id": job_id})

        # Import evaluation dependencies
        import importlib.util
        import datasets as hf_datasets
        import jiwer
        import whisper.normalizers
        from hebrew import Hebrew

        # Load engine once
        engine_path = Path(__file__).parent / "scripts" / "engines" / "qwen_asr_engine.py"
        spec = importlib.util.spec_from_file_location("engine", engine_path)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        transcribe_fn = engine.create_app(model_path=job["model"], device=job["device"])

        # Load spelling equivalences and ignored words
        equivalences = load_equivalences()
        ignored_words = load_ignored_words()

        def clean_unicode(text):
            chars = "\u061C\u200B\u200C\u200D\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069\uFEFF"
            return text.translate({ord(c): None for c in chars})

        def apply_equivalences(text: str) -> str:
            """Replace words with their canonical forms."""
            if not equivalences:
                return text
            words = text.split()
            normalized_words = [equivalences.get(w, w) for w in words]
            return ' '.join(normalized_words)

        def remove_ignored(text: str) -> str:
            """Remove ignored words from text."""
            if not ignored_words:
                return text
            words = text.split()
            filtered = [w for w in words if w not in ignored_words]
            return ' '.join(filtered)

        def normalize(text):
            text = clean_unicode(text)
            text = Hebrew(text).no_niqqud().string
            text = text.replace('"', "").replace("'", "")
            text = whisper.normalizers.BasicTextNormalizer()(text)
            text = apply_equivalences(text)  # Apply spelling equivalences
            text = remove_ignored(text)  # Remove ignored words
            return text

        # Check for checkpoint if resuming
        checkpoint = None
        if job.get("resume"):
            checkpoint = load_checkpoint(job["dataset"], job["model"])

        # Calculate total samples across all datasets
        datasets_to_run = job.get("datasets_to_run", [job["dataset"]])
        total_samples = 0
        dataset_infos = []

        for ds_key in datasets_to_run:
            ds_config = DATASETS[ds_key]
            ds = hf_datasets.load_dataset(ds_config["name"])[ds_config["split"]]
            ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
            ds_total = len(ds) if job["max_samples"] is None else min(job["max_samples"], len(ds))
            dataset_infos.append((ds_key, ds_config, ds, ds_total))
            total_samples += ds_total

        job["total_samples"] = total_samples

        await queue.put({
            "event": "progress",
            "job_id": job_id,
            "completed": 0,
            "total": total_samples,
            "current_wer": 0.0,
        })

        # Run evaluation across all datasets
        # Resume from checkpoint if available
        if checkpoint:
            entries = checkpoint.get("entries", [])
            wer_sum = sum(e["wer"] for e in entries)
            global_idx = len(entries)
            start_dataset_idx = checkpoint.get("current_dataset_idx", 0)
            start_sample_idx = checkpoint.get("current_sample_idx", 0)
            job["completed_samples"] = global_idx
            job["current_wer"] = wer_sum / global_idx if global_idx > 0 else 0.0
            # Send resumed entries to frontend
            for entry_data in entries:
                await queue.put({
                    "event": "progress",
                    "job_id": job_id,
                    "completed": entry_data["id"] + 1,
                    "total": total_samples,
                    "current_wer": job["current_wer"],
                    "current_dataset": entry_data.get("dataset"),
                    "entry": entry_data,
                })
        else:
            wer_sum = 0.0
            entries = []
            global_idx = 0
            start_dataset_idx = 0
            start_sample_idx = 0

        for ds_idx, (ds_key, ds_config, ds, ds_total) in enumerate(dataset_infos):
            # Skip datasets before checkpoint
            if ds_idx < start_dataset_idx:
                continue

            sample_start = start_sample_idx if ds_idx == start_dataset_idx else 0

            for i in range(sample_start, ds_total):
                if job["status"] == "cancelled":
                    break

                entry = ds[i]
                ref_text = entry[ds_config["text_col"]]

                # Transcribe (run in thread pool to not block event loop)
                pred_text, transcription_time, audio_duration = await asyncio.to_thread(
                    transcribe_fn, entry
                )

                # Calculate WER
                norm_ref = normalize(ref_text)
                norm_pred = normalize(pred_text)
                metrics = jiwer.process_words([norm_ref], [norm_pred])

                entry_data = {
                    "id": global_idx,
                    "dataset": ds_key,
                    "dataset_sample_idx": i,  # Index within dataset for audio playback
                    "reference_text": ref_text,
                    "predicted_text": pred_text,
                    "norm_reference_text": norm_ref,
                    "norm_predicted_text": norm_pred,
                    "wer": metrics.wer,
                    "wil": metrics.wil,
                    "audio_duration": audio_duration,
                    "transcription_time": transcription_time,
                }

                entries.append(entry_data)
                wer_sum += metrics.wer
                global_idx += 1
                job["completed_samples"] = global_idx
                job["current_wer"] = wer_sum / global_idx
                job["entries"] = entries

                # Save checkpoint after each sample
                save_checkpoint(job["dataset"], job["model"], {
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "entries": entries,
                    "completed_samples": global_idx,
                    "total_samples": total_samples,
                    "current_dataset_idx": ds_idx,
                    "current_sample_idx": i + 1,
                    "timestamp": datetime.now().isoformat(),
                })

                await queue.put({
                    "event": "progress",
                    "job_id": job_id,
                    "completed": global_idx,
                    "total": total_samples,
                    "current_wer": job["current_wer"],
                    "current_dataset": ds_key,
                    "entry": entry_data,
                })

            if job["status"] == "cancelled":
                break

        if job["status"] != "cancelled":
            # Save results
            output_file = Path(__file__).parent / f"eval_{job['dataset']}_{job_id}.csv"

            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if entries:
                    writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                    writer.writeheader()
                    writer.writerows(entries)

            job["status"] = "completed"
            job["end_time"] = datetime.now().isoformat()
            job["output_file"] = str(output_file)

            # Clear checkpoint on successful completion
            clear_checkpoint(job["dataset"], job["model"])

            await queue.put({
                "event": "complete",
                "job_id": job_id,
                "final_wer": job["current_wer"],
                "output_file": str(output_file),
            })

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["end_time"] = datetime.now().isoformat()

        await queue.put({
            "event": "error",
            "job_id": job_id,
            "error": str(e),
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
