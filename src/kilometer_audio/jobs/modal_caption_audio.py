from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-caption-audio-index")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("faster-whisper")


@app.function(image=image, volumes={"/data": volume}, timeout=6 * 60 * 60)
def transcribe(fold: str = "holdout", timeline_index: int = 0, model_name: str = "small") -> dict[str, Any]:
    from faster_whisper import WhisperModel

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    virtual = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())[timeline_index]
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    audio = output / f"{timeline}-audio.wav"
    if not audio.exists():
        inputs = [item for segment in virtual["segments"] for item in ("-i", segment["audio_path"])]
        filters = ";".join(f"[{i}:a]aresample=16000,asetpts=PTS-STARTPTS[a{i}]" for i in range(len(virtual["segments"])))
        stream = "".join(f"[a{i}]" for i in range(len(virtual["segments"])))
        filters += f";{stream}concat=n={len(virtual['segments'])}:v=0:a=1[out]"
        subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", filters, "-map", "[out]", "-ac", "1", "-ar", "16000", str(audio)], check=True, capture_output=True, timeout=900)
    whisper = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(ROOT / "models" / "whisper"))
    segments, info = whisper.transcribe(str(audio), beam_size=3, vad_filter=True)
    rows = [{"start_sec": round(segment.start, 3), "end_sec": round(segment.end, 3), "text": segment.text.strip()} for segment in segments]
    path = output / f"{timeline}-transcript.json"
    path.write_text(json.dumps({"timeline": timeline, "model": model_name, "duration": info.duration, "segments": rows, "kilometer_validation_used": False}, indent=2) + "\n")
    volume.commit()
    return {"timeline": timeline, "model": model_name, "segments": len(rows), "duration": info.duration, "path": str(path), "kilometer_validation_used": False}


@app.function(image=image, volumes={"/data": volume}, timeout=6 * 60 * 60)
def transcribe_materialized(fold: str = "holdout", timeline_index: int = 0, model_name: str = "small") -> dict[str, Any]:
    """Transcribe a materialized timeline when source segment manifests are unavailable."""
    from faster_whisper import WhisperModel

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    source = ROOT / "materialized" / fold / f"{timeline}.mp4"
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    audio = output / f"{timeline}-materialized-audio.wav"
    if not audio.exists():
        extracted = subprocess.run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(audio)], capture_output=True, text=True, timeout=1800)
        if extracted.returncode != 0:
            return {"timeline": timeline, "model": model_name, "segments": 0, "duration": 0.0, "audio_available": False, "error": extracted.stderr[-1000:], "kilometer_validation_used": False}
    whisper = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(ROOT / "models" / "whisper"))
    segments, info = whisper.transcribe(str(audio), beam_size=3, vad_filter=True)
    rows = [{"start_sec": round(segment.start, 3), "end_sec": round(segment.end, 3), "text": segment.text.strip()} for segment in segments]
    path = output / f"{timeline}-materialized-transcript.json"
    path.write_text(json.dumps({"timeline": timeline, "model": model_name, "duration": info.duration, "segments": rows, "source": str(source), "kilometer_validation_used": False}, indent=2) + "\n")
    volume.commit()
    return {"timeline": timeline, "model": model_name, "segments": len(rows), "duration": info.duration, "audio_available": True, "path": str(path), "kilometer_validation_used": False}


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def audit_materialized_audio(fold: str = "holdout") -> dict[str, Any]:
    """Record audio-stream availability before any ASR or multimodal scoring."""
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for timeline_index in range(5):
        timeline = f"{fold}-{timeline_index:03d}"
        source = ROOT / "materialized" / fold / f"{timeline}.mp4"
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)], capture_output=True, text=True, check=True, timeout=120)
        streams = json.loads(probe.stdout).get("streams", [])
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        rows.append({"timeline": timeline, "source": str(source), "audio_streams": len(audio), "audio_available": bool(audio), "video_streams": sum(stream.get("codec_type") == "video" for stream in streams)})
    path = output / "materialized-audio-manifest.json"
    path.write_text(json.dumps({"fold": fold, "timelines": rows, "kilometer_validation_used": False}, indent=2) + "\n")
    volume.commit()
    return {"fold": fold, "timelines": len(rows), "audio_available": sum(row["audio_available"] for row in rows), "path": str(path), "kilometer_validation_used": False}


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, model: str = "small", materialized: bool = False, audit: bool = False) -> None:
    result = audit_materialized_audio.remote(fold) if audit else transcribe_materialized.remote(fold, timeline, model) if materialized else transcribe.remote(fold, timeline, model)
    print(json.dumps(result, indent=2))
