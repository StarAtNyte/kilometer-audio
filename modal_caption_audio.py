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


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, model: str = "small") -> None:
    print(json.dumps(transcribe.remote(fold, timeline, model), indent=2))
