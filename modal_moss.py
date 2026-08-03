from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

from kite_core import (
    TRACK_ROOT,
    atomic_write_json,
    build_submission,
    option_permutations,
    validate_predictions,
)

APP_NAME = "kilometer-audio-moss"
VOLUME_NAME = "perception-test-2026-data"
MODEL_NAME = os.getenv("KITE_MOSS_MODEL", "OpenMOSS-Team/MOSS-Audio-8B-Thinking")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

moss_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "ca-certificates")
    .run_commands(
        "git clone --depth 1 https://github.com/OpenMOSS/MOSS-Audio.git /opt/moss-audio",
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 "
        "-e '/opt/moss-audio[torch-runtime]'",
    )
    .add_local_file("moss_resume.patch", "/tmp/moss_resume.patch", copy=True)
    .run_commands("git -C /opt/moss-audio apply /tmp/moss_resume.patch")
    .pip_install("librosa>=0.11,<1", "peft>=0.17,<1")
    .env(
        {
            "HF_HOME": str(TRACK_ROOT / "models" / "huggingface"),
            "PYTHONPATH": "/opt/moss-audio",
        }
    )
    .add_local_python_source("kite_core")
)


def _video_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    return str(metadata.get("original_video_id") or payload["video_url"].split("v=")[-1])


def _parse_option_timestamp(option: str) -> float | None:
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", option)
    if not match:
        return None
    return int(match.group(1)) * 60.0 + int(match.group(2))


def _nms_times(scores: Any, rows: list[dict[str, Any]], count: int = 5) -> list[float]:
    import numpy as np

    chosen: list[float] = []
    for raw_index in np.argsort(-scores):
        timestamp = float(rows[int(raw_index)]["start"])
        if all(abs(timestamp - prior) >= 45 for prior in chosen):
            chosen.append(timestamp)
        if len(chosen) == count:
            break
    return chosen


def _candidate_times(payload: dict[str, Any], video_id: str) -> list[float]:
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    parsed = [_parse_option_timestamp(option) for option in payload["options"]]
    if sum(timestamp is not None for timestamp in parsed) >= 3:
        return [float(timestamp) for timestamp in parsed if timestamp is not None]

    embeddings = np.load(TRACK_ROOT / "embeddings" / f"{video_id}.clap.npy").astype("float32")
    rows = [
        json.loads(line)
        for line in (TRACK_ROOT / "embeddings" / f"{video_id}.clap.jsonl").read_text().splitlines()
    ]
    query = payload["question"]
    cache = TRACK_ROOT / "models" / "huggingface"
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused", cache_dir=cache)
    model = ClapModel.from_pretrained("laion/clap-htsat-fused", cache_dir=cache).to("cuda").eval()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        feature = model.get_text_features(**inputs)
        feature = torch.nn.functional.normalize(feature, dim=-1)[0].cpu().numpy()
    return _nms_times(embeddings @ feature, rows)


def _extract_clip(video_id: str, center: float, duration: float = 70.0) -> Path:
    audio_path = TRACK_ROOT / "audio" / f"{video_id}.wav"
    clip_dir = TRACK_ROOT / "clips" / "moss" / video_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    start = max(0.0, center - duration / 2)
    clip_path = clip_dir / f"{round(center):06d}.wav"
    if not clip_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(clip_path),
            ],
            check=True,
        )
    return clip_path


@app.function(
    image=moss_image,
    gpu=os.getenv("KITE_MOSS_GPU", "H100"),
    volumes={"/data": volume},
    memory=65536,
    timeout=6 * 60 * 60,
)
def generate_moss_submission(split: str = "validation") -> dict[str, Any]:
    import sys

    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    sys.path.insert(0, "/opt/moss-audio")
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    volume.reload()
    questions_path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    questions = json.loads(questions_path.read_text())
    model_dir = TRACK_ROOT / "models" / "moss" / MODEL_NAME.rsplit("/", 1)[-1]
    if not model_dir.exists() or not any(model_dir.iterdir()):
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_NAME, local_dir=model_dir)
        volume.commit()

    model = MossAudioModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype="auto", device_map="cuda:0"
    )
    adapter_dir = (
        TRACK_ROOT / "models" / "moss" / "kite2-mcq-lora" / "checkpoint-741"
    )
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Frozen MCQ adapter is missing: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    processor = MossAudioProcessor.from_pretrained(
        model_dir, trust_remote_code=True, enable_time_marker=True
    )

    selected: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    permutations = option_permutations()
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(
        '<think>\n</think>\n{"answer_id": ', add_special_tokens=False
    )
    candidate_token_ids: list[int] = []
    for index in range(5):
        token_ids = tokenizer.encode(str(index), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option digit {index} is not a single token: {token_ids}")
        candidate_token_ids.append(token_ids[0])
    for question_id, payload in questions.items():
        video_id = _video_id(payload)
        candidate_times = _candidate_times(payload, video_id)
        canonical_scores = [0.0] * 5
        responses: list[dict[str, Any]] = []
        for candidate_index, center in enumerate(candidate_times):
            permutation = permutations[candidate_index % len(permutations)]
            displayed_options = [payload["options"][index] for index in permutation]
            options_text = "\n".join(
                f"{index}: {option}" for index, option in enumerate(displayed_options)
            )
            clip_path = _extract_clip(video_id, center)
            raw_audio, sample_rate = sf.read(str(clip_path), dtype="float32")
            if sample_rate != processor.config.mel_sr:
                raise ValueError(
                    f"Unexpected clip sample rate {sample_rate}; expected {processor.config.mel_sr}"
                )
            prompt = (
                "Analyze the supplied excerpt from a long walking-tour recording. "
                f"The excerpt is centered at {center:.1f} seconds in the full video. "
                "Use only audible evidence and evaluate every option. "
                "Your final answer must be one option number.\n\n"
                f"Question: {payload['question']}\nOptions:\n{options_text}\n\n"
                "Answer:"
            )
            inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
            inputs = inputs.to(model.device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
            inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
            prefix = torch.tensor(
                [prefix_ids], device=model.device, dtype=inputs["input_ids"].dtype
            )
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(prefix)], dim=1
            )
            inputs["audio_input_mask"] = torch.cat(
                [
                    inputs["audio_input_mask"],
                    torch.zeros_like(prefix, dtype=torch.bool),
                ],
                dim=1,
            )
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1, candidate_token_ids].float()
            displayed_probabilities = torch.softmax(logits, dim=0).cpu().tolist()
            canonical_probabilities = [0.0] * 5
            for displayed_id, probability in enumerate(displayed_probabilities):
                canonical_id = permutation[displayed_id]
                canonical_probabilities[canonical_id] = probability
                canonical_scores[canonical_id] += probability
            responses.append(
                {
                    "center": center,
                    "permutation": permutation,
                    "displayed_probabilities": displayed_probabilities,
                    "canonical_probabilities": canonical_probabilities,
                }
            )
        answer_id = max(range(5), key=lambda index: canonical_scores[index])
        result = {
            "answer_id": answer_id,
            "scores": canonical_scores,
            "clip_count": len(responses),
            "aggregation": "uniform sum of canonical valid-option probabilities",
            "adapter": "kite2-mcq-lora/checkpoint-741",
        }
        selected[question_id] = int(result["answer_id"])
        evidence[question_id] = {"aggregation": result, "clips": responses}

    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid MOSS submission: " + "; ".join(errors))
    output_dir = TRACK_ROOT / "predictions" / split
    atomic_write_json(output_dir / "frozen_mcq_submission.json", submission)
    atomic_write_json(output_dir / "frozen_mcq_evidence.json", evidence)
    result: dict[str, Any] = {
        "split": split,
        "questions": len(submission),
        "submission_path": str(output_dir / "frozen_mcq_submission.json"),
        "adapter": "kite2-mcq-lora/checkpoint-741",
        "kilometer_validation_used_for_selection": False,
    }
    if all("correct_answer" in payload for payload in questions.values()):
        correct = sum(
            submission[question_id]["answer"] == payload["correct_answer"]
            for question_id, payload in questions.items()
        )
        result.update({"correct": correct, "accuracy": correct / len(questions)})
    volume.commit()
    return result


@app.function(
    image=moss_image,
    gpu=os.getenv("KITE_TRAIN_GPU", "H100"),
    volumes={"/data": volume},
    memory=65536,
    timeout=12 * 60 * 60,
)
def train_sound_lora(epochs: float = 1.0) -> dict[str, Any]:
    volume.reload()
    model_dir = TRACK_ROOT / "models" / "moss" / MODEL_NAME.rsplit("/", 1)[-1]
    if not model_dir.exists():
        raise FileNotFoundError("Run MOSS inference once to cache the base checkpoint")
    folds = Path("/data/allowed-training/perception-test/folds")
    train_path = folds / "moss_sound_train.jsonl"
    dev_path = folds / "moss_sound_dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Run modal_train_data.py --action folds first")
    output_dir = TRACK_ROOT / "models" / "moss" / "kite2-sound-lora"
    command = [
        "python",
        "/opt/moss-audio/finetune/finetune.py",
        "--model_dir",
        str(model_dir),
        "--attn_implementation",
        "eager",
        "--data_path",
        str(train_path),
        "--eval_data_path",
        str(dev_path),
        "--output_dir",
        str(output_dir),
        "--use_lora",
        "--lora_rank",
        "32",
        "--lora_alpha",
        "64",
        "--bf16",
        "--max_len",
        "2048",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--num_train_epochs",
        str(epochs),
        "--learning_rate",
        "1e-4",
        "--logging_steps",
        "10",
        "--save_steps",
        "50",
        "--save_total_limit",
        "2",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "100",
        "--report_to",
        "none",
    ]
    subprocess.run(command, check=True, cwd="/opt/moss-audio")
    volume.commit()
    checkpoints = sorted(path.name for path in output_dir.glob("checkpoint-*"))
    return {
        "output_dir": str(output_dir),
        "epochs": epochs,
        "checkpoints": checkpoints,
        "training_source": "official Perception Test train only",
        "kilometer_validation_used": False,
    }


@app.function(
    image=moss_image,
    gpu=os.getenv("KITE_TRAIN_GPU", "H100"),
    volumes={"/data": volume},
    memory=65536,
    timeout=12 * 60 * 60,
)
def train_mcq_lora(epochs: float = 1.0) -> dict[str, Any]:
    """Train only on official train MCQs; challenge validation stays sealed."""
    volume.reload()
    model_dir = TRACK_ROOT / "models" / "moss" / MODEL_NAME.rsplit("/", 1)[-1]
    folds = Path("/data/allowed-training/perception-test/folds")
    train_path = folds / "moss_mcq_train.jsonl"
    dev_path = folds / "moss_mcq_dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Rebuild folds with modal_train_data.py --action folds")
    output_dir = TRACK_ROOT / "models" / "moss" / "kite2-mcq-lora"
    resume_checkpoint = output_dir / "checkpoint-250"
    command = [
        "python",
        "/opt/moss-audio/finetune/finetune.py",
        "--model_dir",
        str(model_dir),
        "--attn_implementation",
        "eager",
        "--data_path",
        str(train_path),
        "--eval_data_path",
        str(dev_path),
        "--output_dir",
        str(output_dir),
        "--use_lora",
        "--lora_rank",
        "32",
        "--lora_alpha",
        "64",
        "--bf16",
        "--max_len",
        "2048",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--num_train_epochs",
        str(epochs),
        "--learning_rate",
        "5e-5",
        "--logging_steps",
        "25",
        "--save_steps",
        "250",
        "--save_total_limit",
        "2",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "250",
        "--report_to",
        "none",
    ]
    if resume_checkpoint.exists():
        command.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    subprocess.run(command, check=True, cwd="/opt/moss-audio")
    volume.commit()
    return {
        "output_dir": str(output_dir),
        "epochs": epochs,
        "checkpoints": sorted(path.name for path in output_dir.glob("checkpoint-*")),
        "training_source": "official Perception Test MCQ train only",
        "kilometer_validation_used": False,
    }


@app.function(
    image=moss_image,
    gpu=os.getenv("KITE_MOSS_GPU", "H100"),
    volumes={"/data": volume},
    memory=65536,
    timeout=12 * 60 * 60,
)
def evaluate_internal_mcq(
    fold: str = "holdout", adapter: str = "checkpoint-200"
) -> dict[str, Any]:
    """Evaluate a frozen candidate without touching KilometerAudio validation."""
    import sys

    import librosa
    import soundfile as sf
    import torch
    from peft import PeftModel

    sys.path.insert(0, "/opt/moss-audio")
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout evaluation is allowed")
    volume.reload()
    folds = Path("/data/allowed-training/perception-test/folds")
    rows = [json.loads(line) for line in (folds / f"mcq_{fold}.jsonl").read_text().splitlines()]
    model_dir = TRACK_ROOT / "models" / "moss" / MODEL_NAME.rsplit("/", 1)[-1]
    model = MossAudioModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype="auto", device_map="cuda:0"
    )
    adapter_dir: Path | None = None
    if adapter != "base":
        if adapter.startswith("mcq-"):
            adapter_dir = (
                TRACK_ROOT
                / "models"
                / "moss"
                / "kite2-mcq-lora"
                / adapter.removeprefix("mcq-")
            )
        else:
            adapter_dir = TRACK_ROOT / "models" / "moss" / "kite2-sound-lora" / adapter
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        model_dir, trust_remote_code=True, enable_time_marker=True
    )
    correct = 0
    predictions: list[dict[str, Any]] = []
    group_totals: dict[str, dict[str, list[int]]] = {"area": {}, "reasoning": {}}
    for row in rows:
        digest = int.from_bytes(
            hashlib.sha256(row["question_id"].encode()).digest()[:8], "big"
        )
        option_count = len(row["options"])
        shift = digest % option_count
        permutation = tuple((index + shift) % option_count for index in range(option_count))
        if (digest // option_count) % 2:
            permutation = tuple(reversed(permutation))
        displayed = [row["options"][index] for index in permutation]
        options_text = "\n".join(f"{index}: {option}" for index, option in enumerate(displayed))
        audio, sample_rate = sf.read(row["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != processor.config.mel_sr:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=processor.config.mel_sr
            )
        prompt = (
            "Use only the supplied audio to answer this multiple-choice question. "
            "Evaluate every option. Your final answer must be one option number.\n\n"
            f"Question: {row['question']}\nOptions:\n{options_text}\n\n"
            "Answer:"
        )
        inputs = processor(text=prompt, audios=[audio], return_tensors="pt").to(model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
        tokenizer = processor.tokenizer
        # Prefill a closed reasoning block and the answer field. Both candidates see
        # exactly the same prefix, so only their valid answer-token logits differ.
        prefix_ids = tokenizer.encode(
            '<think>\n</think>\n{"answer_id": ', add_special_tokens=False
        )
        prefix = torch.tensor([prefix_ids], device=model.device, dtype=inputs["input_ids"].dtype)
        prefix_mask = torch.ones_like(prefix)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], prefix_mask], dim=1
        )
        inputs["audio_input_mask"] = torch.cat(
            [inputs["audio_input_mask"], torch.zeros_like(prefix, dtype=torch.bool)],
            dim=1,
        )
        candidate_token_ids = []
        for index in range(option_count):
            token_ids = tokenizer.encode(str(index), add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Option digit {index} is not a single token: {token_ids}")
            candidate_token_ids.append(token_ids[0])
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, candidate_token_ids].float()
        probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        displayed_id = int(torch.argmax(logits).item())
        canonical_id = permutation[displayed_id]
        is_correct = canonical_id == int(row["answer_id"])
        correct += is_correct
        for group in ("area", "reasoning"):
            label = str(row.get(group) or "unknown")
            counts = group_totals[group].setdefault(label, [0, 0])
            counts[0] += int(is_correct)
            counts[1] += 1
        predictions.append(
            {
                "question_id": row["question_id"],
                "answer_id": canonical_id,
                "correct_answer_id": row["answer_id"],
                "correct": is_correct,
                "displayed_probabilities": probabilities,
                "permutation": permutation,
            }
        )
    metrics = {
        "fold": fold,
        "candidate": adapter,
        "questions": len(rows),
        "accuracy": correct / len(rows),
        "parse_rate": 1.0,
        "scoring": "closed-think constrained valid-option logits v1",
        "groups": {
            group: {
                label: {"correct": counts[0], "total": counts[1], "accuracy": counts[0] / counts[1]}
                for label, counts in labels.items()
            }
            for group, labels in group_totals.items()
        },
        "training_source": "official Perception Test train only",
        "kilometer_validation_used": False,
    }
    output_dir = TRACK_ROOT / "evaluations" / "internal" / fold
    atomic_write_json(output_dir / f"moss_{adapter}_constrained_metrics.json", metrics)
    atomic_write_json(output_dir / f"moss_{adapter}_constrained_predictions.json", predictions)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    action: str = "answer-split",
    split: str = "validation",
    fold: str = "holdout",
    adapter: str = "checkpoint-200",
) -> None:
    if action == "answer-split":
        print(json.dumps(generate_moss_submission.remote(split), indent=2))
    elif action == "train-sound-lora":
        print(json.dumps(train_sound_lora.remote(), indent=2))
    elif action == "train-mcq-lora":
        print(json.dumps(train_mcq_lora.remote(), indent=2))
    elif action == "evaluate-internal-mcq":
        print(json.dumps(evaluate_internal_mcq.remote(fold, adapter), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
