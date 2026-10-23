from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = "perception-test-2026-data"
TRACK_ROOT = Path("/data/2706-kilometer-audio")
FOLD_ROOT = Path("/data/allowed-training/perception-test/folds")
UNUSABLE_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}
MODEL_NAME = "Qwen/Qwen3.5-27B"
app = modal.App("kilometer-audio-qwen-text-fallback-probe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.9,<2",
        "torch==2.7.1",
        "transformers",
    )
    .env({"HF_HOME": str(TRACK_ROOT / "models" / "huggingface")})
)


def select_constrained_digit(logits: Any, tokenizer: Any, num_options: int) -> tuple[int, list[float]]:
    token_ids = []
    for value in range(num_options):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Digit {value} is not a single token: {encoded}")
        token_ids.append(encoded[0])
    scores = [float(logits[token_id].float().item()) for token_id in token_ids]
    return max(range(num_options), key=scores.__getitem__), scores


def build_prompt(tokenizer: Any, question: str, options: list[str]) -> str:
    option_text = "\n".join(f"{i}: {option}" for i, option in enumerate(options))
    user = (
        "Answer this multiple-choice question from the question wording alone; no video or "
        "audio is available. Evaluate every option carefully, then respond with only the "
        f"option number.\n\nQuestion: {question}\nOptions:\n{option_text}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + '{"answer_id": '


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    return model, tokenizer


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume}, timeout=60 * 60)
def calibrate(fold: str = "holdout", limit: int = 40) -> dict[str, Any]:
    import torch

    volume.reload()
    if fold not in {"dev", "holdout"}:
        raise ValueError("fold must be dev or holdout")
    rows = [json.loads(line) for line in (FOLD_ROOT / f"mcq_{fold}.jsonl").read_text().splitlines()][:limit]
    model, tokenizer = load_model()
    results = []
    correct = 0
    for row in rows:
        prompt = build_prompt(tokenizer, row["question"], row["options"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        prediction, scores = select_constrained_digit(logits, tokenizer, len(row["options"]))
        is_correct = prediction == row["answer_id"]
        correct += is_correct
        results.append({"question_id": row["question_id"], "prediction": prediction,
                         "answer_id": row["answer_id"], "correct": is_correct, "digit_scores": scores})
    result = {"candidate": "qwen3.5-27b_question_only", "model": MODEL_NAME, "fold": fold,
              "questions": len(results), "correct": correct,
              "accuracy": correct / len(results) if results else 0.0,
              "kilometer_validation_used": False, "rows": results}
    output = TRACK_ROOT / "predictions" / f"authorized-{fold}" / "qwen_text_fallback_probe_question_only.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {k: v for k, v in result.items() if k != "rows"}


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume}, timeout=60 * 60)
def test_fallback_cells() -> dict[str, Any]:
    import torch

    volume.reload()
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    evidence = json.loads((TRACK_ROOT / "predictions" / "test" / "kite8_gemini_evidence.json").read_text())
    fallback = json.loads((TRACK_ROOT / "predictions" / "test" / "audio_rag_submission.json").read_text())
    selected = [qid for qid, row in evidence.items()
                if row.get("fallback") and row.get("video_id") in UNUSABLE_VIDEOS]
    model, tokenizer = load_model()
    results = []
    for qid in selected:
        payload = questions[qid]
        prompt = build_prompt(tokenizer, payload["question"], payload["options"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        prediction, scores = select_constrained_digit(logits, tokenizer, len(payload["options"]))
        current = int(fallback[qid]["answer_id"])
        results.append({"question_id": qid, "qwen_prediction": prediction, "digit_scores": scores,
                         "current_fallback_answer": current, "changed": prediction != current})
    result = {"candidate": "qwen3.5-27b_question_only_test_fallback", "model": MODEL_NAME,
              "cells": len(results), "changed": sum(r["changed"] for r in results),
              "kilometer_validation_used": False, "test_answers_used": False, "rows": results}
    output = TRACK_ROOT / "predictions" / "test" / "qwen_text_fallback_probe_cells.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(action: str = "calibrate", fold: str = "holdout", limit: int = 40) -> None:
    if action == "calibrate":
        print(json.dumps(calibrate.remote(fold, limit), indent=2))
    elif action == "test-cells":
        print(json.dumps(test_fallback_cells.remote(), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
