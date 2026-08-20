from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = "perception-test-2026-data"
TRACK_ROOT = Path("/data/2706-kilometer-audio")
FALLBACK_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}
MODEL_NAME = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
app = modal.App("kilometer-audio-fallback-probe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ca-certificates")
    .run_commands(
        "git clone --depth 1 https://github.com/OpenMOSS/MOSS-Audio.git /opt/moss-audio",
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e '/opt/moss-audio[torch-runtime]'",
    )
    .add_local_file("moss_resume.patch", "/tmp/moss_resume.patch", copy=True)
    .run_commands("git -C /opt/moss-audio apply /tmp/moss_resume.patch")
    .pip_install("librosa>=0.11,<1", "peft>=0.17,<1")
    .env({"HF_HOME": str(TRACK_ROOT / "models" / "huggingface"), "PYTHONPATH": "/opt/moss-audio"})
)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=90 * 60)
def probe_test(adapter: str = "mcq-checkpoint-741", fold: str = "test", limit: int = 40) -> dict[str, Any]:
    import sys

    import torch
    from peft import PeftModel

    sys.path.insert(0, "/opt/moss-audio")
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    volume.reload()
    if fold not in {"test", "dev", "holdout"}:
        raise ValueError("fold must be test, dev, or holdout")
    if fold == "test":
        questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
        evidence = json.loads((TRACK_ROOT / "predictions" / "test" / "kite8_gemini_evidence.json").read_text())
        selected = [qid for qid, row in evidence.items() if row.get("fallback") and row.get("video_id") in FALLBACK_VIDEOS]
    else:
        fold_root = Path("/data/allowed-training/perception-test/folds")
        rows = [json.loads(line) for line in (fold_root / f"mcq_{fold}.jsonl").read_text().splitlines()]
        questions = {row["question_id"]: row for row in rows[:limit]}
        selected = list(questions)
    fallback_ids = selected
    model_dir = TRACK_ROOT / "models" / "moss" / MODEL_NAME.rsplit("/", 1)[-1]
    model = MossAudioModel.from_pretrained(model_dir, trust_remote_code=True, dtype="auto", device_map="cuda:0")
    adapter_dir = TRACK_ROOT / "models" / "moss" / "kite2-mcq-lora" / adapter.removeprefix("mcq-")
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    processor = MossAudioProcessor.from_pretrained(model_dir, trust_remote_code=True, enable_time_marker=True)
    rows = []
    for qid in fallback_ids:
        row = questions[qid]
        options = "\n".join(f"{i}: {option}" for i, option in enumerate(row["options"]))
        prompt = ("Answer this multiple-choice question from the question wording alone; no media is available. "
                  "Evaluate every option and output one option number.\n\n"
                  f"Question: {row['question']}\nOptions:\n{options}\n\nAnswer:")
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
        tokenizer = processor.tokenizer
        prefix_ids = tokenizer.encode('<think>\n</think>\n{"answer_id": ', add_special_tokens=False)
        prefix = torch.tensor([prefix_ids], device=model.device, dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(prefix)], dim=1)
        inputs["audio_input_mask"] = torch.cat([inputs["audio_input_mask"], torch.zeros_like(prefix, dtype=torch.bool)], dim=1)
        token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(len(row["options"]))]
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, token_ids].float()
        probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        prediction = int(torch.argmax(logits).item())
        rows.append({"question_id": qid, "prediction": prediction, "probabilities": probabilities, "fallback_answer": 0})
    correct = None
    if fold != "test":
        correct = sum(row["prediction"] == questions[row["question_id"]]["answer_id"] for row in rows)
    result = {"candidate": "moss_question_only", "adapter": adapter, "fold": fold, "questions": len(rows), "correct": correct, "accuracy": correct / len(rows) if correct is not None and rows else None, "video_labels_used": False, "rows": rows}
    output = TRACK_ROOT / "predictions" / ("test" if fold == "test" else f"authorized-{fold}") / "fallback_probe_question_only.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(adapter: str = "mcq-checkpoint-741", fold: str = "test", limit: int = 40) -> None:
    print(json.dumps(probe_test.remote(adapter, fold, limit), indent=2))
