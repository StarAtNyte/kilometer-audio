"""Build an option-conditioned, beam-2 multimodal test evidence artifact."""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
ROOT = Path("/data/2706-kilometer-audio")
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3", "pillow>=11,<13", "torch==2.7.1", "transformers>=4.57,<5")
)
app = modal.App("kilometer-audio-option-only-test")


def option_queries(question: str, options: list[str]) -> list[str]:
    words = [re.findall(r"[a-z0-9]+", option.lower()) for option in options]
    frequency: dict[str, int] = {}
    for row in words:
        for word in set(row):
            frequency[word] = frequency.get(word, 0) + 1
    return [
        question + " " + " ".join(word for word in row if frequency[word] == 1)
        if any(frequency[word] == 1 for word in row)
        else question + " " + option
        for option, row in zip(options, words, strict=True)
    ]


def merge(intervals: list[list[float]], gap: float = 2.0) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


@app.function(image=IMAGE, gpu="A100-80GB", volumes={"/data": VOLUME}, timeout=8 * 60 * 60)
def build(beam: int = 2, output_name: str = "kite9_option_only_evidence.json") -> dict[str, object]:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor

    VOLUME.reload()
    questions = json.loads((ROOT / "manifests/test_questions.json").read_text())
    legacy = json.loads((ROOT / "predictions/test/kite7_rrf_evidence.json").read_text())
    output = ROOT / "predictions/test" / output_name
    existing = json.loads(output.read_text()) if output.exists() else {}
    cache = ROOT / "models/huggingface"
    siglip_name = "google/siglip2-so400m-patch14-384"
    clap_name = "laion/clap-htsat-fused"
    siglip_processor = AutoProcessor.from_pretrained(siglip_name, cache_dir=cache)
    siglip = AutoModel.from_pretrained(siglip_name, cache_dir=cache, dtype=torch.float16).to("cuda").eval()
    clap_processor = ClapProcessor.from_pretrained(clap_name, cache_dir=cache)
    clap = ClapModel.from_pretrained(clap_name, cache_dir=cache).to("cuda").eval()

    # Cache text features once per question; media/index reads remain resumable.
    records = dict(existing)
    for position, (answer_id, question) in enumerate(questions.items(), 1):
        if answer_id in records and records[answer_id].get("intervals"):
            continue
        video_id = str(question["metadata"]["original_video_id"])
        sig_rows_path = ROOT / "embeddings" / f"{video_id}.siglip2.jsonl"
        sig_npy_path = ROOT / "embeddings" / f"{video_id}.siglip2.npy"
        clap_rows_path = ROOT / "embeddings" / f"{video_id}.clap.jsonl"
        clap_npy_path = ROOT / "embeddings" / f"{video_id}.clap.npy"
        if not all(path.exists() for path in (sig_rows_path, sig_npy_path, clap_rows_path, clap_npy_path)):
            old = legacy.get(answer_id, {})
            records[answer_id] = {
                "answer_id": answer_id,
                "video_id": video_id,
                "intervals": old.get("intervals", []) or [[0.0, 1.0]],
                "sources": [{"kind": "legacy_rrf_index_fallback"}],
                "policy": "option-only-index-fallback",
                "missing_index": not bool(old.get("intervals")),
                "fallback_uniform": not bool(old.get("intervals")),
            }
            continue
        queries = option_queries(question["question"], question["options"])
        with torch.inference_mode():
            sig_inputs = siglip_processor(text=queries, padding="max_length", truncation=True, return_tensors="pt")
            sig_inputs = {key: value.to("cuda") for key, value in sig_inputs.items()}
            sig_features = torch.nn.functional.normalize(siglip.get_text_features(**sig_inputs), dim=-1).float()
            clap_inputs = clap_processor(text=queries, return_tensors="pt", padding=True)
            clap_inputs = {key: value.to("cuda") for key, value in clap_inputs.items()}
            clap_features = clap.get_text_features(**clap_inputs)
            clap_features = torch.nn.functional.normalize(getattr(clap_features, "pooler_output", clap_features), dim=-1).float()
        sig_embeddings = torch.from_numpy(np.load(sig_npy_path).astype("float32")).to("cuda")
        clap_embeddings = torch.from_numpy(np.load(clap_npy_path).astype("float32")).to("cuda")
        with torch.inference_mode():
            sig_scores = sig_embeddings @ sig_features.T
            clap_scores = clap_embeddings @ clap_features.T
        sig_rows = [json.loads(line) for line in sig_rows_path.read_text().splitlines()]
        clap_rows = [json.loads(line) for line in clap_rows_path.read_text().splitlines()]
        centers: list[float] = []
        sources: list[dict[str, object]] = []
        for option_index in range(len(queries)):
            for raw_index in torch.argsort(sig_scores[:, option_index], descending=True).tolist()[:beam]:
                timestamp = float(sig_rows[raw_index]["timestamp"])
                if all(abs(timestamp - old) >= 12.0 for old in centers):
                    centers.append(timestamp)
                    sources.append({"kind": "siglip_option", "option": option_index, "timestamp": timestamp})
            for raw_index in torch.argsort(clap_scores[:, option_index], descending=True).tolist()[:beam]:
                start = float(clap_rows[raw_index]["start"])
                timestamp = start + float(clap_rows[raw_index].get("window_seconds", 10.0)) / 2.0
                if all(abs(timestamp - old) >= 12.0 for old in centers):
                    centers.append(timestamp)
                    sources.append({"kind": "clap_option", "option": option_index, "timestamp": timestamp})
        intervals = merge([[max(0.0, center - 12.0), center + 12.0] for center in centers])
        records[answer_id] = {"answer_id": answer_id, "video_id": video_id, "intervals": intervals, "centers": centers, "sources": sources, "beam": beam, "policy": "option-only-discriminative-siglip-clap"}
        if position == 1 or position % 10 == 0:
            output.write_text(json.dumps(records, indent=2) + "\n")
            VOLUME.commit()
            print(f"[option-only] {position}/{len(questions)}", flush=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    VOLUME.commit()
    return {"questions": len(questions), "records": len(records), "indexed": sum(bool(row.get("intervals")) for row in records.values()), "output": str(output), "beam": beam}


@app.local_entrypoint()
def main(beam: int = 2) -> None:
    print(build.remote(beam))
