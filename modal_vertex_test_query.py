"""Checkpointed Vertex Gemini 3.6 test pass over cached multimodal evidence."""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time
from pathlib import Path
import modal

VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("google-genai").add_local_python_source("kite_core")
SECRET = modal.Secret.from_name("gcp-vertex-service-account")
app = modal.App("kilometer-audio-vertex-test-query")

def parse_answer(s: str):
    m = re.search(r"(?<!\d)([0-4])(?!\d)", s or "")
    return int(m.group(1)) if m else None

def render(source: Path, intervals, out: Path, max_intervals: int = 5):
    duration = source_duration(source)
    max_window = 90.0
    parts=[]
    normalized = []
    for raw in intervals:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        start, end = float(raw[0]), float(raw[1])
        if end < start:
            start, end = end, start
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end - start > max_window:
            center = (start + end) / 2.0
            start = max(0.0, center - max_window / 2.0)
            end = min(duration, start + max_window)
            start = max(0.0, end - max_window)
        if end - start >= 0.25:
            normalized.append((start, end))
    for i,(start,end) in enumerate(normalized[:max_intervals]):
        p=out.parent/f"p{i}.mp4"
        import subprocess
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(start),"-t",str(end-start),"-i",str(source),"-vf","fps=0.5,scale=320:-2","-c:v","libx264","-preset","ultrafast","-c:a","aac","-pix_fmt","yuv420p","-y",str(p)],check=True,timeout=240)
        parts.append(p)
    if not parts:
        raise ValueError(f"No valid evidence intervals for {source}")
    m=out.parent/"concat.txt"; m.write_text("\n".join(f"file '{p}'" for p in parts)+"\n")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",str(m),"-c","copy","-y",str(out)],check=True,timeout=240)

def source_duration(source: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return float(result.stdout.strip())

def fallback_intervals(source: Path, dense: bool = False) -> list[list[float]]:
    duration = source_duration(source)
    window = 12.0 if dense else 24.0
    count = min(20 if dense else 10, max(1, int(duration // (90 if dense else 180)) + 1))
    if count == 1:
        starts = [0.0]
    else:
        stride = max(1.0, (duration - window) / (count - 1))
        starts = [min(duration - window, index * stride) for index in range(count)]
    return [[max(0.0, start), min(duration, start + window)] for start in starts]

@app.function(image=IMAGE, volumes={"/data":VOLUME}, secrets=[SECRET], timeout=24*60*60)
def run(model="gemini-3.6-flash", cooldown_sec=30.0, shard=0, shards=1, fallback_only=False, dense_recovery=False, evidence_filename="kite3_evidence.json", output_tag="", max_intervals=5):
    from google import genai
    from google.genai import types
    VOLUME.reload()
    root=Path("/data/2706-kilometer-audio")
    q=json.loads((root/"manifests/test_questions.json").read_text())
    ev=json.loads((root/"predictions/test"/evidence_filename).read_text())
    fallback=json.loads((root/"predictions/test/kite8_gemini_evidence.json").read_text())
    outdir=root/("predictions/test/vertex-query-recovery-dense" if dense_recovery else "predictions/test/vertex-query-recovery" if fallback_only else ("predictions/test/" + output_tag if output_tag else "predictions/test/vertex-query")); outdir.mkdir(parents=True,exist_ok=True)
    suffix = "recovery-dense" if dense_recovery else "recovery" if fallback_only else output_tag or "shard"
    partial=outdir/f"gemini-3.6-flash-{suffix}-{shard}.partial.json"; rows=[]
    if partial.exists():
        try: rows=json.loads(partial.read_text())
        except Exception: rows=[]
    # A prior run may have checkpointed missing-source skips or unparsable
    # responses.  Once media is repaired those rows must be retried, rather
    # than being treated as completed and silently carried into the result.
    rows=[r for r in rows if r.get("prediction") is not None and not r.get("skipped")]
    done={r["answer_id"] for r in rows}
    creds=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds: raise RuntimeError("missing Vertex credentials")
    cp=Path(tempfile.gettempdir())/"vertex.json"; cp.write_text(creds); os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=str(cp)
    client=genai.Client(vertexai=True,project="veotest-460812",location="global",http_options=types.HttpOptions(timeout=180_000))
    with tempfile.TemporaryDirectory() as td:
      td=Path(td)
      for position,(aid,question) in enumerate(q.items()):
        if position % shards != shard: continue
        is_fallback = bool(fallback.get(aid,{}).get("fallback"))
        if aid in done or (fallback_only != is_fallback): continue
        item=ev.get(aid,{})
        vid=question.get("metadata",{}).get("original_video_id") or item.get("video_id")
        source=Path(f"/data/shared/walking-tours/{vid}/{vid}.mkv")
        if not source.exists():
            rows.append({"answer_id":aid,"prediction":None,"skipped":"missing_source"}); partial.write_text(json.dumps(rows)); continue
        clip=td/f"{len(rows)}.mp4"
        intervals = (fallback_intervals(source, dense=True) if item.get("fallback_uniform") else fallback_intervals(source, dense=dense_recovery)) if fallback_only or item.get("fallback_uniform") else item.get("intervals",[])
        if not intervals:
            rows.append({"answer_id": aid, "prediction": None, "skipped": "missing_evidence_index"})
            partial.write_text(json.dumps(rows, indent=2) + "\n")
            VOLUME.commit()
            continue
        render(source, intervals, clip, max_intervals=max_intervals)
        prompt=("Answer the multiple-choice question using only the attached selected walking-tour evidence. "
                "Inspect all intervals, reason silently about temporal order, counts, audio, and visual evidence, "
                "then return exactly one answer index digit 0, 1, 2, 3, or 4 and no explanation.\n"
                f"Question: {question['question']}\nOptions: " + " | ".join(f"{i}. {x}" for i,x in enumerate(question['options'])))
        response=None
        last_exc=None
        # One normal pacing delay, then at most one short retry.  Failed
        # quota/transient calls remain empty and are retried by later sweeps.
        for attempt in range(2):
          time.sleep(cooldown_sec if attempt==0 else 10)
          try:
            part=types.Part(inline_data=types.Blob(data=clip.read_bytes(),mime_type="video/mp4"),video_metadata=types.VideoMetadata(fps=0.5))
            response=client.models.generate_content(model=model,contents=[types.Content(role="user",parts=[part,types.Part.from_text(text=prompt)])],config=types.GenerateContentConfig(temperature=0,max_output_tokens=2048,thinking_config=types.ThinkingConfig(thinking_level="high")))
            break
          except Exception as exc:
            last_exc=exc
            retryable=any(x in str(exc) for x in ("429","504","DEADLINE_EXCEEDED"))
            if not retryable or attempt==1:
                break
        if response is None:
            rows.append({"answer_id": aid, "prediction": None, "deferred": str(last_exc)[:500], "intervals": intervals})
            partial.write_text(json.dumps(rows, indent=2) + "\n")
            VOLUME.commit()
            continue
        raw=getattr(response,"text","") or ""
        rows.append({"answer_id":aid,"prediction":parse_answer(raw),"raw":raw,"intervals":intervals,"fallback_recovery":fallback_only,"dense_recovery":dense_recovery})
        partial.write_text(json.dumps(rows,indent=2)+"\n")
        VOLUME.commit()
    (outdir/f"gemini-3.6-flash-{suffix}-{shard}.json").write_text(json.dumps({"model":model,"rows":rows,"completed":len(rows),"fallback_only":fallback_only,"evidence_filename":evidence_filename,"max_intervals":max_intervals},indent=2)+"\n")
    VOLUME.commit(); return {"completed":len(rows),"parsed":sum(r.get("prediction") is not None for r in rows),"skipped":sum(bool(r.get("skipped")) for r in rows),"fallback_only":fallback_only}

@app.function(image=IMAGE, volumes={"/data":VOLUME}, timeout=10*60)
def progress(shard: int = 0, dense_recovery: bool = False, fallback_only: bool = False) -> dict:
    VOLUME.reload()
    root = Path("/data/2706-kilometer-audio")
    outdir = root / ("predictions/test/vertex-query-recovery-dense" if dense_recovery else "predictions/test/vertex-query-recovery" if fallback_only else "predictions/test/vertex-query")
    path = outdir / f"gemini-3.6-flash-{'recovery-dense' if dense_recovery else 'recovery' if fallback_only else 'shard'}-{shard}.partial.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = json.loads(path.read_text())
    return {"exists": True, "path": str(path), "completed": len(rows), "parsed": sum(r.get("prediction") is not None for r in rows), "last_answer_id": rows[-1].get("answer_id") if rows else None}

@app.function(image=IMAGE, volumes={"/data":VOLUME}, timeout=30*60)
def aggregate_recovery() -> dict:
    from kite_core import build_submission, validate_predictions
    VOLUME.reload()
    root=Path("/data/2706-kilometer-audio")
    questions=json.loads((root/"manifests/test_questions.json").read_text())
    base=json.loads((root/"predictions/test/audio_rag_submission.json").read_text())
    recovery=json.loads((root/"predictions/test/vertex-query-recovery/gemini-3.6-flash-recovery-0.json").read_text())["rows"]
    selected={aid:int(row["answer_id"]) for aid,row in base.items()}
    records={}
    for row in recovery:
        if row.get("prediction") is not None:
            selected[row["answer_id"]]=int(row["prediction"])
        records[row["answer_id"]]=row
    submission=build_submission(questions,selected)
    errors=validate_predictions(questions,submission)
    if errors: raise ValueError("Invalid recovery submission: "+"; ".join(errors))
    output=root/"predictions/test"
    (output/"kite8_gemini_recovered_submission.json").write_text(json.dumps(submission,indent=2)+"\n")
    (output/"kite8_gemini_recovery_evidence.json").write_text(json.dumps({"base":"audio_rag_submission","recovery":records,"changed":[aid for aid in records if int(base[aid]["answer_id"]) != int(submission[aid]["answer_id"])]},indent=2)+"\n")
    VOLUME.commit()
    return {"questions":len(submission),"recovered":len(records),"parsed":sum(r.get("prediction") is not None for r in recovery),"changed":sum(int(base[aid]["answer_id"]) != int(submission[aid]["answer_id"]) for aid in records),"submission_path":str(output/"kite8_gemini_recovered_submission.json")}

@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def aggregate_option_only() -> dict:
    from kite_core import build_submission, validate_predictions
    VOLUME.reload()
    root = Path("/data/2706-kilometer-audio")
    questions = json.loads((root / "manifests/test_questions.json").read_text())
    base = json.loads((root / "predictions/test/kite8_gemini_submission.json").read_text())
    partials = sorted((root / "predictions/test/kite9-option-only").glob("gemini-3.6-flash-kite9-option-only-*.partial.json"))
    rows = []
    for partial in partials:
        rows.extend(json.loads(partial.read_text()))
    selected = {aid: int(row["answer_id"]) for aid, row in base.items()}
    changed = []
    for row in rows:
        if row.get("prediction") is None:
            continue
        aid = row["answer_id"]
        if selected[aid] != int(row["prediction"]):
            changed.append(aid)
        selected[aid] = int(row["prediction"])
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid option-only submission: " + "; ".join(errors))
    output = root / "predictions/test"
    (output / "kite9_option_only_submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    (output / "kite9_option_only_report.json").write_text(json.dumps({"base": "kite8_gemini_submission", "rows": len(rows), "parsed": sum(row.get("prediction") is not None for row in rows), "changed": changed, "skipped": sum(row.get("prediction") is None for row in rows)}, indent=2) + "\n")
    VOLUME.commit()
    return {"questions": len(submission), "new_rows": len(rows), "parsed": sum(row.get("prediction") is not None for row in rows), "changed": len(changed), "skipped": sum(row.get("prediction") is None for row in rows), "path": str(output / "kite9_option_only_submission.json")}

@app.function(image=IMAGE, volumes={"/data":VOLUME}, timeout=30*60)
def audit_fallback_sources() -> dict:
    VOLUME.reload()
    root=Path("/data/2706-kilometer-audio")
    questions=json.loads((root/"manifests/test_questions.json").read_text())
    ids=sorted({payload.get("metadata",{}).get("original_video_id") for payload in questions.values()})
    fallback_ids={payload.get("video_id") for payload in json.loads((root/"predictions/test/kite8_gemini_evidence.json").read_text()).values() if payload.get("fallback")}
    report=[]
    for vid in sorted(fallback_ids):
        source=Path("/data/shared/walking-tours")/vid/f"{vid}.mkv"
        duration=source_duration(source) if source.exists() else None
        probe=subprocess.run(["ffprobe","-v","error","-show_streams","-of","json",str(source)],capture_output=True,text=True,check=False) if source.exists() else None
        streams=json.loads(probe.stdout).get("streams",[]) if probe and probe.returncode==0 else []
        report.append({"video_id":vid,"exists":source.exists(),"duration_sec":duration,"size_bytes":source.stat().st_size if source.exists() else None,"video_streams":sum(s.get("codec_type")=="video" for s in streams),"audio_streams":sum(s.get("codec_type")=="audio" for s in streams),"audio_codecs":sorted({s.get("codec_name") for s in streams if s.get("codec_type")=="audio"})})
    return {"fallback_videos":len(report),"sources":report}

@app.function(image=IMAGE, volumes={"/data":VOLUME}, timeout=10*60)
def audit_required_sources() -> dict:
    VOLUME.reload()
    root = Path("/data/2706-kilometer-audio")
    questions = json.loads((root / "manifests/test_questions.json").read_text())
    videos = sorted({q.get("metadata", {}).get("original_video_id") for q in questions.values() if q.get("metadata", {}).get("original_video_id")})
    missing = [vid for vid in videos if not (Path("/data/shared/walking-tours") / vid / f"{vid}.mkv").exists()]
    return {"questions": len(questions), "unique_videos": len(videos), "present": len(videos) - len(missing), "missing": missing}

@app.local_entrypoint()
def main(cooldown_sec:float=30.0, shard:int=0, shards:int=1, fallback_only:bool=False, dense_recovery:bool=False, action:str="run", evidence_filename:str="kite3_evidence.json", output_tag:str="", max_intervals:int=5):
    if action == "aggregate_recovery":
        print(aggregate_recovery.remote())
    elif action == "aggregate_option_only":
        print(aggregate_option_only.remote())
    elif action == "audit_sources":
        print(audit_fallback_sources.remote())
    elif action == "audit_required_sources":
        print(audit_required_sources.remote())
    elif action == "progress":
        print(progress.remote(shard=shard, dense_recovery=dense_recovery, fallback_only=fallback_only))
    else:
        print(run.remote(cooldown_sec=cooldown_sec, shard=shard, shards=shards, fallback_only=fallback_only, dense_recovery=dense_recovery, evidence_filename=evidence_filename, output_tag=output_tag, max_intervals=max_intervals))
