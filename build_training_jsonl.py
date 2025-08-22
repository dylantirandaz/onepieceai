"""
Build training-ready JSONL from scraped One Piece dataset.

Input (choose one):
  - out/onepiece_chapters.jsonl  (preferred; produced by the scraper)
  - out/onepiece_chapters.csv

Outputs:
  - train_next_beats.jsonl              (prompt/completion pairs)
  - train_synopsis_from_beats.jsonl     (prompt/completion pairs)
  - train_title_sugg.jsonl              (prompt/completion pairs)
  - train_thread_resolution.jsonl       (prompt/completion pairs, weak labels)
  - multitask_mix.jsonl                 (multi-task amalgam, with task tags)
  - rag_corpus.jsonl                    (RAG-friendly chunks)
  - splits/{train,val,test}.ids.txt     (chapter ID splits for reproducibility)

Notes:
- All generated data inherits CC BY-SA when derived from CC BY-SA summaries.
- The script uses conservative summarization and does not include any copyrighted manga text.

Usage:
    python build_training_jsonl.py --in out/onepiece_chapters.jsonl --out out_train --context 8
"""
import os, json, csv, re, random, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except:
                    pass
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        rows = df.fillna("").to_dict(orient="records")
        list_fields = ["beats_predicate","characters_core","locations_core","threads_opened","threads_closed"]
        for r in rows:
            for lf in list_fields:
                v = r.get(lf, "")
                if isinstance(v, str) and v.startswith("["):
                    try: r[lf] = json.loads(v)
                    except: r[lf] = []
    else:
        raise ValueError("Unsupported input type. Use .jsonl or .csv")
    for r in rows:
        try:
            r["chapter_id"] = int(r.get("chapter_id", 0))
        except:
            r["chapter_id"] = 0
    rows = [r for r in rows if r["chapter_id"] > 0]
    rows.sort(key=lambda x: x["chapter_id"])
    return rows

def clean_str(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def join_list(xs: List[str], sep=", ") -> str:
    return sep.join([clean_str(x) for x in xs if clean_str(x)])

def make_context_block(rows: List[Dict[str, Any]], idx: int, k: int) -> str:
    """
    Build a short context window of previous k chapters (beats + 1-line summary).
    """
    start = max(0, idx - k)
    prev = rows[start:idx]
    blocks = []
    for r in prev[-k:]:
        cid = r["chapter_id"]
        title = clean_str(r.get("title_en","")) or f"Chapter {cid}"
        beats = r.get("beats_predicate") or []
        beats_txt = "; ".join([clean_str(b) for b in beats]) if beats else ""
        summ = clean_str(r.get("summary_short_paraphrase",""))
        snippet = beats_txt if beats_txt else summ[:240]
        block = f"- Ch.{cid} {title}: {snippet}"
        blocks.append(block)
    return "\n".join(blocks)

def infer_simple_threads(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Weak heuristic if threads fields are empty: treat new named entities as opened,
    and if a name disappears for >N chapters later (not handled here), you'd mark closed.
    Here we just pass existing fields through or fallback to [] to stay safe.
    """
    opened = record.get("threads_opened") or []
    closed = record.get("threads_closed") or []
    return (opened, closed)

def build_next_beats_examples(rows: List[Dict[str, Any]], ctx_k: int) -> List[Dict[str, Any]]:
    examples = []
    for i, r in enumerate(rows):
        if i == 0: continue
        prev_ctx = make_context_block(rows, i, ctx_k)
        curr_id = r["chapter_id"]
        beats = r.get("beats_predicate") or []
        if not beats:
            summ = clean_str(r.get("summary_short_paraphrase",""))
            if not summ: continue
            sent = re.split(r"(?<=[.!?])\s+", summ)
            beats = [s.strip() for s in sent if len(s.strip())>0][:3]
        completion_obj = {"beats": beats}
        ex = {
            "chapter_id": curr_id,
            "prompt": (
                "You are a One Piece story forecaster. Using only the prior chapter context, "
                "predict the *core beats* of the next chapter in 3–7 bullets. Keep them short.\n\n"
                f"### Prior context (last {ctx_k}):\n{prev_ctx}\n\n"
                f"### Task: Predict beats for Chapter {curr_id}\n"
            ).strip(),
            "completion": json.dumps(completion_obj, ensure_ascii=False)
        }
        examples.append(ex)
    return examples

def build_synopsis_from_beats(rows: List[Dict[str, Any]], ctx_k: int) -> List[Dict[str, Any]]:
    examples = []
    for i, r in enumerate(rows):
        if i == 0: continue
        curr_id = r["chapter_id"]
        beats = r.get("beats_predicate") or []
        summary = clean_str(r.get("summary_short_paraphrase",""))
        if not summary: 
            continue
        if not beats:
            sent = re.split(r"(?<=[.!?])\s+", summary)
            beats = [s.strip() for s in sent if len(s.strip())>0][:3]
        prompt = (
            "You are a chapter writer. Expand the provided beats into a tight 2–3 paragraph synopsis "
            "without quoting lines. Keep proper nouns and events.\n\n"
            f"Beats for Chapter {curr_id}:\n- " + "\n- ".join(beats) + "\n"
        )
        completion = summary
        examples.append({"chapter_id": curr_id, "prompt": prompt.strip(), "completion": completion})
    return examples

def build_title_sugg(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for r in rows:
        cid = r["chapter_id"]
        title = clean_str(r.get("title_en",""))
        beats = r.get("beats_predicate") or []
        summ = clean_str(r.get("summary_short_paraphrase",""))
        context = ""
        if beats: context = "Beats:\n- " + "\n- ".join(beats) + "\n"
        elif summ: context = "Summary:\n" + summ + "\n"
        else: continue
        prompt = (
            "Suggest 3 concise chapter title candidates that fit Oda's style (evocative, spoiler-light). "
            "Return JSON list of strings.\n\n" + context
        )
        completion = json.dumps([title], ensure_ascii=False) if title else json.dumps([], ensure_ascii=False)
        examples.append({"chapter_id": cid, "prompt": prompt.strip(), "completion": completion})
    return examples

def build_thread_resolution(rows: List[Dict[str, Any]], ctx_k: int) -> List[Dict[str, Any]]:
    examples = []
    for i, r in enumerate(rows):
        if i == 0: continue
        cid = r["chapter_id"]
        opened, closed = infer_simple_threads(r)
        if not opened and not closed:
            continue
        prev_ctx = make_context_block(rows, i, ctx_k)
        prompt = (
            "Given the prior context, predict which ongoing threads get *opened* or *closed* in the next chapter. "
            "Return a JSON object with keys 'opened' and 'closed' (arrays of short labels).\n\n"
            f"### Prior context (last {ctx_k}):\n{prev_ctx}\n\n"
            f"### Predict thread changes for Chapter {cid}\n"
        )
        completion = json.dumps({"opened": opened, "closed": closed}, ensure_ascii=False)
        examples.append({"chapter_id": cid, "prompt": prompt.strip(), "completion": completion})
    return examples

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def make_splits(rows: List[Dict[str, Any]], out_dir: Path, test_ratio: float = 0.1, val_ratio: float = 0.1):
    ids = [r["chapter_id"] for r in rows]
    ids = sorted(list(set(ids)))
    n = len(ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_ids = ids[-n_test:]
    val_ids = ids[-(n_test + n_val):-n_test]
    train_ids = ids[:-(n_test + n_val)]
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits" / "train.ids.txt").write_text("\n".join(map(str, train_ids)), encoding="utf-8")
    (out_dir / "splits" / "val.ids.txt").write_text("\n".join(map(str, val_ids)), encoding="utf-8")
    (out_dir / "splits" / "test.ids.txt").write_text("\n".join(map(str, test_ids)), encoding="utf-8")
    return set(train_ids), set(val_ids), set(test_ids)

def filter_by_ids(rows: List[Dict[str, Any]], idset: set) -> List[Dict[str, Any]]:
    return [r for r in rows if r["chapter_id"] in idset]

def build_rag_corpus(rows: List[Dict[str, Any]], ctx_k: int) -> List[Dict[str, Any]]:
    corpus = []
    for r in rows:
        cid = r["chapter_id"]
        title = clean_str(r.get("title_en","")) or f"Chapter {cid}"
        beats = r.get("beats_predicate") or []
        summ = clean_str(r.get("summary_short_paraphrase",""))
        doc = "Beats: " + "; ".join(beats) + "\nSummary: " + summ
        corpus.append({
            "id": f"ch_{cid}",
            "title": title,
            "chapter_id": cid,
            "arc": clean_str(r.get("arc","")),
            "text": doc.strip()
        })
    return corpus

def build_multitask_mix(beats, synopsis, titles, threads) -> List[Dict[str, Any]]:
    def tag(task, ex):
        return {
            "chapter_id": ex["chapter_id"],
            "prompt": f"[TASK={task}]\n" + ex["prompt"],
            "completion": ex["completion"]
        }
    mix = []
    for ex in beats:   mix.append(tag("NEXT_BEATS", ex))
    for ex in synopsis: mix.append(tag("SYNOPSIS_FROM_BEATS", ex))
    for ex in titles:  mix.append(tag("TITLE_SUGGEST", ex))
    for ex in threads: mix.append(tag("THREAD_RESOLUTION", ex))
    random.shuffle(mix)
    return mix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input dataset (jsonl or csv)")
    ap.add_argument("--out", dest="out_dir", type=str, default="out_train", help="Output directory")
    ap.add_argument("--context", type=int, default=8, help="How many previous chapters to include in prompts")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(inp)
    if not rows:
        raise SystemExit("No rows loaded. Check your input path.")

    beats = build_next_beats_examples(rows, ctx_k=args.context)
    synopsis = build_synopsis_from_beats(rows, ctx_k=args.context)
    titles = build_title_sugg(rows)
    threads = build_thread_resolution(rows, ctx_k=args.context)
    mix = build_multitask_mix(beats, synopsis, titles, threads)
    rag = build_rag_corpus(rows, ctx_k=args.context)

    train_ids, val_ids, test_ids = make_splits(rows, out_dir)

    def split_and_write(name, exs):
        train = [e for e in exs if e["chapter_id"] in train_ids]
        val = [e for e in exs if e["chapter_id"] in val_ids]
        test = [e for e in exs if e["chapter_id"] in test_ids]
        write_jsonl(out_dir / f"{name}.train.jsonl", train)
        write_jsonl(out_dir / f"{name}.val.jsonl", val)
        write_jsonl(out_dir / f"{name}.test.jsonl", test)

    split_and_write("train_next_beats", beats)
    split_and_write("train_synopsis_from_beats", synopsis)
    split_and_write("train_title_sugg", titles)
    split_and_write("train_thread_resolution", threads)
    split_and_write("multitask_mix", mix)

    write_jsonl(out_dir / "rag_corpus.jsonl", rag)

    readme = f"""
# Training JSONL Outputs

Generated from: `{inp}`
Context window: {args.context}

Files:
- train_next_beats.[train|val|test].jsonl
- train_synopsis_from_beats.[train|val|test].jsonl
- train_title_sugg.[train|val|test].jsonl
- train_thread_resolution.[train|val|test].jsonl
- multitask_mix.[train|val|test].jsonl
- rag_corpus.jsonl
- splits/train.ids.txt, val.ids.txt, test.ids.txt

## Fine-tuning formats
Each JSONL line is:
{{"prompt": "...","completion":"...","chapter_id": N}}

- For OpenAI fine-tuning: feed the files directly (rename to your liking).
- For HuggingFace SFT (trl): map to a single text field:
  prompt + "\\n\\n### ANSWER\\n" + completion

## RAG
`rag_corpus.jsonl` has `id`, `title`, `chapter_id`, `arc`, `text`. Embed `text` and retrieve by `chapter_id` proximity for forecasting.

## License
Content derived from CC BY-SA summaries → your outputs inherit CC BY-SA. Keep attribution in your data pipeline.
"""
    (out_dir / "README_TRAIN.md").write_text(readme.strip(), encoding="utf-8")

    print("Wrote training sets to:", out_dir)

if __name__ == "__main__":
    main()
