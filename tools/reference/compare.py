#!/usr/bin/env python3
"""
Compare instantcua engine golden output directory against tools/reference/qwen3vl_ref.py golden data.
Reports:
- Vision Encoder pooler & deepstack cosine similarity and max abs diff
- First-token logit cosine similarity, max abs diff, and top-1 agreement
- Teacher-forced / greedy action accuracy and bounding-box hit rate on the 10 GUI benchmark scenes
"""
import json
import os
import re
import sys
import numpy as np


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def parse_click(text: str):
    m = re.search(r"click\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def main():
    ref_dir = "scratch/golden"
    eng_dir = sys.argv[1] if len(sys.argv) > 1 else "scratch/out_f16"

    with open(os.path.join(ref_dir, "summary.json")) as f:
        ref_summary = json.load(f)

    if os.path.exists(os.path.join(eng_dir, "task0_pooler.bin")):
        ref_pooler = np.load(os.path.join(ref_dir, "task0_pooler.npy"))
        eng_pooler = np.fromfile(os.path.join(eng_dir, "task0_pooler.bin"), dtype=np.float32).reshape(ref_pooler.shape)
        print(f"Vision Pooler   : cos={cos_sim(ref_pooler, eng_pooler):.6f}  max_diff={np.max(np.abs(ref_pooler - eng_pooler)):.5f}")

    for di in range(3):
        p = os.path.join(eng_dir, f"task0_deepstack_{di}.bin")
        if os.path.exists(p):
            ref_ds = np.load(os.path.join(ref_dir, f"task0_deepstack_{di}.npy"))
            eng_ds = np.fromfile(p, dtype=np.float32).reshape(ref_ds.shape)
            print(f"DeepStack[{di}]    : cos={cos_sim(ref_ds, eng_ds):.6f}  max_diff={np.max(np.abs(ref_ds - eng_ds)):.5f}")

    if os.path.exists(os.path.join(eng_dir, "task0_first_logits.bin")):
        ref_l = np.load(os.path.join(ref_dir, "task0_first_logits.npy"))
        eng_l = np.fromfile(os.path.join(eng_dir, "task0_first_logits.bin"), dtype=np.float32)
        print(
            f"First Logits    : cos={cos_sim(ref_l, eng_l):.6f}  max_diff={np.max(np.abs(ref_l - eng_l)):.5f}  "
            f"ref_top1={int(np.argmax(ref_l))} eng_top1={int(np.argmax(eng_l))}"
        )

    eng_summary_path = os.path.join(eng_dir, "summary.json")
    if os.path.exists(eng_summary_path):
        with open(eng_summary_path) as f:
            eng_summary = json.load(f)
        exact_matches = 0
        box_hits = 0
        ref_box_hits = 0
        for r, e in zip(ref_summary, eng_summary):
            same = r["text"].strip() == e["text"].strip()
            if same:
                exact_matches += 1
            box = r.get("targetBox1000")
            rc = parse_click(r["text"])
            ec = parse_click(e["text"])
            r_hit = bool(box and rc and box[0] <= rc[0] <= box[2] and box[1] <= rc[1] <= box[3])
            e_hit = bool(box and ec and box[0] <= ec[0] <= box[2] and box[1] <= ec[1] <= box[3])
            if r_hit:
                ref_box_hits += 1
            if e_hit:
                box_hits += 1
            print(f"  {r['id']:22s} | ref: {r['text'].strip():20s} ({'HIT' if r_hit else 'MISS'}) | eng: {e['text'].strip():20s} ({'HIT' if e_hit else 'MISS'}) | {'==' if same else '!='}")
        n = len(ref_summary)
        print(f"\nIdentical greedy outputs: {exact_matches}/{n}")
        print(f"Target bounding-box hits: engine {box_hits}/{n} (fp32 ref {ref_box_hits}/{n})")


if __name__ == "__main__":
    main()
