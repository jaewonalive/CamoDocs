#!/usr/bin/env python3
"""Recombine sharded Stage-3 outputs into a single adv-text file.

Stage 3 (mix_and_create_adv_result.py) can be run data-parallel by launching N
copies with --manual_world_size N --manual_rank 0..N-1; each writes its own
JSON. This merges them and renames 'adv_texts_concat' (the merged
benign+adversarial documents) to 'adv_texts', which is the field the Stage-4
evaluator reads via --custom_attack_path.

Usage:
    python merge_shards.py --shard_dir ${DATA_ROOT}/stage3_output/camodocs- \
                           --output    ${DATA_ROOT}/camodocs_adv_text_merged.json
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--shard_dir", required=True, help="directory holding the per-rank JSON shards")
ap.add_argument("--output", required=True, help="path of the merged JSON to write")
ap.add_argument("--indent", type=int, default=2)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.shard_dir, "**", "*.json"), recursive=True))
if not files:
    raise SystemExit(f"no JSON shards found under {args.shard_dir}")

merged, skipped = {}, 0
origin = {}            # qid -> the shard it came from, for duplicate reporting
conflicts = []
for fp in files:
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for qid, rec in data.items():
        if not isinstance(rec, dict):
            raise SystemExit(
                f"{fp}: record for {qid!r} is {type(rec).__name__}, expected an object. "
                "Is this a Stage-3 shard?"
            )
        if "adv_texts_concat" not in rec:          # not finished by Stage 3
            skipped += 1
            continue
        rec["adv_texts"] = rec.pop("adv_texts_concat")
        if qid in merged:
            conflicts.append((qid, origin[qid], fp))
        merged[qid] = rec
        origin[qid] = fp
        n += 1
    print(f"  {os.path.basename(fp)}: {n} records")

print(f"\nshards read    : {len(files)}")
print(f"queries merged : {len(merged)}")
if skipped:
    print(f"skipped (no adv_texts_concat, Stage 3 unfinished): {skipped}")

# Fail before writing: a duplicate means the input set is wrong (e.g. shards
# from two Stage-3 runs in one directory). Which record would win depends on
# filename order, so the merge would be arbitrary rather than incorrect-but-
# deterministic.
if conflicts:
    print(f"\nERROR: {len(conflicts)} duplicate qid(s) across shards:")
    for qid, first, second in conflicts[:10]:
        print(f"  {qid}: {os.path.basename(first)} vs {os.path.basename(second)}")
    if len(conflicts) > 10:
        print(f"  ... and {len(conflicts) - 10} more")
    raise SystemExit(
        "Refusing to write. Merge shards from a single Stage-3 run "
        "(one --manual_world_size), or point --shard_dir at just that run."
    )

if not merged:
    raise SystemExit(
        f"Refusing to write: no shard contained a finished record "
        f"({skipped} skipped). Stage 3 may not have completed."
    )

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=args.indent)
print(f"-> {args.output}")
