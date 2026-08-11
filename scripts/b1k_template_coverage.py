#!/usr/bin/env python3
"""Run the instruction templates over every BEHAVIOR-1K skill segment.

Reports the render-failure rate and why each failure happened, which is the
G1 acceptance gate for the converter (docs/b1k/04-validation-plan.md).

Usage:
    python scripts/b1k_template_coverage.py <b1k_root> [--samples 40]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys

from dataset_upload.dataset_loaders.b1k_object_names import (
    OBJECT_NAME_MAP,
    PSEUDO_OBJECTS,
)
from dataset_upload.dataset_loaders.b1k_skill_templates import (
    TEMPLATES,
    render_instruction,
    spatial_kind,
)


def flat(value):
    if isinstance(value, list):
        for item in value:
            yield from flat(item)
    elif value is not None:
        yield item if False else value


def classify_failure(skill, objs, spatial):
    """Why did render_instruction return None?"""
    if not skill:
        return "no_skill"
    for obj in objs:
        if obj not in PSEUDO_OBJECTS and obj not in OBJECT_NAME_MAP:
            return "unknown_object"
    for token in spatial:
        try:
            if spatial_kind(token) is None:
                return "dropped_spatial_prefix(grid)"
        except KeyError:
            return f"unknown_spatial_prefix({token})"
    if not any(s == skill for s, _ in TEMPLATES):
        return "unknown_skill"
    return f"unsupported_arity({skill}|{len(objs)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "annotations/task-*/episode_*.json")))
    if not files:
        raise SystemExit(f"no annotations under {args.root}")

    total = 0
    ok = 0
    failures: collections.Counter = collections.Counter()
    instructions: collections.Counter = collections.Counter()
    per_skill_fail: collections.Counter = collections.Counter()
    samples: list[tuple[str, str]] = []
    rng = random.Random(0)

    for path in files:
        with open(path) as f:
            ann = json.load(f)
        for skill in ann.get("skill_annotation") or []:
            descs = skill.get("skill_description") or []
            desc = descs[0] if descs and descs[0] is not None else None
            spatial = [t for t in flat(skill.get("spatial_prefix") or [])]
            memory = [t for t in flat(skill.get("memory_prefix") or [])]
            for group in skill.get("object_id") or []:
                objs = list(flat(group))
                total += 1
                text = render_instruction(desc, [objs], memory, spatial)
                if text is None:
                    reason = classify_failure(desc, objs, spatial)
                    failures[reason] += 1
                    per_skill_fail[desc] += 1
                else:
                    ok += 1
                    instructions[text] += 1
                    if len(samples) < args.samples or rng.random() < 0.001:
                        entry = (f"{desc} {objs} m={memory} s={spatial}", text)
                        if len(samples) < args.samples:
                            samples.append(entry)
                        else:
                            samples[rng.randrange(len(samples))] = entry

    fail = total - ok
    print(f"annotation files      : {len(files)}")
    print(f"object groups (segments): {total}")
    print(f"rendered              : {ok}  ({ok / total:.2%})")
    print(f"failed                : {fail}  ({fail / total:.2%})")
    print(f"unique instructions   : {len(instructions)}")
    print()
    print("failure reasons:")
    for reason, n in failures.most_common():
        print(f"  {n:>7d}  {n / total:6.2%}  {reason}")
    print()
    print("most common instructions:")
    for text, n in instructions.most_common(12):
        print(f"  {n:>7d}  {text}")
    print()
    print("random rendered samples:")
    for src, text in samples[:25]:
        print(f"  {src}\n      -> {text}")

    gate = fail / total < 0.05
    print()
    print(f"G1 gate (failure rate < 5%): {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
