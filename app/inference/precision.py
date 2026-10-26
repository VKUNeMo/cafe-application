"""Class-precision gating; no labels from uploaded images are required."""

import csv
import math
from pathlib import Path


def load_precision_table(path: Path, classes: list[str]) -> dict[str, dict[str, float]]:
    table = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"class", "precision_big", "precision_small"} <= set(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Bảng precision cần các cột class, precision_big, precision_small."
            )
        for row in reader:
            name = row["class"]
            if name in table:
                raise ValueError(f"Lớp bị lặp trong bảng precision: {name}")
            scores = {
                branch: float(row[f"precision_{branch}"]) for branch in ("big", "small")
            }
            if any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in scores.values()
            ):
                raise ValueError(f"Precision không hợp lệ cho lớp {name}.")
            table[name] = scores
    if set(table) != set(classes):
        raise ValueError(
            "Tên lớp trong bảng precision không khớp checkpoint Big/Small."
        )
    return table


def select_by_precision(
    big: dict, small: dict, classes: list[str], table: dict
) -> dict:
    big_label, small_label = big["pred_class"], small["pred_class"]
    pb, ps = table[big_label]["big"], table[small_label]["small"]
    if big_label == small_label:
        choose_big = big["pred_conf"] >= small["pred_conf"]
        method = "agreement"
    else:
        choose_big = pb >= ps
        method = (
            "precision_tie_big"
            if pb == ps
            else "precision_big"
            if choose_big
            else "precision_small"
        )
    chosen, other = (big, small) if choose_big else (small, big)
    if big_label == small_label:
        order = sorted(
            range(len(classes)),
            key=lambda i: (float(chosen["probs"][i]), i),
            reverse=True,
        )
        second_idx = next(i for i in order if classes[i] != chosen["pred_class"])
        second, second_conf = classes[second_idx], float(chosen["probs"][second_idx])
    else:
        second, second_conf = other["pred_class"], float(other["pred_conf"])
    return {
        "pred_class": chosen["pred_class"],
        "confidence": float(chosen["pred_conf"]),
        "top2_class": second,
        "top2_confidence": second_conf,
        "method": method,
        "selected_branch": "big" if choose_big else "small",
        "precision_comparison": {
            "big_label": big_label,
            "big_precision": pb,
            "small_label": small_label,
            "small_precision": ps,
        },
    }
