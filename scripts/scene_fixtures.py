"""Instruction and world fixture selection for the independent test harness."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 1-based instruction row. Observations exercise extraction, navigation and
# vision with the physical grasp disabled for uncalibrated object shapes.
CASES = [
    ("two_items_19", "complete", 7, "transfer"),
    ("one_item_07", "complete", 3, "observe"),
    ("all_duplicate_objects_11", "complete", 1, "observe"),
    ("zero_items_01", "complete", 5, "observe"),
    ("three_items_11", "incomplete", 1, "clarify"),
]


TRANSFER_CASES = [
    ("two_items_19", "complete", 7, "transfer"),
    ("one_item_07", "complete", 3, "transfer"),
    ("one_item_13", "complete", 5, "transfer"),
    ("two_items_02", "complete", 1, "transfer"),
    ("one_item_10", "complete", 6, "transfer"),
]


def read(path):
    return json.loads(path.read_text())


def fixtures(world, suite, index):
    table = ROOT / "tests/worlds" / (world.rsplit("_", 1)[0] + ".json")
    scene = next(row for row in read(table) if row["id"] == world)
    sdf = (table.parent / scene["world_file"]).resolve()
    manifest = read(table.parent / scene["placements_file"])
    if hashlib.sha256(sdf.read_bytes()).hexdigest() != manifest["world_sha256"] or manifest["case"] != world:
        raise ValueError(f"World/manifest mismatch for {world}")
    instruction = read(ROOT / "tests/instructions" / f"{suite}.json")[index-1]
    return scene, manifest, instruction
