#!/usr/bin/env python3
"""Generate exactly 20 cases for each independently defined world section.

Run all sections: python3 scripts/generate_world_scenes.py
Run one section: python3 scripts/generate_world_scenes.py --section two_items

Both output folders are created automatically, relative to the repository root.
Fixtures stay in tests/worlds/*.json; SDF scenes and spawn manifests stay flat
in worlds/. Generation is deterministic, with fixed 190 mm-separated slots.
An item type is a colour/shape pair. Zero-item trials share an empty scene;
one-item trials repeat after the 18 possible item/table inventories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/empty_table_world.sdf"
SECTIONS = ("zero_items", "one_item", "two_items", "three_items",
            "all_duplicate_objects", "mixed_duplicates", "all_unique_objects")
CASES_PER_SECTION = 20
TABLE_MODELS = {"A": "pick_table", "B": "place_table"}
COLORS = {"red": "0.9 0.015 0.015 1", "blue": "0.015 0.08 0.95 1",
          "green": "0.015 0.8 0.08 1"}
SHAPES = ("cube", "cylinder", "sphere")
ITEM_TYPES = tuple(product(COLORS, SHAPES))
SIZE_M = .05
MASS_KG = .10
FRICTION = 1.5
SAFETY_ZONE_RADIUS_M = .09
SLOT_OFFSETS_M = {0: (), 1: (0.0,), 2: (-.19, .19), 3: (-.19, 0.0, .19)}


def build_cases(section: str) -> list[dict]:
    """Build a section directly, rather than filtering a shared case pool."""
    cases = []
    for index in range(CASES_PER_SECTION):
        if section in SECTIONS[:4]:
            count = SECTIONS.index(section)
            pattern = ("same", "mixed", "unique")[index % 3] if count == 3 else (
                "same" if index % 2 == 0 else "unique")
        else:
            count = 3 if section == "mixed_duplicates" else 2 + index % 2
            pattern = {"all_duplicate_objects": "same", "mixed_duplicates": "mixed",
                       "all_unique_objects": "unique"}[section]

        item_index = (index // 2 if count == 1 else
                      index // 3 + 3 * (index % 3) if section == "three_items" else
                      index) % len(ITEM_TYPES)
        item = ITEM_TYPES[item_index]
        if pattern == "same":
            inventory = [item] * count
        elif pattern == "mixed":
            other_colour = tuple(COLORS)[(tuple(COLORS).index(item[0]) + 1) % 3]
            inventory = [item, item, (other_colour, item[1])]
        else:
            inventory = [ITEM_TYPES[(item_index + 4 * offset) % len(ITEM_TYPES)]
                         for offset in range(count)]
        # Count-zero and count-one sections cannot contain mixed inventories.
        inventory = inventory[:count]
        if count == 1:
            count_a = 1 - index % 2
        elif section in ("all_duplicate_objects", "all_unique_objects"):
            count_a = (index // 2 + index // 18) % (count + 1)
        elif count == 3:
            count_a = (index // 3) % (count + 1)
        else:
            count_a = (index + index // 18) % (count + 1)
        cases.append({
            "id": f"{section}_{index + 1:02d}",
            "table_A": [" ".join(item) for item in inventory[:count_a]],
            "table_B": [" ".join(item) for item in inventory[count_a:]],
        })
    return cases


def _table_data(world: ET.Element) -> dict:
    models = {model.get("name"): model for model in world.findall("model")}
    result = {}
    for table_id, model_name in TABLE_MODELS.items():
        model = models[model_name]
        pose = [float(part) for part in model.findtext("pose").split()]
        top = model.find("link/collision[@name='top_collision']")
        top_pose = [float(part) for part in top.findtext("pose").split()]
        top_size = [float(part) for part in top.findtext("geometry/box/size").split()]
        result[table_id] = {
            "model": model_name,
            "pose_world": pose,
            "top_center_xy_m": [pose[0] + top_pose[0], pose[1] + top_pose[1]],
            "top_height_m": pose[2] + top_pose[2] + top_size[2] / 2,
            "top_size_m": top_size,
        }
    return result


def _poses(table_data: dict, count: int) -> list[list[float]]:
    """Use known, separated slots in the supplied axis-aligned table template."""
    center_x, center_y = table_data["top_center_xy_m"]
    z = round(table_data["top_height_m"] + SIZE_M / 2 + .002, 6)
    return [[round(center_x - .10, 6), round(center_y + offset, 6), z, 0.0, 0.0, 0.0]
            for offset in SLOT_OFFSETS_M[count]]


def _fields(parent: ET.Element, **values: str) -> None:
    """Add a group of XML text fields in the supplied order."""
    for name, value in values.items():
        ET.SubElement(parent, name).text = value


def _dimensions(shape: str) -> list | dict:
    if shape == "cube":
        return [SIZE_M] * 3
    if shape == "cylinder":
        return {"radius": SIZE_M / 2, "length": SIZE_M}
    return {"radius": SIZE_M / 2}


def _geometry(parent: ET.Element, shape: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    primitive = ET.SubElement(geometry, "box" if shape == "cube" else shape)
    dimensions = _dimensions(shape)
    if shape == "cube":
        _fields(primitive, size=" ".join(map(str, dimensions)))
    else:
        _fields(primitive, **{key: str(value) for key, value in dimensions.items()})


def _inertia(shape: str) -> tuple[float, float, float]:
    if shape == "cube":
        axis = MASS_KG * SIZE_M ** 2 / 6
        return axis, axis, axis
    radius = SIZE_M / 2
    if shape == "cylinder":
        transverse = MASS_KG * (3 * radius ** 2 + SIZE_M ** 2) / 12
        return transverse, transverse, MASS_KG * radius ** 2 / 2
    axis = 2 * MASS_KG * radius ** 2 / 5
    return axis, axis, axis


def _add_object(world: ET.Element, item: dict) -> None:
    model = ET.SubElement(world, "model", name=item["name"])
    _fields(model, static="false",
            pose=" ".join(f"{number:.6f}" for number in item["pose_world"]))
    link = ET.SubElement(model, "link", name=item["shape"])
    inertial = ET.SubElement(link, "inertial")
    _fields(inertial, mass=f"{MASS_KG:.6f}")
    inertia = ET.SubElement(inertial, "inertia")
    ixx, iyy, izz = _inertia(item["shape"])
    _fields(inertia, ixx=f"{ixx:.12g}", iyy=f"{iyy:.12g}", izz=f"{izz:.12g}",
            ixy="0", ixz="0", iyz="0")

    collision = ET.SubElement(link, "collision", name="object_collision")
    _geometry(collision, item["shape"])
    surface = ET.SubElement(collision, "surface")
    friction = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction, "ode")
    _fields(ode, mu=f"{FRICTION:.6f}", mu2=f"{FRICTION:.6f}")

    visual = ET.SubElement(link, "visual", name="object_visual")
    _geometry(visual, item["shape"])
    material = ET.SubElement(visual, "material")
    _fields(material, ambient=COLORS[item["color"]], diffuse=COLORS[item["color"]])
    publisher = ET.SubElement(model, "plugin", filename="gz-sim-pose-publisher-system",
                              name="gz::sim::systems::PosePublisher")
    _fields(publisher, publish_model_pose="true", publish_link_pose="false",
            publish_nested_model_pose="false", use_pose_vector_msg="true", update_frequency="30")


def generate_world(case: dict, world_path: Path, *, table_xy=None) -> Path:
    """Write a scene and its independent initial spawn-pose record."""
    tree = ET.parse(TEMPLATE)
    world = tree.getroot().find("world")
    tables = _table_data(world)
    objects = []
    counts = defaultdict(int)
    for table_id in TABLE_MODELS:
        inventory = case[f"table_{table_id}"]
        poses = _poses(tables[table_id], len(inventory))
        if table_xy and table_id in table_xy:
            for pose, xy in zip(poses, table_xy[table_id], strict=True):
                pose[:2] = xy
        for label, pose in zip(inventory, poses):
            color, shape = label.split()
            counts[label] += 1
            item = {
                "name": f"{color}_{shape}_{counts[label]}",
                "shape": shape, "color": color, "table": table_id,
                "dimensions_m": _dimensions(shape), "mass_kg": MASS_KG,
                "pose_world": pose,
                "safety_zone": {"shape": "circle", "frame": "world",
                                "center_xy_m": pose[:2], "radius_m": SAFETY_ZONE_RADIUS_M},
            }
            _add_object(world, item)
            objects.append(item)
    ET.indent(tree, space="  ")
    world_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    report = {
        "version": 2, "case": case["id"],
        "pose_kind": "generated_spawn_pose_world; not measured after Gazebo settling",
        "world_file": world_path.name,
        "world_sha256": hashlib.sha256(world_bytes).hexdigest(),
        "tables": tables, "objects": objects,
    }
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_bytes(world_bytes)
    manifest_path = world_path.with_suffix(".placements.json")
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    return manifest_path


def generate_section(section: str, cases_dir: Path = ROOT / "tests/worlds",
                     scenes_dir: Path = ROOT / "worlds") -> list[dict]:
    """Regenerate one section's 20 fixtures/scenes without changing other sections."""
    cases_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / TEMPLATE.name).write_bytes(TEMPLATE.read_bytes())
    cases = build_cases(section)
    for case in cases:
        world_path = scenes_dir / f"{case['id']}.sdf"
        manifest_path = generate_world(case, world_path)
        # Resolve these references relative to the fixture JSON's directory.
        case["world_file"] = os.path.relpath(world_path, cases_dir)
        case["placements_file"] = os.path.relpath(manifest_path, cases_dir)
    (cases_dir / f"{section}.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=(*SECTIONS, "all"), default="all")
    args = parser.parse_args()
    sections = SECTIONS if args.section == "all" else (args.section,)
    for section in sections:
        cases = generate_section(section)
        print(f"{section}: {len(cases)} cases and world scenes")


if __name__ == "__main__":
    main()
