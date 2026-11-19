"""Artifacts from one graph invocation. Evaluation reads these after control exits."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from target_selection import numbered_boxes


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class RunEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

    def call(self, record):
        with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def camera(self, image, metadata, depth=None):
        image.save(self.directory / "source.png")
        write_json(self.directory / "camera.json", metadata)
        if depth is not None:
            np.save(self.directory / "depth.npy", depth, allow_pickle=False)

    def geometry(self, boxes, plan=None, *, selected=None, spatial_only=False):
        metadata = json.loads((self.directory / "camera.json").read_text())
        pixel = None
        if plan and "odom_from_camera" in metadata:
            # Invert the same timestamped rigid transform used for unprojection.
            matrix = np.array(metadata["odom_from_camera"])
            xyz = matrix[:3, :3].T @ (np.array(plan["center"]) - matrix[:3, 3])
            uvw = np.array(metadata["k"]).reshape(3, 3) @ xyz
            if uvw[2] > 0:
                pixel = (uvw[:2] / uvw[2]).tolist()
        write_json(self.directory / "geometry.json", {
            "boxes_xyxy_px": boxes, "selected_box_xyxy_px": selected, "frame": "odom" if plan else None, "units": "m" if plan else None,
            "plan": plan, "projected_target_center_px": pixel,
            **({"projected_cube_center_px": pixel} if not plan or plan.get("shape", "cube") == "cube" else {})})
        annotate(self.directory / "source.png", self.directory / "annotated.png", boxes, plan, pixel, selected, spatial_only=spatial_only)

    def finish(self, result):
        write_json(self.directory / "result.json", {k: v for k, v in result.items()
                   if k not in ("image_url", "chat_history")})


def annotate(source, output, boxes, plan=None, pixel=None, selected=None, *, spatial_only=False):
    with Image.open(source) as original:
        image = original.convert("RGB")
    if spatial_only:
        lines = ["Matching objects in the wrist-camera view."]
        if selected is not None:
            lines.append("Yellow outline shows the selected object.")
    elif len(boxes) > 1:
        lines = [f"{len(boxes)} matching objects, numbered from left to right."]
        if selected is not None:
            lines.append(f"Selected box {numbered_boxes(boxes).index(selected) + 1}   {selected} px")
    else:
        lines = ["Model box (x1, y1, x2, y2) px   " + str(boxes)]
    if plan:
        for label, key in (("Object center", "center"), ("Grasp target (arm flange)", "grasp")):
            lines.append(f"{label}   ({', '.join(f'{v:.4f}' for v in plan[key])}) m   [odom]")
        lines.append("Cross = projected object center" if pixel is not None else
                     "3D coordinates from run log; camera transform unavailable for overlay")
    canvas = Image.new("RGB", (image.width, image.height + 16 + 23*len(lines)), "#172027")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for number, box in enumerate(numbered_boxes(boxes), 1):
        color = "yellow" if box == selected else "#38f1d2"
        draw.rectangle(box, outline=color, width=3)
        if len(boxes) > 1 and not spatial_only:
            draw.text((box[0]+3, box[1]+3), str(number), font=font, fill=color, stroke_width=2, stroke_fill="black")
    if pixel is not None and 0 <= pixel[0] < image.width and 0 <= pixel[1] < image.height:
        x, y = pixel
        draw.line((x-6, y, x+6, y), fill="yellow", width=2)
        draw.line((x, y-6, x, y+6), fill="yellow", width=2)
    for i, line in enumerate(lines):
        draw.text((10, image.height+8+23*i), line, fill="#eff7fa", font=font)
    canvas.save(output)
