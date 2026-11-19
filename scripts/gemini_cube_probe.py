"""Offline fixture/scoring and opt-in production-equivalent Gemini vision probe."""
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
import time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DIRECT_PROMPT = ('pick up the left-most red cube. Return the bounding box of only that cube. '
                 'Return box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000. Return an empty list if absent.')
ALL_PROMPT = ('Detect all red cube objects. Return box_2d as [ymin, xmin, ymax, xmax] '
              'normalized to 0-1000. Return an empty list if absent.')


def fixture(output):
    """Paste the same simulator cube twice; annotations exclude the shadow patch."""
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'tests/images/table_A.png'
    image = Image.open(source).convert('RGB')
    patch = image.crop((225, 82, 292, 151))
    # Same-size duplicates prevent size/colour from identifying the requested cube.
    image.paste(patch, (80, 165))
    image.paste(patch, (300, 275))
    refs = [[86, 171, 127, 217], [231, 88, 272, 134], [306, 281, 347, 327]]
    path = output / 'input.png'
    image.save(path)
    manifest = {'image': path.name, 'size': list(image.size), 'reference_boxes_xyxy': refs,
                'target_index': 0, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'coordinate_convention': 'half-open xyxy',
                'annotation': 'Original cube pixels x231..271 and y88..133 inclusive; half-open [231,88,272,134], translated with patches. Shadow excluded. Left-most means image horizontal center.'}
    (output / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return path, manifest


def to_pixels(box, width, height):
    if len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError('Expected four integer normalized coordinates')
    y1, x1, y2, x2 = box
    if not (0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000):
        raise ValueError('Expected ordered coordinates within 0–1000')
    return [round(x1 * width / 1000), round(y1 * height / 1000),
            round(x2 * width / 1000), round(y2 * height / 1000)]


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union else 0.0


def score(normalized, manifest, mode):
    pixels = [to_pixels(b, *manifest['size']) for b in normalized]
    refs = manifest['reference_boxes_xyxy']
    matrix = [[iou(box, ref) for ref in refs] for box in pixels]
    selected = min(range(len(pixels)), key=lambda i: (pixels[i][0] + pixels[i][2]) / 2) if pixels else None
    selected_iou = matrix[selected][manifest['target_index']] if selected is not None else 0
    # Direct success requires exactly one answer; all-mode requires three distinct detections.
    matches = [max(range(len(refs)), key=lambda j: row[j]) for row in matrix]
    detection_pass = len(pixels) == 3 and len(set(matches)) == 3 and all(max(row) >= .5 for row in matrix)
    return {'pixel_boxes_xyxy': pixels, 'iou_matrix': matrix, 'selected_prediction_index': selected,
            'selected_target_iou': selected_iou, 'count': len(pixels),
            'selection_pass': selected_iou >= .5 and (mode == 'all' or len(pixels) == 1),
            'all_detection_pass': detection_pass if mode == 'all' else None,
            'threshold': .5}


def overlay(path, manifest, report, output):
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(manifest['reference_boxes_xyxy']):
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], outline='lime', width=2)
        draw.text((box[0], box[1]-12), f'ref {i}', fill='lime')
    for i, box in enumerate(report['pixel_boxes_xyxy']):
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], outline='magenta', width=2)
        draw.text((box[0], box[3]+2), f'prediction {i}', fill='magenta')
    image.save(output)


def live(image, prompt, env_file, model):
    # No robot imports or other model calls; explicit environment path avoids cwd ambiguity.
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field, model_validator
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    load_dotenv(env_file)

    class BoundingBox(BaseModel):
        box_2d: list[int] = Field(min_length=4, max_length=4, description='[ymin, xmin, ymax, xmax] normalized to 0-1000.')

        @model_validator(mode='after')
        def valid_box(self):
            to_pixels(self.box_2d, 1000, 1000)
            return self

    class BoundingBoxes(BaseModel):
        boxes: list[BoundingBox]

    model = model or os.getenv('BOUNDING_BOX_MODEL', 'gemini-3.8-flash')
    client = ChatGoogleGenerativeAI(model=model, temperature=0, timeout=45, max_retries=1).with_structured_output(BoundingBoxes)
    url = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')
    start = time.monotonic()
    result = client.invoke([HumanMessage(content=[{'type': 'text', 'text': prompt},
                          {'type': 'image_url', 'image_url': {'url': url}}])])
    return result.model_dump(), model, time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/gemini-cube-probe')
    parser.add_argument('--mode', choices=['direct', 'all'], default='direct')
    parser.add_argument('--live', action='store_true', help='Explicitly enable a billable Gemini network call')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--model')
    args = parser.parse_args()
    path, manifest = fixture(args.output_dir)
    if not args.live:
        overlay(path, manifest, {'pixel_boxes_xyxy': []}, args.output_dir / 'reference.png')
        print(f'Fixture only (no API call): {path}')
        return
    if args.env_file is None:
        parser.error('--live requires explicit --env-file')
    prompt = DIRECT_PROMPT if args.mode == 'direct' else ALL_PROMPT
    raw, model, elapsed = live(path, prompt, args.env_file, args.model)
    report = score([b['box_2d'] for b in raw['boxes']], manifest, args.mode)
    evidence = {'mode': args.mode, 'prompt': prompt, 'model': model, 'elapsed_seconds': elapsed,
                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'raw_structured_response': raw,
                'scores': report, 'fixture': manifest}
    (args.output_dir / f'{args.mode}.json').write_text(json.dumps(evidence, indent=2) + '\n')
    overlay(path, manifest, report, args.output_dir / f'{args.mode}_overlay.png')
    print(json.dumps(evidence, indent=2))
    if not report['selection_pass'] or (args.mode == 'all' and not report['all_detection_pass']):
        sys.exit(1)


if __name__ == '__main__':
    main()
