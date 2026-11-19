"""Independent actual Gazebo pose verifier; no observations go to the controller."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
from scene_fixtures import fixtures, ROOT
from hashlib import sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--instruction', help='Defaults to the selected instruction fixture')
parser.add_argument('--world', default='two_items_19')
parser.add_argument('--instruction-row', type=int, default=7)
parser.add_argument('--reply', action='append', default=[], help='Test operator replies in the order requested')
parser.add_argument('--expect-leftmost', action='store_true', help='Verify the leftmost duplicate in the source camera frame')
parser.add_argument('--expect-position', choices=['leftmost','rightmost','middle','topmost','bottommost'])
parser.add_argument('--scene-manifest', type=Path, help='Generated trial world and spawn manifest')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
direction = args.expect_position or ('leftmost' if args.expect_leftmost else None)
if args.scene_manifest:
    manifest = json.loads(args.scene_manifest.read_text())
    sdf = args.scene_manifest.parent / manifest['world_file']
    if manifest['case'] != args.world or sha256(sdf.read_bytes()).hexdigest() != manifest['world_sha256']:
        raise ValueError('Trial world does not match spawn manifest')
    case = json.loads((ROOT/'tests/instructions/complete.json').read_text())[args.instruction_row-1]
    scene = {'table_'+t:[o['color']+' '+o['shape'] for o in manifest['objects'] if o['table']==t] for t in ('A','B')}
else:
    scene, manifest, case = fixtures(args.world, "complete", args.instruction_row)
expected = case["expected"]
source = expected["start_location"].split()[-1]
destination = expected["end_location"].split()[-1]
objects = manifest["objects"]
for table in (source, destination):
    inventory = Counter(o["color"]+" "+o["shape"] for o in objects if o["table"] == table)
    if inventory != Counter(scene["table_"+table]):
        raise ValueError("World inventory and placement manifest disagree")
candidates = [o["name"] for o in objects if o["table"] == source and o["color"]+" "+o["shape"] == expected["target_object"]]
if not candidates or (len(candidates) > 1 and not direction):
    parser.error('A duplicate trial needs an independent expected choice (--expect-leftmost)')
target = candidates[0] if len(candidates) == 1 else None
names = {o["name"] for o in objects}
latest={};maxima={};trace=[];lock=threading.Lock()
# Keep the native Gazebo subscriber in a dedicated process. Its C++ transport
# threads otherwise race interpreter teardown after a successfully written verdict.
observer = subprocess.Popen(["gz", "topic", "-e", "-t", "/world/default/pose/info", "--json-output"],
                            stdout=subprocess.PIPE, text=True, start_new_session=True)
reader_errors = []
def read_poses():
    try:
        for line in observer.stdout:
            poses = json.loads(line).get("pose", [])
            now = time.time()
            with lock:
                for pose in poses:
                    name = pose.get("name")
                    if name in names:
                        point = [pose["position"].get(axis, 0.) for axis in "xyz"]
                        latest[name] = point
                        maxima[name] = max(maxima.get(name, -math.inf), point[2])
                        trace.append({"wall_time": now, "name": name, "position": point})
    except (ValueError, KeyError, OSError) as exc:
        reader_errors.append(str(exc))

reader = threading.Thread(target=read_poses)
reader.start()
report={'passed':False,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'controller_source':'ROS RGB-D, CameraInfo and TF only',
        'verifier_source':'separate Gazebo pose subscriber, never supplied to controller',
        'physical_attachment_used':False, 'world':args.world, 'world_sha256':manifest['world_sha256'],
        'target_model':target, 'source_table':source, 'destination_table':destination}
process=None
try:
    deadline=time.monotonic()+30
    while not names.issubset(latest):
        if time.monotonic()>deadline:raise RuntimeError('Missing Gazebo fixture object poses')
        time.sleep(.1)
    time.sleep(1)
    with lock:initial={k:list(v) for k,v in latest.items()}
    report['initial_world_poses']=initial
    for obj in objects:
        if math.dist(initial[obj["name"]], obj["pose_world"][:3]) > .015:
            raise RuntimeError(f"Initial pose of {obj['name']} differs from the fixture")

    with (args.output/'controller.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'/agent/scripts/run_pick_place_task.py','--output',str(args.output), '--instruction', args.instruction or case['instruction'],
                                  *(['--spatial-choices-only'] if direction else []),
                                  *[word for reply in args.reply for word in ('--reply', reply)]],
                                 stdout=log,stderr=subprocess.STDOUT)
        result=process.wait(timeout=1200)
    report['controller_exit_code']=result
    time.sleep(3)
    if not reader.is_alive() or observer.poll() is not None or reader_errors:
        raise RuntimeError("Independent Gazebo pose stream stopped before verification")
    with lock:final={k:list(v) for k,v in latest.items()};maximum=dict(maxima)
    report['final_world_poses']=final;report['maximum_world_z']=maximum
    graph=json.loads((args.output/'graph-result.json').read_text())
    if direction:
        from score_spatial_choice import score_choice
        dialogue = graph.get('selection_dialogue', [])
        boxes = graph.get('boxes', [])
        replies = [turn['reply'] for turn in dialogue]
        report.update(direction=direction, dialogue=dialogue, boxes_detected=len(boxes),
                      graph_status=graph.get('execution_status'), graph_error=graph.get('error'),
                      clarification_correct=bool(replies and replies==args.reply and len(boxes)==len(candidates)),
                      selection_correct=False, selected_box=graph.get('selected_box'))
        if not report['clarification_correct']:
            raise RuntimeError("Expected full duplicate detection and the prescribed clarification dialogue")
        camera = json.loads((args.output/'evidence/camera.json').read_text())
        score = score_choice(initial,candidates,camera,boxes,graph['selected_box'],direction)
        report.update(score)
        target = score['target_model']
        if not score['selection_correct']:
            raise RuntimeError("Independent projection did not confirm the requested cube")
    if result != 0:
        raise RuntimeError(graph.get('error') or 'Transport graph did not complete')
    report["lift_m"] = maximum[target] - initial[target][2]
    if report["lift_m"] <= .10:
        raise RuntimeError("No independently observed physical lift")
    x, y, z = final[target]
    table = manifest["tables"][destination]
    cx, cy = table["top_center_xy_m"]
    width, depth, _ = table["top_size_m"]
    if (abs(x-cx) > width/2-.025 or abs(y-cy) > depth/2-.025
            or abs(z-table["top_height_m"]-.025) >= .02):
        raise RuntimeError(f"Target is not supported on table {destination}")
    bystanders = names - {target}
    displacement = {name: math.dist(initial[name], final[name]) for name in bystanders}
    if direction:
        with lock:
            for sample in trace:
                name = sample['name']
                if name in bystanders:
                    displacement[name] = max(displacement[name], math.dist(initial[name], sample['position']))
        report['other_max_displacement_m'] = displacement
    for name, distance in displacement.items():
        if distance > .015:
            raise RuntimeError(f"Non-target {name} moved")
    if graph.get('execution_status')!='pick_place_complete' or not graph.get('placement_verified'):
        raise RuntimeError('Graph did not verify placement')
    report['passed']=True
except Exception as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    if observer.poll() is None:
        os.killpg(observer.pid, signal.SIGTERM)
        try:
            observer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(observer.pid, signal.SIGKILL)
            observer.wait()
    reader.join(timeout=5)
    observer.stdout.close()
    with lock:
        report.setdefault('final_world_poses',{k:list(v) for k,v in latest.items()})
        report.setdefault('maximum_world_z',dict(maxima))
        (args.output/'independent-cube-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if report['passed'] else 1)
