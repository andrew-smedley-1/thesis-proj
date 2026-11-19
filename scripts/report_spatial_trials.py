"""Write a report and compact per-trial evidence from the completed live suite."""
import argparse
import json
import shutil
from datetime import datetime
from statistics import median
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def pickup_reach_followup(report_dir):
    paired = read(report_dir/'pickup-reach/comparison.json')
    if not paired:
        return []
    results = {variant: sum(row[variant]['verified_transfer'] for row in paired)
               for variant in ('original','offset','centerline')}
    total = len(paired)
    reached = sum(row['centerline']['approach_executed'] for row in paired)
    lines = ['', '## Pickup approach investigation and paired reruns', '',
             'In each of the three failed baseline trials, the requested cube was correctly identified. '
             'RRTConnect could not sample a valid goal state for the downward-facing flange at the hover waypoint, '
             'about 17.3 cm above the grasp. The arm stopped before descending.',
             '',
             'A motion-free MoveIt IK replay reconstructed the approximate parked base pose from each recorded camera '
             'transform, the rigid wrist mount and the commanded overview pose. At the original dock the solver returned '
             'NO_IK_SOLUTION for all three hovers, and again with the hover lowered by 5.3 cm. The hover from '
             'successful control scene spatial_03 solved. This isolates kinematic reach from obstacle checks. '
             'The live transfers below used normal collision checking. [IK measurements](pickup-reach/ik.json)',
             '',
             'Two pickup positions were tested. First, moving the dock from (1.10, 0.00) m to (1.22, −0.14) m '
             'in the map frame gave IK solutions for all three hovers and grasps. In the red topmost and green '
             'leftmost trials, the wrist camera confirmed the cube after lifting, but the independent pose trace '
             'showed that it dropped during the drive to Table B.',
             '',
             'The **centerline pickup dock** at **(1.36, 0.00) m** advances the base 26 cm along the original travel line. '
             'All three hover and grasp endpoints solved in the IK replay at that dock. Nav2 parks before the camera '
             'captures the source image, so the operator chooses from the view used to compute the grasp. '
             'The wrist orientation, grasp height and collision checks are the same in all three conditions.',
             '',
             f'MoveIt executed the approach and proceeded to descent in **{reached}/{total}** centerline trials. '
             f'Verified transfers in the matched scenes were **{results["original"]}/{total} at the original dock**, '
             f'**{results["offset"]}/{total} at the offset dock**, and **{results["centerline"]}/{total} at the centerline dock**.',
             '',
             '| Scene and choice | Original dock | Offset dock | Centerline dock | Peak rise at centerline |',
             '|---|---|---|---|---:|']
    def outcome(attempt):
        if attempt['verified_transfer']:
            return 'Verified transfer'
        if not attempt['approach_executed']:
            return 'Stopped at approach'
        if not attempt['held_after_lift']:
            return 'Reached cube, pickup unverified'
        if attempt['held_after_lift'] and not attempt['held_after_navigation']:
            return 'Lifted, then lost in transit'
        return attempt['error'] or 'Transfer unverified'
    for row in paired:
        name = row['case']
        color = row['target_model'].split('_cube_')[0]
        links = [f"[{outcome(row[v])}](pickup-reach/{v}/{name}/verification.json)" for v in ('offset','centerline')]
        lines.append(f"| {name} ({row['choice']} {color}) | {outcome(row['original'])} | {links[0]} | {links[1]} | "
                     f"{row['centerline']['max_rise_m']*100:.1f} cm |")
    lines += ['', 'Each condition used fresh simulations with matching world checksums, instructions, and scripted replies. '
              'Peak rise is the maximum measured change in cube height over the whole trial. Each of the three previously '
              'failed scenes was repeated once per revised dock. The original ten-scene rate remains **7/10**. '
              'The seven original successes were not rerun with the revised dock. [Paired measurements](pickup-reach/comparison.json)']
    for row in paired:
        attempt = row['centerline']
        if not attempt['approach_executed'] or attempt['held_after_lift']:
            continue
        lines += ['', f"In {row['case']}, the centerline failure occurred after the gripper closed. The "
                  f"{row['choice']} cube rose {attempt['max_rise_m'] * 1000:.1f} mm, below the 10 cm lift criterion. The wrist check stopped "
                  'the task before transport. The pose trace establishes that the grasp failed to lift the cube. '
                  'Jaw contact and grasp alignment would need separate measurements to identify the cause.']
    return lines


def summarize(run, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for folder in sorted(run.glob('spatial_*')):
        trial = read(folder/'trial.json')
        if not trial or not ('infrastructure_error' in trial or 'verifier_exit_code' in trial):
            raise ValueError(f'Incomplete trial at {folder}')
        verdict = read(folder/'verification.json')
        graph = read(folder/'evidence/result.json')
        calls_path = folder/'evidence/calls.jsonl'
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        dialogue = graph.get('selection_dialogue', [])
        actions = read(folder/'camera/manipulation.json') or []
        verified_lift = any(a['event']=='held_verified' for a in actions)
        error = trial.get('infrastructure_error') or graph.get('error') or verdict.get('failure',{}).get('message') or ''
        row = dict(trial=folder.name, instruction=trial['instruction'], replies=trial['replies'],
                   expected_position=trial['expected_position'], world_sha256=verdict.get('world_sha256'),
                   observed_dialogue=dialogue, detected=len(graph.get('boxes', [])), asked=bool(dialogue),
                   full_clarification=bool(verdict.get('clarification_correct')),
                   selected_correct=bool(verdict.get('selection_correct')), held_after_lift=verified_lift,
                   transfer_verified=bool(verdict.get('passed')),
                   expected_instance=verdict.get('target_model'), selected_instance=verdict.get('chosen_model'),
                   graph_status=graph.get('execution_status'), failure=error,
                   model_calls=[c['operation'] for c in calls])
        if verdict.get('finished_at_utc') and verdict.get('started_at_utc'):
            row['verification_elapsed_s'] = (datetime.fromisoformat(verdict['finished_at_utc']) -
                                              datetime.fromisoformat(verdict['started_at_utc'])).total_seconds()
        audit = report_dir/folder.name
        audit.mkdir(exist_ok=True)
        if error.startswith('approach:') and (folder/'simulator.log').is_file():
            matches = [line for line in (folder/'simulator.log').read_text(errors='replace').splitlines()
                       if 'Unable to sample any valid states for goal tree' in line]
            if matches:
                row['diagnosis'] = 'MoveIt could not find a valid arm configuration for the approach pose. The cube was not lifted.'
                (audit/'planner.log').write_text(matches[-1]+'\n')
        for source, name in [(folder/'trial.json','trial.json'),(folder/'verification.json','verification.json'),
                             (folder/'evidence/geometry.json','geometry.json'),(folder/'evidence/source.png','source.png'),
                             (folder/'evidence/annotated.png','selection.png'),(calls_path,'calls.jsonl')]:
            if source.exists():
                shutil.copy2(source,audit/name)
        results.append(row)
    total = len(results)
    if total != 10:
        raise ValueError(f'Expected ten trials; found {total}')
    counts = {key:sum(r[key] for r in results) for key in ('asked','full_clarification','selected_correct','held_after_lift','transfer_verified')}
    repeats = [r for r in results if len(r['replies'])>1]
    recovered = sum(r['full_clarification'] for r in repeats)
    (report_dir/'results.json').write_text(json.dumps({'counts':counts,'trials':results},indent=2)+'\n')
    lines = ['# Spatial clarification in ten live Gazebo transfers','',
             f"The robot resolved the requested position in **{counts['selected_correct']}/{total}** trials. "
             f"Gazebo verified transfer of the requested cube in **{counts['transfer_verified']}/{total}**.",'',
             'Each trial began with three same-color cubes on Table A and an empty Table B. '
             'Ten different layouts covered red, green, and blue cubes and five camera-relative positions. '
             'The scripted operator supplied a reply only after the graph paused. The camera views and prompts used spatial descriptions.', '',
             '| Measure | Result |','|---|---:|',
             f"| Returned three candidate boxes | {sum(r['detected']==3 for r in results)}/{total} |",
             f"| Asked the operator | {counts['asked']}/{total} |",
             f"| Recorded the prescribed dialogue with three candidates | {counts['full_clarification']}/{total} |",
             f"| Asked again after an underspecified reply, then accepted a position | {recovered}/{len(repeats)} |",
             f"| Selected the intended physical instance | {counts['selected_correct']}/{total} |",
             f"| Confirmed a held object using wrist RGB-D after lift | {counts['held_after_lift']}/{total} |",
             f"| Independently verified transfer and stationary bystanders | {counts['transfer_verified']}/{total} |",'',
             '| Trial | Cube | Operator replies, in order | Detections | Correct selection | Verified transfer |',
             '|---|---|---|---:|---|---|']
    for r in results:
        color = next(c for c in ['red','green','blue'] if c in r['instruction'])
        utterances = ' → '.join('“'+reply+'”' for reply in r['replies'])
        yn = lambda value: 'Yes' if value else 'No'
        evidence_file = 'selection.png' if (report_dir/r['trial']/'selection.png').is_file() else 'trial.json'
        lines.append(f"| [{r['trial']}]({r['trial']}/{evidence_file}) | {color} | {utterances} | {r['detected']} | "
                     f"{yn(r['selected_correct'])} | {yn(r['transfer_verified'])} |")
    single_detection_call = sum(r['model_calls'].count('bounding_boxes') == 1 for r in results)
    lines += ['', f'{single_detection_call}/{total} trials used a single bounding-box model call across the instruction and follow-up replies.']
    times = [r['verification_elapsed_s'] for r in results if 'verification_elapsed_s' in r]
    if times:
        lines += ['', f'Median wall time from verifier start to verdict was {median(times)/60:.1f} minutes '
                      f'across {len(times)} trials (range {min(times)/60:.1f}–{max(times)/60:.1f} minutes). '
                      'This includes model calls and simulated robot motion, with up to two trials running concurrently.']
    failed = [r for r in results if not r['transfer_verified']]
    if failed:
        lines += ['', '## Trials without a verified transfer', '']
        for r in failed:
            lines.append(f"- **{r['trial']}** ({r['expected_position']}). {r.get('diagnosis') or r['failure']}")
    lines += ['','## Scoring','',
              'The verifier projects independently measured Gazebo cube centers into the source camera frame. '
              'It derives the requested position from those projections and pairs all detections with the projected centers, '
              'allowing a common translation for odometry drift. Selection requires a unique geometric match to the expected cube. '
              'Physical success additionally requires a lift over 10 cm, a verified release on Table B, and at most 1.5 cm of movement by each bystander.', '',
              'Each case was measured once. These results describe the ten scripted scenes in simulation. '
              'The clarification and transfer rates both use all ten trials as their denominator.']
    followup = pickup_reach_followup(report_dir)
    if followup:
        lines[4:4] = ['[MoveIt diagnosis and paired reruns of the three failed scenes](#pickup-approach-investigation-and-paired-reruns)', '']
        lines += followup
    (report_dir/'report.md').write_text('\n'.join(lines)+'\n')
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('report', type=Path)
    args=parser.parse_args()
    print(summarize(args.run,args.report))
