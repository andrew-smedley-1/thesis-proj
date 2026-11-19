"""Run the actual transport graph using calibrated two-table station configuration."""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic import build_robot_brain, run_with_selection
from run_evidence import RunEvidence
from pick_place import PickPlaceController, load_stations
from ros_bridge import ROSBridge, BridgeConfig, load_locations

async def run(args):
    with ROSBridge(BridgeConfig(goal_timeout=180),enable_motion=True,
                   locations=load_locations('/agent/config/two_table_locations.json')) as bridge:
        manipulator=PickPlaceController(bridge,load_stations('/agent/config/two_table_stations.json'),args.output/'camera')
        evidence=RunEvidence(args.output/"evidence")
        graph=build_robot_brain(bridge,evidence=evidence,observe_only=args.observe_only,spatial_choices_only=args.spatial_choices_only,manipulator=manipulator,capture_timeout=15)
        result=await run_with_selection(graph, {'task_input':args.instruction},
                                        {'configurable':{'thread_id':uuid4().hex}}, args.reply)
        evidence.finish(result)
        summary={k:v for k,v in result.items() if k!='image_url'}
        (args.output/'graph-result.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
        print(json.dumps(summary,default=str),flush=True)
        expected = ('inspection_complete', 'object_absent', 'needs_clarification') if args.observe_only else ('pick_place_complete',)
        return 0 if result.get('execution_status') in expected else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reply', action='append', default=[], help='Answer a duplicate-object question if asked')
    parser.add_argument('--spatial-choices-only', action='store_true', help='Use position descriptions and unnumbered camera boxes')
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--instruction', default='Move the red cube from table A to table B.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.WARNING);logging.getLogger('llm_timing').setLevel(logging.INFO)
    raise SystemExit(asyncio.run(run(args)))
