"""Run position-only clarification tests in fresh isolated Gazebo/ROS simulations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4
from generate_world_scenes import generate_world

ROOT = Path(__file__).resolve().parents[1]
TRIALS = json.loads((ROOT / 'tests/instructions/spatial_duplicate_trials.json').read_text())
INSTRUCTIONS = json.loads((ROOT / 'tests/instructions/complete.json').read_text())
ENGINE_LOCK = threading.Lock()


def run_trial(args, index):
    spec = TRIALS[index-1]
    instruction = INSTRUCTIONS[spec['instruction_row']-1]
    world = f'spatial_{index:02d}'
    out = args.output.resolve() / world
    out.mkdir(parents=True, exist_ok=False)
    target = instruction['expected']['target_object']
    generate_world({'id': world, 'table_A':[target]*3, 'table_B':[]}, out/'scene.sdf',
                   table_xy={'A':spec['table_A_xy_m']})
    name = 'moma-spatial-'+uuid4().hex[:10]
    entry = [str(args.helper), name, '/ws/scripts/entrypoint.sh']
    info = {**spec, 'world':world, 'instruction':instruction['instruction'], 'container':name}
    def save():
        (out/'trial.json').write_text(json.dumps(info,indent=2)+'\n')
    save()
    launched = False
    try:
        command = [str(args.engine),'run','--detach','--name',name,'--network=host','--shm-size=1g',
                   '--stop-signal','SIGINT','--stop-timeout','30','-e','DISPLAY','-e','XAUTHORITY=/tmp/sim.xauth',
                   '-e','QT_X11_NO_MITSHM=1','-e','ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST',
                   '-e',f'ROS_DOMAIN_ID={args.domain_base+index}','-e',f'GZ_PARTITION={name}',
                   '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
                   '-v',f'{ROOT}:/agent:ro','-v',f'{out}:/trial',
                   '-v',f'{args.python_packages.resolve()}:/opt/moma-python:ro',
                   '-v',f'{args.env_file.resolve()}:/agent/.env:ro','-v','/tmp/.X11-unix:/tmp/.X11-unix:ro',
                   '-v',f"{os.environ['XAUTHORITY']}:/tmp/sim.xauth:ro",args.image,
                   'ros2','launch','/agent/scripts/two_table_transport.launch.py','world:=/trial/scene.sdf','headless:=true']
        with ENGINE_LOCK:
            subprocess.run(command,check=True,capture_output=True,text=True)
        launched = True
        deadline = time.monotonic()+240
        while time.monotonic()<deadline:
            ready = subprocess.run(entry+['python3','/agent/ros_bridge.py','status'],text=True,
                                   capture_output=True,timeout=35)
            if ready.returncode == 0:
                status = json.loads(ready.stdout.strip().splitlines()[-1])
                if status['ready'] and status['camera']['ready']:
                    break
            time.sleep(2)
        else:
            raise RuntimeError('ROS action servers and camera did not become ready')
        args_verifier = ['python3','/agent/scripts/verify_pick_place_live.py','--world',world,
                         '--scene-manifest','/trial/scene.placements.json','--output','/trial',
                         '--instruction-row',str(spec['instruction_row']),'--expect-position',spec['expected_position']]
        for reply in spec['replies']:
            args_verifier.extend(['--reply',reply])
        with (out/'verifier.log').open('w') as log:
            proc = subprocess.run(entry+['bash','-c','export PYTHONPATH="/opt/moma-python:/agent:$PYTHONPATH"; exec "$@"',
                                         'trial',*args_verifier],stdout=log,stderr=subprocess.STDOUT,timeout=1320)
        info['verifier_exit_code'] = proc.returncode
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        info['infrastructure_error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            info['infrastructure_detail'] = exc.stderr
    finally:
        if launched:
            with (out/'simulator.log').open('w') as log:
                subprocess.run([str(args.engine),'logs',name],stdout=log,stderr=subprocess.STDOUT)
            with ENGINE_LOCK:
                stopped = subprocess.run([str(args.engine),'stop',name],capture_output=True)
                info['simulator_stopped'] = stopped.returncode == 0
                if stopped.returncode == 0:
                    subprocess.run([str(args.engine),'rm',name],check=True,capture_output=True)
        save()
    print(world, 'verifier exit', info.get('verifier_exit_code'), info.get('infrastructure_error',''), flush=True)
    return 'infrastructure_error' not in info and info.get('simulator_stopped', False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('engine','helper','env-file','python-packages','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--image',default='localhost/moma-sim:scenes-dev')
    parser.add_argument('--domain-base',type=int,default=60)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--cases',type=int,nargs='+',choices=range(1,len(TRIALS)+1),default=list(range(1,len(TRIALS)+1)))
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(lambda i: run_trial(args,i), args.cases))
    raise SystemExit(0 if all(completed) else 1)
