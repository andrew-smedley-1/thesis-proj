# Duplicate-cube clarification trial

Start the two-table simulator with the checkout mounted at `/agent` and a writable output directory mounted at `/trial`. The scene has three 50 mm red cubes on Table A and an empty Table B.

```sh
ros2 launch /agent/scripts/two_table_transport.launch.py world:=/agent/tests/scenes/duplicate_clarification_01.sdf headless:=false
```

In a second terminal in the same ROS environment, run the agent interactively. When it shows the camera image path and asks which cube to pick, type `left most` or `pick up the left most red cube`.

```sh
python3 /agent/scripts/run_pick_place_task.py --spatial-choices-only --output /trial/interactive --instruction 'Pick up the red cube from Table A and set it down on Table B.'
```

For an independently scored trial, start with a fresh instance of the scene and use a fresh output directory. The verifier reads instruction 7 from `tests/instructions/complete.json`, passes the test reply to the agent at clarification time, and checks the selected cube and the two bystanders through Gazebo poses.

```sh
python3 /agent/scripts/verify_pick_place_live.py --world duplicate_clarification_01 --scene-manifest /agent/tests/scenes/duplicate_clarification_01.placements.json --instruction-row 7 --output /trial/scored --expect-leftmost --reply 'left most'
```

The offline graph tests use the generated replies in `tests/instructions/duplicate_red_cube_clarifications.json`. Run them with `python -m unittest discover -s tests -p test_duplicate_clarification.py`.

The ten-trial suite in `tests/instructions/spatial_duplicate_trials.json` covers leftmost, rightmost, middle, topmost, and bottommost. All directions refer to the source camera image. The two multi-turn cases first repeat the object description, then supply a position. Each trial starts a fresh Gazebo world with three cubes and records a separate pose trace.

Use `python scripts/run_spatial_trials.py --help` for the local simulator engine, environment and output options. Score the resulting directory with `python scripts/report_spatial_trials.py RUN_DIRECTORY REPORT_DIRECTORY`.

### Reproducing the Table A pickup comparison

The station configuration includes an optional `pickup_pose` in the map frame. For a transport from Table A, Nav2 uses that pose before the source camera capture. The named Table A location still handles ordinary navigation and destination visits.

Run the suite with `--cases 4 6 9` and a fresh output directory for each pickup pose. The comparison used the named Table A pose (1.10, 0.00) m, an offset pose (1.22, −0.14) m, and the current pose (1.36, 0.00) m. Each used map yaw zero.

For the no-motion IK diagnostic in a sourced ROS simulator, run `python3 scripts/probe_pickup_reach.py --shift .26 0` followed by the original trial directories for scenes 04, 06, 09 and successful control 03. Use `--shift .12 -.14` for the offset pose. The probe uses the source camera metadata, manipulation log and recorded grasp plan from each directory.
