# MoMa Simulator Console

The console runs locally and uses plain HTML, CSS and JavaScript. It displays the live Gazebo GUI and wrist camera, simulator and agent logs, and images captured by the agent during an instruction. Anyone with the console open on the same machine sees the same world and chat.

## Start the live simulator

In the existing MoMa simulator environment, mount this worktree at `/agent` and source ROS and the simulator workspace as usual. Use the environment that already runs `scripts/run_pick_place_task.py`, with the existing Python dependencies and model configuration. From the sourced shell, run:

```sh
python3 /agent/simulator-web/server.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). With a container, use host networking as in the spatial trial runner. The console starts and stops its own Gazebo instance and defaults to ROS domain 73. If domain 73 is in use, set `ROS_DOMAIN_ID` to a free value before launching the console. Use `--port` to choose another browser port.

1. Select a world and click **Load world**. The menu includes the original duplicate-clarification world and the ten spatial trial configurations. Reloading starts a fresh simulation.
2. When the world is ready, type your own instruction or pick a test instruction to fill the chat box. Click **Send** to run it.
3. If several objects match, inspect the image in chat and answer with a camera-relative position such as **leftmost**. You can click the image to enlarge it or reply **cancel** while the agent is asking for clarification.

**Gazebo overview** shows the live Gazebo GUI window, with the robot and both tables in view. **Wrist camera** shows the robot’s view at the same time. Both feeds update during motion.

For the overview, the environment running the website needs an accessible X11 display (including XWayland or a dedicated virtual display), `xwininfo` and `xwd`. In a container, pass `DISPLAY`, mount the X11 socket and mount its authorization file read-only. On the prepared Jazzy host, mount the host's `/usr/bin/xwininfo` and `/usr/bin/xwd` read-only under `/usr/local/bin`. The prepared image also needs the host's `/usr/lib/x86_64-linux-gnu/libxkbfile.so.1` mounted read-only at the same path for `xwd`. Use a dedicated display with one Gazebo GUI window for consistent capture. The server falls back to the wrist camera if capture prerequisites are missing. Use `--wrist-only` to disable the GUI explicitly.

When launching from a worktree with no model configuration, mount the main project’s configured `.env` read-only at `/agent/.env`.

Click **Stop world** during startup, an active task, or clarification to cancel the request and shut down Gazebo. When the site returns to **Select a world**, choose and load a fresh scene. Stopping, reloading, and switching clear the session’s chat and logs.

The chat shows source, selection, and destination images as the agent produces them. Logs stream during startup and execution. In **Activity logs**, choose **Combined**, **Gazebo**, or **Agent** to filter what you see. Run evidence and generated worlds are stored in a temporary directory for the server session.

## Replay without ROS

From the worktree root, run:

```sh
python3 simulator-web/server.py --replay
```

Load the recorded world, send its test instruction, and answer **leftmost**. This mode uses the checked-in camera frames, timing records and placement verification from the duplicate-cube run.
