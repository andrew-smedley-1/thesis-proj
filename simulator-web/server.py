"""Local web console for the duplicate-object Gazebo workflow."""
import argparse
import asyncio
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
from threading import Event, RLock, Thread
from tempfile import TemporaryDirectory
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

sys.dont_write_bytecode = True
WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
SCENE = "duplicate_clarification_01"
RECORDING = ROOT / "artifacts" / "duplicate-clarification"
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
RESULT_MESSAGES = {
    "pick_place_complete": "Transfer complete. The destination camera confirmed placement.",
    "inspection_complete": (
        "Inspection complete. See the camera response. To pick up a cube, send a transfer "
        "instruction with the source and destination."
    ),
    "object_absent": "I couldn't see the requested object.",
    "nav_success": "I reached the requested location.",
    "target_selection_cancelled": "Selection cancelled.",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stop_process(process, timeout):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class Console:
    def __init__(self, directory, replay, overview=False):
        self.directory = directory
        self.replay = replay
        self.overview = overview
        fixtures = ROOT / "tests" / "instructions"
        complete = read_json(fixtures / "complete.json")
        incomplete = read_json(fixtures / "incomplete.json")
        self.sample = read_json(fixtures / "duplicate_red_cube_clarifications.json")["task_input"]
        self.instructions = [{"group": "Duplicate clarification", "text": self.sample}]
        self.worlds = {SCENE: {"label": "Duplicate clarification · 3 red cubes", "sample": self.sample}}
        if not replay:
            for index, trial in enumerate(read_json(fixtures / "spatial_duplicate_trials.json"), 1):
                row = complete[trial["instruction_row"] - 1]
                target = row["expected"]["target_object"]
                self.worlds[f"spatial_{index:02d}"] = {
                    "label": f"Spatial {index:02d} · 3 × {target}", "sample": row["instruction"],
                    "target": target, "positions": trial["table_A_xy_m"],
                }
            for group, rows in (("Complete", complete), ("Incomplete", incomplete)):
                self.instructions.extend({"group": group, "text": row["instruction"]} for row in rows)
        self.lock = RLock()
        self.replies = Queue()
        self.tasks = Queue()
        self.task_done = Event()
        self.task_done.set()
        self.active_task = None
        self.stop_worker = None
        self.session = uuid4().hex
        self.events = deque(maxlen=600)
        self.images = {}
        self.sequence = 0
        self.world = None
        self.recorded_complete = False
        self.closing = False
        self.phase = "select"
        self.process = self.bridge = self.worker = None
        self.gui = self.gui_view = None
        self.task_worker = Thread(target=self.work_tasks, daemon=True)
        self.task_worker.start()

    def emit(self, kind, text, **fields):
        with self.lock:
            self.sequence += 1
            self.events.append({"id": self.sequence, "kind": kind, "text": text, "time": time.time(), **fields})

    def image(self, label, path):
        if not path.is_file():
            return
        data = path.read_bytes()
        with self.lock:
            key = uuid4().hex
            self.images[key] = data
            if len(self.images) > 60:
                del self.images[next(iter(self.images))]
            self.emit("image", label, url=f"/api/image/{key}")

    def state(self, after, session=None):
        with self.lock:
            if session is not None and session != self.session:
                after = 0
            return {
                "world": self.world, "phase": self.phase, "session": self.session,
                "events": [event for event in self.events if event["id"] > after],
            }

    def reset_session(self):
        self.session = uuid4().hex
        self.events.clear()
        self.images.clear()
        self.replies = Queue()

    def start_world(self, world):
        with self.lock:
            if world not in self.worlds:
                raise ValueError("Select a world from the list.")
            if self.closing or self.phase in {"loading", "running", "reply", "stopping"}:
                raise RuntimeError("Stop the current world or wait for the current operation to finish.")
            self.reset_session()
            self.world = world
            self.phase = "loading"
            self.emit("log", f"Loading {self.worlds[world]['label']}.")
            self.worker = Thread(target=self.load_world, args=(world,), daemon=True)
            self.worker.start()

    def load_world(self, world):
        if self.replay:
            with self.lock:
                if self.closing or self.phase != "loading":
                    return
                self.recorded_complete = False
                self.phase = "ready"
                self.emit("log", "Recorded world loaded.")
            return
        try:
            from ros_bridge import BridgeConfig, ROSBridge, load_locations

            self.stop_world()
            if world == SCENE:
                scene = ROOT / "tests" / "scenes" / f"{world}.sdf"
            else:
                from scripts.generate_world_scenes import generate_world

                spec = self.worlds[world]
                scene = self.directory / f"{world}.sdf"
                generate_world({"id": world, "table_A": [spec["target"]] * 3, "table_B": []},
                               scene, table_xy={"A": spec["positions"]})
            with self.lock:
                if self.closing or self.phase != "loading":
                    return
                process = self.process = subprocess.Popen(
                    ["ros2", "launch", str(ROOT / "scripts" / "two_table_transport.launch.py"),
                     f"world:={scene}", "headless:=true"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                    start_new_session=True,
                )
                Thread(target=self.read_simulator, args=(process,), daemon=True).start()
                bridge = self.bridge = ROSBridge(
                    BridgeConfig(goal_timeout=180), enable_motion=True,
                    locations=load_locations(ROOT / "config" / "two_table_locations.json"),
                )
            deadline = time.monotonic() + 240
            while process.poll() is None and time.monotonic() < deadline:
                with self.lock:
                    if self.closing or self.phase != "loading":
                        return
                ready = bridge.readiness()
                if ready["ready"] and ready["camera"]["ready"]:
                    with self.lock:
                        if self.closing or self.phase != "loading" or self.process is not process:
                            return
                        self.phase = "ready"
                        self.emit("log", "Gazebo, camera, Nav2 and MoveIt are ready.")
                        if self.overview:
                            self.start_gui()
                    return
                time.sleep(1)
            raise RuntimeError("Gazebo exited or the ROS camera and action servers did not become ready.")
        except Exception as exc:
            self.stop_world()
            with self.lock:
                if self.phase == "loading":
                    self.phase = "failed"
                    self.emit("log", f"World startup failed: {exc}")

    def read_simulator(self, process):
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    if self.process is process and line.strip():
                        self.emit("simulator", line.strip()[:2000])
        code = process.wait()
        with self.lock:
            if self.process is process and self.phase in {"ready", "running", "reply"}:
                self.emit("log", f"Gazebo exited with code {code}.")
                if self.phase == "reply":
                    self.replies.put("cancel")
                if self.phase == "ready":
                    self.phase = "failed"

    def start_gui(self):
        from gazebo_view import GazeboView

        try:
            config = ROOT / "config" / "two_table.gui.config"
            self.gui = subprocess.Popen(
                ["gz", "sim", "-g", "--gui-config", str(config)],
                env={**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1", "QT_XCB_GL_INTEGRATION": "xcb_glx"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True,
            )
            self.gui_view = GazeboView()
            Thread(target=self.read_gui, args=(self.gui,), daemon=True).start()
        except OSError as exc:
            self.emit("log", f"Could not open the Gazebo overview: {exc}")

    def read_gui(self, process):
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    if self.gui is process and line.strip():
                        self.emit("simulator", f"[Gazebo GUI] {line.strip()[:1950]}")
        code = process.wait()
        with self.lock:
            if self.gui is process:
                self.emit("log", f"The Gazebo GUI exited with code {code}.")

    def stop_world(self):
        with self.lock:
            process, bridge, gui = self.process, self.bridge, self.gui
            self.process = self.bridge = self.gui = self.gui_view = None
        try:
            if bridge is not None:
                bridge.close()
        finally:
            try:
                stop_process(gui, timeout=5)
            finally:
                stop_process(process, timeout=20)

    def cancel_task(self):
        if self.phase == "reply":
            self.replies.put(None)
        if self.active_task is not None:
            self.active_task.get_loop().call_soon_threadsafe(self.active_task.cancel)

    def request_stop(self):
        with self.lock:
            if self.closing or self.world is None or self.phase == "stopping":
                return
            self.cancel_task()
            self.phase = "stopping"
            self.stop_worker = Thread(target=self.finish_stop, args=(self.worker,), daemon=True)
            self.stop_worker.start()

    def finish_stop(self, loader):
        failure = None
        try:
            if loader is not None:
                loader.join()
            self.stop_world()
        except Exception as exc:
            failure = exc
            logging.exception("Could not stop the simulator world")
        finally:
            self.task_done.wait()
            with self.lock:
                if failure is None:
                    self.reset_session()
                    self.world = None
                    self.phase = "select"
                else:
                    self.phase = "failed"
                    self.emit("log", f"Could not stop the world: {failure}")
                self.stop_worker = None

    def close(self):
        with self.lock:
            self.closing = True
            self.cancel_task()
            self.tasks.put(None)
            stopping = self.stop_worker
        if stopping is not None:
            stopping.join()
        else:
            self.stop_world()
        self.task_worker.join(timeout=3)

    def overview_frame(self):
        with self.lock:
            view = self.gui_view
            if self.phase in {"loading", "failed", "stopping"} or self.gui is None or self.gui.poll() is not None:
                return None
        data = view.capture()
        return (data, "image/jpeg") if data else None

    def frame(self):
        with self.lock:
            if self.world is None or self.phase in {"loading", "failed", "stopping"}:
                return None
            bridge = self.bridge
        if self.replay:
            filename = "destination.png" if self.phase == "ready" and self.recorded_complete else "source.png"
            return (RECORDING / filename).read_bytes(), "image/png"
        if bridge is None:
            return None
        from PIL import Image
        from ros_bridge import CameraUnavailableError

        try:
            bgr, _, _ = bridge.camera_snapshot(max_age=2, max_skew=0.2).to_arrays()
        except (CameraUnavailableError, RuntimeError, ValueError):
            return None
        with io.BytesIO() as buffer:
            Image.fromarray(bgr[:, :, ::-1]).save(buffer, format="JPEG", quality=75)
            return buffer.getvalue(), "image/jpeg"

    def send(self, message):
        with self.lock:
            if self.closing or self.phase not in {"ready", "reply"}:
                raise RuntimeError("Load a world and wait for it to become ready.")
            reply = self.phase == "reply"
            self.phase = "running"
            self.emit("user", message)
            if reply:
                self.replies.put(message)
            else:
                self.task_done.clear()
                self.tasks.put(message)

    def ask(self, question):
        with self.lock:
            if self.closing or self.phase == "stopping":
                raise asyncio.CancelledError
            if not self.replay and (self.process is None or self.process.poll() is not None):
                raise RuntimeError("The selected world is no longer running.")
            self.phase = "reply"
            self.emit("agent", question)
            replies = self.replies
        reply = replies.get()
        if reply is None:
            raise asyncio.CancelledError
        return reply

    def work_tasks(self):
        # The model clients reuse asynchronous connections across instructions.
        with asyncio.Runner() as runner:
            while (message := self.tasks.get()) is not None:
                self.run_task(message, runner)

    def run_task(self, message, runner):
        try:
            with self.lock:
                if self.closing or self.phase == "stopping":
                    return
            if self.replay:
                self.run_recording(message)
            else:
                runner.run(self.run_live_task(message))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.emit("agent", f"The request failed ({type(exc).__name__}).")
            logging.exception("Simulator web request failed")
        finally:
            with self.lock:
                if not self.closing and self.phase != "stopping":
                    self.phase = ("ready" if self.replay or
                                  (self.process and self.process.poll() is None) else "failed")
                self.task_done.set()

    async def run_live_task(self, message):
        with self.lock:
            if self.closing or self.phase == "stopping":
                return
            self.active_task = asyncio.current_task()
        try:
            await self.run_graph(message)
        finally:
            with self.lock:
                self.active_task = None

    def run_recording(self, message):
        if message.casefold() != self.sample.casefold():
            self.emit("agent", "Choose the recorded instruction from the test cases to replay this run.")
            return
        self.recorded_complete = False
        recorded = read_json(RECORDING / "verification.json")
        self.image("Source camera", RECORDING / "source.png")
        for line in (RECORDING / "calls.jsonl").read_text().splitlines():
            call = json.loads(line)
            self.emit("log", f"{call['operation']} · {call['status']} · {call['duration_ms']:.0f} ms")
        self.image("Object selection", RECORDING / "selection.png")
        reply = self.ask(recorded["dialogue"][0]["question"])
        while " ".join(reply.lower().split()) not in {
            "leftmost", "left most", "left", "1", "box 1", recorded["dialogue"][0]["reply"],
        }:
            if reply.lower() in {"cancel", "stop"}:
                self.emit("agent", "Replay cancelled.")
                return
            reply = self.ask("This recording follows the leftmost cube. Enter leftmost to continue.")
        self.recorded_complete = True
        self.image("Destination verification", RECORDING / "destination.png")
        self.emit("log", "Recorded placement verification passed.")
        self.emit("agent", RESULT_MESSAGES["pick_place_complete"])

    async def run_graph(self, message):
        from langgraph.types import Command
        from agentic import build_robot_brain
        from pick_place import PickPlaceController, load_stations
        from run_evidence import RunEvidence

        console = self
        session = self.session

        class LoggedController(PickPlaceController):
            def event(self, name, **values):
                super().event(name, **values)
                with console.lock:
                    if console.session == session:
                        console.emit("log", json.dumps({"event": name, **values}, default=str))

        class LoggedEvidence(RunEvidence):
            def call(self, record):
                super().call(record)
                console.emit("log", f"{record['operation']} · {record['status']} · {record['duration_ms']:.0f} ms")

        while True:
            task_id = uuid4().hex
            directory = self.directory / task_id
            evidence = LoggedEvidence(directory / "evidence")
            camera = directory / "camera"
            controller = LoggedController(self.bridge, load_stations(ROOT / "config" / "two_table_stations.json"), camera)
            graph = build_robot_brain(self.bridge, evidence=evidence, spatial_choices_only=True,
                                      manipulator=controller, capture_timeout=15)
            config = {"configurable": {"thread_id": task_id}, "recursion_limit": 100}
            inputs = {"task_input": message, "chat_history": []}
            self.emit("log", "Agent started.")
            while True:
                prompt = None
                async for update in graph.astream(inputs, config, stream_mode="updates"):
                    for node, value in update.items():
                        if node == "__interrupt__":
                            prompt = value[0].value["question"]
                        elif isinstance(value, dict):
                            self.graph_update(node, value, evidence.directory, camera)
                if prompt is None:
                    break
                inputs = Command(resume=await asyncio.to_thread(self.ask, prompt))
            result = (await graph.aget_state(config)).values
            evidence.finish(result)
            if result.get("missing_fields") and result.get("chat_history"):
                details = await asyncio.to_thread(self.ask, result["chat_history"][-1].content)
                if details.casefold() in {"cancel", "stop"}:
                    self.emit("agent", "Request cancelled.")
                    return
                message = f"{message}\nClarification: {details}"[:8000]
                continue
            status = result.get("execution_status", "unknown")
            self.emit("agent", result.get("error") or RESULT_MESSAGES.get(
                status, f"I finished processing the request. Status: {status.replace('_', ' ')}."))
            return

    def graph_update(self, node, update, evidence, camera):
        status = update.get("execution_status")
        self.emit("log", node.replace("_", " ") + (f" · {status.replace('_', ' ')}" if status else ""))
        if node == "capture_image":
            self.image("Source camera", evidence / "source.png")
        if node == "bounding_boxes":
            self.image("Object selection", evidence / "annotated.png")
        if node == "plan_grasp" and status == "grasp_planned":
            self.image("Selected object and grasp plan", evidence / "annotated.png")
        if node == "verify_placement":
            self.image("Destination verification", camera / "destination-after.png")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, status, data=b"", content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if data:
            self.wfile.write(data)

    def json(self, status, data):
        self.respond(status, json.dumps(data).encode())

    def local_request(self):
        port = self.server.server_port
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (origin and origin != f"http://{host}"):
            self.json(403, {"error": "Open the console on localhost."})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        route = urlsplit(self.path)
        path = route.path
        console = self.server.console
        if path in FILES:
            name, content_type = FILES[path]
            self.respond(200, (WEB / name).read_bytes(), content_type)
        elif path == "/api/options":
            self.json(200, {"worlds": [{"id": key, "label": spec["label"], "sample": spec["sample"]}
                                      for key, spec in console.worlds.items()],
                            "instructions": console.instructions, "replay": console.replay,
                            "overview": console.overview})
        elif path == "/api/state":
            params = parse_qs(route.query)
            try:
                after = int(params.get("after", ["0"])[0])
            except ValueError:
                self.json(400, {"error": "Invalid event cursor."})
                return
            self.json(200, console.state(after, params.get("session", [None])[0]))
        elif path in {"/api/frame", "/api/overview"}:
            frame = console.overview_frame() if path == "/api/overview" else console.frame()
            if frame:
                data, content_type = frame
                self.respond(200, data, content_type)
            else:
                self.respond(204)
        elif path.startswith("/api/image/"):
            with console.lock:
                data = console.images.get(path.rsplit("/", 1)[-1])
            self.respond(200, data, "image/png") if data else self.respond(404)
        else:
            self.respond(404)

    def do_POST(self):
        if not self.local_request():
            return
        if self.path not in {"/api/world", "/api/world/stop", "/api/message"}:
            self.respond(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 12000 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON request.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/world/stop":
                self.server.console.request_stop()
            else:
                key = "world" if self.path == "/api/world" else "message"
                value = body.get(key)
                if not isinstance(value, str) or not value.strip() or len(value) > 2000:
                    raise ValueError(f"Enter a valid {key}.")
                if key == "world":
                    self.server.console.start_world(value)
                else:
                    self.server.console.send(value.strip())
        except (ValueError, UnicodeError) as exc:
            self.json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.json(409, {"error": str(exc)})
        else:
            self.respond(204)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--replay", action="store_true", help="Play the recorded duplicate-cube run without ROS")
    parser.add_argument("--wrist-only", action="store_true", help="Disable the Gazebo GUI overview")
    args = parser.parse_args()
    overview = False
    if not args.replay:
        if ROOT != Path("/agent") and not Path("/agent/scripts/two_table_map.py").exists():
            parser.error("Run live mode in the simulator environment with this worktree mounted at /agent.")
        os.environ.setdefault("ROS_DOMAIN_ID", "73")
        os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        os.environ["GZ_PARTITION"] = f"moma-web-{uuid4().hex}"
        sys.path.insert(0, str(ROOT))
        import rclpy  # Require a sourced ROS environment before accepting connections.
        import agentic  # Load the existing model configuration before launching ROS.
        if not args.wrist_only:
            from gazebo_view import capture_available

            overview = capture_available()
            if not overview:
                logging.warning("Gazebo overview needs DISPLAY, xwininfo and xwd.")
    logging.basicConfig(level=logging.WARNING)
    with TemporaryDirectory(prefix="moma-web-") as directory:
        console = Console(Path(directory), args.replay, overview)
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.console = console
            print(f"MoMa Simulator Console: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                console.close()


if __name__ == "__main__":
    main()
