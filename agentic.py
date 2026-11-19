"""LangGraph navigation and live-camera perception for the MoMa simulator."""
import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import time
import sys
from typing import List, Literal, TypedDict
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from target_selection import numbered_boxes, resolve_choice
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, model_validator

from llm_timing import track_llm_call
from run_evidence import RunEvidence
from robot_tools import build_robot_tools
from ros_bridge import CameraUnavailableError, ROSBridge, UnknownLocationError, load_locations

load_dotenv(Path(__file__).with_name(".env"))


class RobotState(TypedDict, total=False):
    task_input: str
    task_type: Literal["navigate", "inspect", "transport"]
    target_object: str | None
    start_location: str | None
    end_location: str | None
    missing_fields: List[str]
    chat_history: List[BaseMessage]
    item_presence: str
    current_phase: str
    execution_status: str
    selected_box: List[int]
    target_selector: str | None
    selection_dialogue: list[dict]
    boxes: List[List[int]]
    image_url: str
    image: str
    image_width: int
    image_height: int
    camera_after: float
    camera_metadata: dict
    navigation_result: dict
    error: str
    grasp_plan: dict
    place_surface: list
    grasp_verified: bool
    placement_verified: bool
    transport_result: dict


class TaskAnalysis(BaseModel):
    task_type: Literal["navigate", "inspect", "transport"] = Field(
        default="transport", description="navigate: drive only; inspect: look for an object; transport: move an object.")
    target_selector: str | None = Field(default=None, description="Explicit image-relative choice such as leftmost, middle or box 2; otherwise null.")
    target_object: str | None = Field(description="The specific item required; null for navigation only.")
    start_location: str | None = Field(description="Object source or inspection location; null for the current camera view.")
    end_location: str | None = Field(description="Destination for navigation or transport; null for inspection only.")


class BoundingBox(BaseModel):
    box_2d: List[int] = Field(min_length=4, max_length=4,
        description="[ymin, xmin, ymax, xmax] normalized to 0-1000.")

    @model_validator(mode="after")
    def valid_box(self):
        y1, x1, y2, x2 = self.box_2d
        if not (0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000):
            raise ValueError("Bounding box must have ordered coordinates within 0-1000")
        return self


class BoundingBoxes(BaseModel):
    boxes: List[BoundingBox]


@dataclass
class Models:
    analysis: object
    presence: object
    boxes: object
    analysis_name: str = "injected"
    presence_name: str = "injected"
    boxes_name: str = "injected"


def create_models():
    """Construct the configured real clients only when a graph invocation needs them."""
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai.decisions import OpenAIDecisions

    analysis_name = os.getenv("AGENT_MODEL", "gpt-6-luna")
    presence_name = os.getenv("PRESENCE_MODEL", analysis_name)
    boxes_name = os.getenv("BOUNDING_BOX_MODEL", "gemini-3.8-flash")
    return Models(
        ChatOpenAI(model=analysis_name, timeout=45, max_retries=1).with_structured_output(TaskAnalysis),
        OpenAIDecisions(model=presence_name, timeout=45, max_retries=1),
        ChatGoogleGenerativeAI(model=boxes_name, temperature=0, timeout=45, max_retries=1)
            .with_structured_output(BoundingBoxes),
        analysis_name, presence_name, boxes_name,
    )


def build_robot_brain(bridge=None, *, models=None, capture_timeout=5.0, checkpointer=None, manipulator=None, evidence=None, observe_only=False, spatial_choices_only=False):
    """Build a graph bound to one bridge. With no bridge, require an explicit saved image.

    The application owns the bridge lifecycle. No bridge or API client is created at
    import time. Camera pixels sent to the configured vision models are RGB PNGs.
    """
    if not np.isfinite(capture_timeout) or capture_timeout <= 0:
        raise ValueError("capture_timeout must be positive and finite")
    clients = models
    tools = {tool.name: tool for tool in build_robot_tools(bridge)} if bridge is not None else {}

    def get_models():
        nonlocal clients
        if clients is None:
            clients = create_models()
        return clients

    def model_error(stage, exc):
        # Avoid copying credential-bearing provider URLs into graph state or reports.
        return {"execution_status": "model_error", "current_phase": "failed",
                "error": f"{stage} failed ({type(exc).__name__})"}

    async def decompose(state):
        reset = {"boxes": [], "selected_box": [], "selection_dialogue": [], "item_presence": "", "missing_fields": [], "error": "",
                 "navigation_result": {}, "camera_metadata": {}, "grasp_plan": {},
                 "grasp_verified": False, "placement_verified": False, "transport_result": {}, "place_surface": []}
        if bridge is not None:
            reset.update(image="", image_url="", image_width=0, image_height=0)
        try:
            m = get_models()
            with track_llm_call("task_decomposition", m.analysis_name, evidence) as call:
                analysis = await m.analysis.ainvoke(
                    "Extract the task without inventing information. A pure drive/go/navigate request "
                    "has task_type=navigate and its destination in end_location, target_object and "
                    "start_location null. Looking for/inspecting an object has task_type=inspect, "
                    "target_object and optional start_location; end_location null. Use transport for "
                    "pick up/move/bring an object, requiring target_object, start_location and end_location. "
                    "Keep the color and shape in target_object. Extract an explicitly requested image-relative "
                    "choice in target_selector. Leave it null if the instruction does not specify which instance. "
                    "Relative user locations such as 'here', 'there' or 'over there' are unresolved: "
                    "return null for them. An explicit request to inspect the current camera view may "
                    "use start_location=null. Preserve location names as stated; do not invent coordinates. "
                    f"Instruction: {state['task_input']}"
                )
                analysis = TaskAnalysis.model_validate(analysis)
                call["result"] = analysis.model_dump()
        except Exception as exc:
            return {**reset, **model_error("Task decomposition", exc)}
        required = (["end_location"] if analysis.task_type == "navigate" else
                    ["target_object", "start_location", "end_location"] if analysis.task_type == "transport"
                    else ["target_object"])
        missing = [name for name in required if not getattr(analysis, name)]
        return {**reset, **analysis.model_dump(), "missing_fields": missing,
                "execution_status": "needs_clarification" if missing else "analysed",
                "current_phase": "missing_object" if missing else "go_to_start"}

    def clarify(state):
        message = f"Please clarify these missing details: {', '.join(state['missing_fields'])}."
        return {"chat_history": state.get("chat_history", []) + [AIMessage(content=message)]}

    def validate_locations(state):
        if bridge is None:
            if state["task_type"] == "navigate":
                return {"execution_status": "bridge_unavailable", "current_phase": "failed",
                        "error": "Navigation requires a ROSBridge instance"}
            return {"execution_status": "locations_validated"}
        # Validate every named location before driving, including a transport destination.
        try:
            for name in (state.get("start_location"), state.get("end_location")):
                if name:
                    bridge.lookup_location(name)
        except (UnknownLocationError, ValueError) as exc:
            return {"execution_status": "unknown_location", "current_phase": "failed", "error": str(exc)}
        return {"execution_status": "locations_validated"}

    async def go_to(name, pose=None):
        if pose is None:
            pose = bridge.lookup_location(name)
        arrived = await tools["navigate_robot"].ainvoke(asdict(pose))
        if not arrived:
            raise RuntimeError(f"Nav2 did not confirm arrival at {name!r}")
        return {"success": True, "location": name}

    async def move_arm(xyz, label, *, frame="odom", linear=False, shorter_lift=None):
        x, y, z = xyz
        manipulator.event(label, target=list(xyz), frame=frame, linear=linear)
        result = await tools["move_robot_arm"].ainvoke(
            dict(x=x, y=y, z=z, qx=1., qy=0., qz=0., qw=0., frame_id=frame, linear=linear))
        if (not result["success"] and shorter_lift is not None
                and result.get("details", {}).get("error_code") == -31):
            # MoveIt rejected the endpoint before executing. Keep the retreat vertical.
            manipulator.event("lift_endpoint_unreachable", moveit_result=result)
            return await move_arm(shorter_lift, "shorter_vertical_lift", frame=frame, linear=True)
        if not result["success"]:
            raise RuntimeError(f"{label}: {result}")
        await asyncio.sleep(.4)  # Allow the arm and wrist camera to settle.

    async def stow():
        await move_arm([.4, -.4, 1.35], "carry_pose", frame="base_link")

    async def command_gripper(position, *, contact=False):
        result = await tools["command_robot_gripper"].ainvoke({"position": position, "effort": 60.})
        manipulator.event("gripper", target=position, **result)
        details = result.get("details", {})
        expected_contact = (contact and result.get("ros_status") in (4, 6) and details.get("stalled")
                            and .05 < details.get("position", 0) < position + .01)
        if not result["success"] and not expected_contact:
            raise RuntimeError(f"Gripper command to {position}: {result}")
        await asyncio.sleep(.7)  # Allow finger contact or release to settle.

    async def prepare_transport(state):
        try:
            if observe_only:
                manipulator.source = manipulator.station(state["start_location"])
                await asyncio.to_thread(manipulator.add_tables)
                await stow()
                return {"execution_status": "transport_ready"}
            await asyncio.to_thread(manipulator.prepare, state["start_location"],
                                    state["end_location"], state["target_object"])
            await stow()
            await go_to(state["end_location"])
            await move_arm(manipulator.destination.view_pose, "destination_overview", frame="base_link")
            surface = await asyncio.to_thread(manipulator.inspect_destination)
            await stow()
            return {"execution_status": "transport_ready", "place_surface": surface}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def prepare_camera(state):
        try:
            view = await asyncio.to_thread(manipulator.source_view_pose)
            await move_arm(view, "source_overview", frame="base_link")
            return {"camera_after": time.monotonic(), "execution_status": "camera_positioned"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def navigate(state):
        destination = state["end_location"] if state["task_type"] == "navigate" else state.get("start_location")
        if bridge is not None and destination:
            try:
                pickup = None
                if state["task_type"] == "transport" and manipulator is not None and not observe_only:
                    pickup = manipulator.source.pickup_pose
                navigation_result = await go_to(destination, pickup)
                if pickup is not None:
                    navigation_result["pickup_pose"] = asdict(pickup)
            except Exception as exc:
                return {"execution_status": "navigation_failed", "current_phase": "failed",
                        "navigation_result": {"success": False, "location": destination}, "error": str(exc)}
        else:
            navigation_result = {}
        if state["task_type"] == "navigate":
            return {"execution_status": "nav_success", "current_phase": "done",
                    "navigation_result": navigation_result}
        return {"execution_status": "nav_success", "current_phase": "perceiving",
                "navigation_result": navigation_result, "camera_after": time.monotonic()}

    async def capture(state):
        if bridge is None:
            try:
                path = Path(state["image"])
                with Image.open(path) as source:
                    width, height = source.size
                    if evidence:
                        evidence.camera(source.convert("RGB"), {"source": "saved_image", "width": width, "height": height})
                        path = evidence.directory / "source.png"
                return {"image_url": image_data_url(path), "image_width": width, "image_height": height,
                        "execution_status": "image_ready"}
            except (KeyError, OSError) as exc:
                return {"execution_status": "camera_unavailable", "current_phase": "failed",
                        "error": f"An explicit saved image is required without ROS ({type(exc).__name__})"}
        deadline = time.monotonic() + capture_timeout
        last_error = "No camera frame received"
        while time.monotonic() < deadline:
            try:
                snapshot = bridge.camera_snapshot(received_after=state["camera_after"])
                rgb, depth, info = await asyncio.to_thread(snapshot.to_arrays)
                rgb = np.asarray(rgb)
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError("Camera RGB conversion must produce a BGR8 image")
                calibration = None
                if manipulator is not None and state["task_type"] == "transport":
                    calibration = await asyncio.to_thread(manipulator.remember_frame, snapshot, rgb, depth, info)
                pil = Image.fromarray(rgb[:, :, ::-1])
                buffer = io.BytesIO()
                pil.save(buffer, format="PNG")
                metadata = {"frame_id": snapshot.rgb.header.frame_id, "depth_encoding": snapshot.depth.encoding,
                            "stamp_sec": snapshot.rgb.header.stamp.sec, "stamp_nanosec": snapshot.rgb.header.stamp.nanosec}
                if evidence:
                    evidence.camera(pil, {**metadata, "width": pil.width, "height": pil.height,
                                          "k": list(info.k), **(calibration or {})}, depth)
                return {"image": "", "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                        "image_width": pil.width, "image_height": pil.height, "camera_metadata": metadata,
                        "execution_status": "image_ready"}
            except CameraUnavailableError as exc:
                last_error = str(exc)
                await asyncio.sleep(0.05)
            except Exception as exc:
                return {"execution_status": "camera_error", "current_phase": "failed",
                        "error": f"Camera conversion failed ({type(exc).__name__})"}
        return {"execution_status": "camera_unavailable", "current_phase": "failed", "error": last_error}

    async def presence(state):
        from langchain_openai.decisions import Predicate
        try:
            m = get_models()
            with track_llm_call("item_presence_detection", m.presence_name, evidence) as call:
                response = await m.presence.ainvoke({
                    "input": HumanMessage(content=[{"type": "image_url", "image_url": {"url": state["image_url"]}}]),
                    "questions": {"object_presence": Predicate(instructions=f"Is {state['target_object']} present?")},
                })
                if hasattr(response, "model_dump"):
                    call["result"] = response.model_dump(mode="json")
                answer = response.predicates.get("object_presence")
                probability = answer.probability if answer is not None else None
                if probability is None or not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("No valid presence probability")
                assessment = "present" if probability >= 0.8 else "absent" if probability <= 0.2 else "uncertain"
                call.setdefault("result", {"probability": probability})
                call.update(probability=probability, assessment=assessment)
            return {"item_presence": assessment, "execution_status": f"object_{assessment}",
                    "current_phase": "perceiving" if assessment == "present" else "done"}
        except Exception as exc:
            return model_error("Presence detection", exc)

    async def boxes(state):
        try:
            m = get_models()
            with track_llm_call("bounding_boxes", m.boxes_name, evidence) as call:
                result = await m.boxes.ainvoke([HumanMessage(content=[
                    {"type": "text", "text": f"Detect all {state['target_object']} objects. Return box_2d "
                     "as [ymin, xmin, ymax, xmax] normalized to 0-1000. Return an empty list if absent."},
                    {"type": "image_url", "image_url": {"url": state["image_url"]}},
                ])])
                result = BoundingBoxes.model_validate(result)
                call["result"] = result.model_dump()
            pixel_boxes = [normalised_to_pixels(b.box_2d, state["image_width"], state["image_height"])
                           for b in result.boxes]
            if evidence:
                evidence.geometry(pixel_boxes, spatial_only=spatial_choices_only)
            transport = state["task_type"] == "transport" and not observe_only
            if transport and manipulator is not None and len(pixel_boxes) > 1:
                choice = resolve_choice(state.get("target_selector"), state["target_object"], pixel_boxes, spatial_only=spatial_choices_only)
                return {"boxes": pixel_boxes, "selected_box": choice or [],
                        "execution_status": "ready_to_grasp" if choice else "awaiting_target_selection",
                        "current_phase": "picking" if choice else "selecting_target"}
            return {"boxes": pixel_boxes, "selected_box": pixel_boxes[0] if transport and len(pixel_boxes) == 1 else [],
                    "execution_status": "no_detections" if not pixel_boxes else
                        ("ready_to_grasp" if manipulator is not None else "awaiting_grasp_planning")
                        if transport else "inspection_complete",
                    "current_phase": "picking" if pixel_boxes and transport else "done"}
        except Exception as exc:
            return model_error("Bounding-box detection", exc)

    def ask_target(state):
        candidates = numbered_boxes(state["boxes"])
        target = state["target_object"]
        question = (f"I see {len(candidates)} {target.removeprefix('the ')}s on {state['start_location']}. "
                    "Which one should I pick? Describe its position in the camera view, "
                    "for example leftmost, rightmost, middle, topmost, or bottommost.")
        if not spatial_choices_only:
            question += f" You can also give a box number from 1 to {len(candidates)}."
        if state["selection_dialogue"]:
            question = "That reply does not identify one box. " + question
        reply = interrupt({"question": question, "boxes": candidates,
                           "image": str(evidence.directory / "annotated.png") if evidence else None})
        dialogue = [*state["selection_dialogue"], {"question": question, "reply": reply}]
        if isinstance(reply, str) and reply.strip().casefold() in ("cancel", "stop"):
            return {"selection_dialogue": dialogue, "execution_status": "target_selection_cancelled",
                    "current_phase": "done"}
        choice = resolve_choice(reply, target, candidates, spatial_only=spatial_choices_only)
        return {"selection_dialogue": dialogue, "selected_box": choice or [],
                "execution_status": "ready_to_grasp" if choice else "awaiting_target_selection",
                "current_phase": "picking" if choice else "selecting_target"}

    async def plan_grasp(state):
        try:
            plan = await asyncio.to_thread(manipulator.plan_grasp, [state["selected_box"]])
            if evidence:
                evidence.geometry(state["boxes"], plan, selected=state["selected_box"], spatial_only=spatial_choices_only)
            return {"grasp_plan": plan, "execution_status": "grasp_planned"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def pick_object(state):
        try:
            plan = state["grasp_plan"]
            await asyncio.to_thread(manipulator.allow_contact)
            await command_gripper(.01)
            await move_arm(plan["lift"], "approach")
            await move_arm(plan["grasp"], "descend", linear=True)
            await command_gripper(.51 if plan.get("shape") == "cylinder" else .54, contact=True)
            await asyncio.to_thread(manipulator.attach)
            x, y, z = plan["grasp"]
            await move_arm(plan["lift"], "lift", linear=True, shorter_lift=[x, y, z + .12])
            if not await asyncio.to_thread(manipulator.verify_held, "held-after-lift"):
                raise RuntimeError("Pickup was not verified")
            return {"grasp_verified": True, "execution_status": "grasp_verified", "current_phase": "go_to_end"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def carry_to_destination(state):
        try:
            await stow()
            result = await go_to(state["end_location"])
            if not await asyncio.to_thread(manipulator.verify_held, "held-after-navigation"):
                raise RuntimeError("Grasp was not verified after navigation")
            return {"transport_result": result, "execution_status": "carry_complete", "current_phase": "placing"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def place_object(state):
        try:
            x, y, plane = state["place_surface"]
            place_z = plane + .247
            hover = max(state["grasp_plan"]["lift"][2], place_z + .173)
            await move_arm([x, y, hover], "place_approach")
            await move_arm([x, y, place_z], "lower_to_place", linear=True)
            await command_gripper(.01)
            await asyncio.to_thread(manipulator.detach)
            await move_arm([x, y, hover], "place_retreat", linear=True)
            await asyncio.to_thread(manipulator.restore_collisions)
            await move_arm(manipulator.destination.view_pose, "destination_verification_overview", frame="base_link")
            return {"execution_status": "released"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    async def verify_placement(state):
        try:
            result = await asyncio.to_thread(manipulator.verify_place)
            if not result.get("verified"):
                raise RuntimeError("Placement was not verified")
            return {"placement_verified": True, "execution_status": "pick_place_complete", "current_phase": "done"}
        except Exception as exc:
            return {"execution_status": "manipulation_failed", "current_phase": "failed", "error": str(exc)}

    builder = StateGraph(RobotState)
    for name, node in (("decompose_task", decompose), ("request_task_clarification", clarify),
                       ("validate_locations", validate_locations), ("navigate_to_location", navigate),
                       ("capture_image", capture), ("item_presence", presence), ("bounding_boxes", boxes),
                       ("prepare_transport", prepare_transport), ("prepare_camera", prepare_camera),
                       ("ask_target", ask_target), ("plan_grasp", plan_grasp), ("pick_object", pick_object),
                       ("carry_to_destination", carry_to_destination), ("place_object", place_object),
                       ("verify_placement", verify_placement)):
        builder.add_node(name, node)
    builder.add_edge(START, "decompose_task")
    builder.add_conditional_edges("decompose_task", lambda s: END if s["execution_status"] == "model_error" else
                                 "request_task_clarification" if s["missing_fields"] else "validate_locations")
    builder.add_edge("request_task_clarification", END)
    builder.add_conditional_edges("validate_locations", lambda s: END if s["execution_status"] != "locations_validated"
                                 else "prepare_transport" if s["task_type"] == "transport" and manipulator is not None
                                 else "navigate_to_location")
    builder.add_conditional_edges("prepare_transport", lambda s:
                                 "navigate_to_location" if s["execution_status"] == "transport_ready" else END)
    builder.add_conditional_edges("navigate_to_location", lambda s:
                                 END if s["current_phase"] != "perceiving" else "prepare_camera"
                                 if s["task_type"] == "transport" and manipulator is not None else "capture_image")
    builder.add_conditional_edges("prepare_camera", lambda s:
                                 "capture_image" if s["execution_status"] == "camera_positioned" else END)
    builder.add_conditional_edges("capture_image", lambda s:
                                 "item_presence" if s["execution_status"] == "image_ready" else END)
    builder.add_conditional_edges("item_presence", lambda s:
                                 "bounding_boxes" if s["item_presence"] == "present" else END)
    def after_detection(state):
        return {"ready_to_grasp": "plan_grasp", "awaiting_target_selection": "ask_target"}.get(
            state["execution_status"], END)
    builder.add_conditional_edges("bounding_boxes", after_detection)
    builder.add_conditional_edges("ask_target", after_detection)
    for source, destination, status in (("plan_grasp", "pick_object", "grasp_planned"),
                                       ("pick_object", "carry_to_destination", "grasp_verified"),
                                       ("carry_to_destination", "place_object", "carry_complete"),
                                       ("place_object", "verify_placement", "released")):
        builder.add_conditional_edges(source, lambda s, dest=destination, expected=status:
                                     dest if s["execution_status"] == expected else END)
    builder.add_edge("verify_placement", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def normalised_to_pixels(box_2d: List[int], width: int, height: int) -> List[int]:
    ymin, xmin, ymax, xmax = box_2d
    return [round(xmin * width / 1000), round(ymin * height / 1000),
            round(xmax * width / 1000), round(ymax * height / 1000)]


def tool_unproject_pixel(u, v, depth_frame, camera_info, depth_encoding="16UC1"):
    """Unproject valid depth near a pixel, respecting native ROS depth units."""
    if depth_encoding not in ("16UC1", "32FC1"):
        raise ValueError("Unsupported depth encoding")
    height, width = depth_frame.shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Pixel outside depth image")
    window = depth_frame[max(0, v-2):min(height, v+3), max(0, u-2):min(width, u+3)]
    valid = window[np.isfinite(window) & (window > 0)]
    if not valid.size:
        raise ValueError("Depth invalid")
    z = float(np.median(valid)) / (1000.0 if depth_encoding == "16UC1" else 1.0)
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    if not all(np.isfinite(n) for n in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("Camera intrinsics invalid")
    return (u-cx)*z/fx, (v-cy)*z/fy, z


def image_data_url(path: Path) -> str:
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def draw_bounding_boxes(image_path, boxes, output_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)
    image.save(output_path)


# Backwards-compatible saved-image graph. ROS applications explicitly bind their bridge.
robot_brain = build_robot_brain()
print(robot_brain.get_graph().draw_mermaid())

async def run_with_selection(graph, state, config, replies=()):
    """Keep the active controller and its registered camera observation across turns."""
    scripted = iter(replies)
    result = await graph.ainvoke(state, config=config)
    while result.get("__interrupt__"):
        prompt = result["__interrupt__"][0].value
        print(prompt["question"], flush=True)
        if prompt.get("image"):
            print(f"Camera view: {prompt['image']}", flush=True)
        reply = next(scripted, None)
        if reply is None:
            if not sys.stdin.isatty():
                break
            try:
                reply = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
        else:
            print(f"> {reply}", flush=True)
        result = await graph.ainvoke(Command(resume=reply), config=config)
    return {k: v for k, v in result.items() if k != "__interrupt__"}


async def run_cli(args):
    config = {"configurable": {"thread_id": uuid4().hex}}
    state = {"task_input": args.instruction, "chat_history": []}
    evidence = RunEvidence(args.artifacts / config["configurable"]["thread_id"])
    if args.image:
        state["image"] = str(args.image)
        result = await build_robot_brain(evidence=evidence).ainvoke(state, config=config)
        evidence.finish(result)
        return result
    with ROSBridge(enable_motion=args.enable_motion, locations=load_locations(args.locations)) as bridge:
        manipulator = None
        if args.stations:
            from pick_place import PickPlaceController, load_stations
            manipulator = PickPlaceController(bridge, load_stations(args.stations), evidence.directory / "camera")
        graph = build_robot_brain(bridge, manipulator=manipulator, capture_timeout=15, evidence=evidence)
        result = await run_with_selection(graph, state, config, args.reply)
        evidence.finish(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--locations", type=Path, default=Path(__file__).with_name("locations.json"))
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--image", type=Path, help="Saved-image perception mode without ROS or navigation")
    parser.add_argument("--stations", type=Path, help="Calibrated two-table manipulation station configuration")
    parser.add_argument("--reply", action="append", default=[], help="Scripted reply, consumed only when asked")
    parser.add_argument("--artifacts", type=Path, default=Path("logs/pick-place"))
    args = parser.parse_args()
    log_dir = Path(__file__).with_name("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_dir / "agentic.log"), logging.StreamHandler()])
    try:
        result = asyncio.run(run_cli(args))
        print(json.dumps({k: v for k, v in result.items() if k != "image_url"}, default=str, indent=2))
        return 0 if result.get("execution_status") in ("nav_success", "inspection_complete", "object_absent",
                                                       "pick_place_complete") else 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
