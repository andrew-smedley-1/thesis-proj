"""Create LangChain tools around an explicitly supplied ROSBridge instance."""


def build_robot_tools(bridge):
    # The bridge and its CLI do not depend on LangChain.
    from langchain_core.tools import StructuredTool

    async def navigate_robot(x: float, y: float, yaw: float = 0.0, frame_id: str = "map") -> bool:
        """Await Nav2 arrival at a measured pose in meters and yaw radians."""
        return await bridge.navigate_async(x, y, yaw, frame_id)

    async def navigate_robot_to_location(location_name: str) -> dict:
        """Navigate to a configured measured location. Unknown names return unknown_location."""
        try:
            return (await bridge.navigate_to_location_async(location_name)).to_dict()
        except ValueError as exc:
            return {"status": "invalid_input", "success": False, "message": str(exc)}

    async def robot_camera_status() -> dict:
        """Check latest camera availability and encodings. Returns metadata, not image pixels."""
        return bridge.camera_status()

    async def move_robot_arm(x: float, y: float, z: float, qx: float, qy: float,
                             qz: float, qw: float, frame_id: str = "base_link", linear: bool = False) -> dict:
        """Move link_6 to a known pose using MoveIt: meters and unit xyzw quaternion.

        This is the arm flange, not the fingertips. Requires known collision geometry.
        """
        try:
            return (await bridge.move_arm_async(x, y, z, qx, qy, qz, qw, frame_id, linear=linear)).to_dict()
        except ValueError as exc:
            return {"status": "invalid_input", "success": False, "message": str(exc)}

    async def command_robot_gripper(position: float, effort: float = 50.0) -> dict:
        """Set finger_joint radians [0, 0.7], effort <=60. Open ~0.01, close ~0.6.

        A stalled gripper or completed command does not prove an object is held.
        """
        try:
            return (await bridge.grasp_async(position, effort)).to_dict()
        except ValueError as exc:
            return {"status": "invalid_input", "success": False, "message": str(exc)}

    async def robot_readiness() -> dict:
        """Check ROS action discovery, simulation clock and this bridge's motion state."""
        return bridge.readiness()

    async def cancel_robot_motion() -> dict:
        """Request cancellation of this bridge's current goal; this is not a hardware emergency stop."""
        return bridge.cancel().to_dict()

    return [StructuredTool.from_function(coroutine=f) for f in (
        navigate_robot, navigate_robot_to_location, move_robot_arm, command_robot_gripper,
        robot_readiness, robot_camera_status, cancel_robot_motion)]
