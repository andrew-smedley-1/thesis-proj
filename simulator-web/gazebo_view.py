"""Capture only the Gazebo GUI's X11 client window."""
import io
import os
import re
import shutil
import struct
import subprocess
from threading import Lock
import time


GAZEBO_WINDOW = re.compile(
    r'^\s*(0x[0-9a-fA-F]+)\s+.*\("gz-sim-gui" "Gazebo GUI"\)\s+(\d+)x(\d+)',
    re.MULTILINE,
)


XWD_HEADER = struct.Struct(">25I")
RAW_MODES = {(24, 0): "BGR", (24, 1): "RGB", (32, 0): "BGRX", (32, 1): "XRGB"}


def capture_available():
    return bool(os.environ.get("DISPLAY") and shutil.which("xwininfo") and shutil.which("xwd"))


def decode_window(data):
    from PIL import Image

    (header_size, version, layout, _, width, height, x_offset, byte_order,
     _, _, _, bits_per_pixel, stride, visual, red, green, blue, _, _, colors, *_) = XWD_HEADER.unpack_from(data)
    mode = RAW_MODES.get((bits_per_pixel, byte_order))
    if (version != 7 or layout != 2 or x_offset != 0 or visual not in (4, 5) or mode is None
            or (red, green, blue) != (0xFF0000, 0xFF00, 0xFF)):
        raise ValueError("Unsupported X11 window pixel format")
    offset = header_size + colors * 12
    return Image.frombytes("RGB", (width, height), data[offset:], "raw", mode, stride, 1)


class GazeboView:
    def __init__(self):
        self.display = os.environ["DISPLAY"]
        self.lock = Lock()
        self.window = None
        self.search_after = 0.0
        self.frame = None
        self.captured_at = 0.0

    def find_window(self):
        tree = subprocess.check_output(
            ["xwininfo", "-display", self.display, "-root", "-tree"],
            text=True, errors="replace", stderr=subprocess.DEVNULL, timeout=2,
        )
        candidates = [int(window, 16) for window, width, height in GAZEBO_WINDOW.findall(tree)
                      if int(width) >= 300 and int(height) >= 200]
        return candidates[0] if len(candidates) == 1 else None

    def capture(self):
        with self.lock:
            now = time.monotonic()
            if self.frame and now - self.captured_at < 0.15:
                return self.frame
            try:
                if self.window is None and now >= self.search_after:
                    self.search_after = now + 1
                    self.window = self.find_window()
                if self.window is None:
                    return None
                raw = subprocess.check_output(
                    ["xwd", "-display", self.display, "-silent", "-id", hex(self.window)],
                    stderr=subprocess.DEVNULL, timeout=2,
                )
                with decode_window(raw) as image, io.BytesIO() as buffer:
                    if not image.getbbox():
                        return None
                    image.thumbnail((1200, 900))
                    image.save(buffer, format="JPEG", quality=80)
                    self.frame = buffer.getvalue()
                    self.captured_at = time.monotonic()
                    return self.frame
            except (OSError, ValueError, struct.error, subprocess.SubprocessError):
                self.window = None
                self.frame = None
                return None
