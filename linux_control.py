# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import time
from dataclasses import dataclass
import scrcpy
from scrcpy import const

@dataclass
class TouchPoint:
    x: int
    y: int
    active: bool = False

class ScrcpyController:

    def __init__(self, serial: str | None=None, max_width: int=800, bitrate: int=8000000, wait_after_start: float=1.5):
        self.serial = serial
        self.client = scrcpy.Client(device=serial, max_width=max_width, bitrate=bitrate, block_frame=False, stay_awake=True)
        self.client.start(threaded=True)
        time.sleep(max(0.1, wait_after_start))
        self.points: dict[int, TouchPoint] = {}

    def touch_down(self, pointer_id: int, x: int, y: int) -> None:
        self.client.control.touch(x, y, action=const.ACTION_DOWN, touch_id=pointer_id)
        self.points[pointer_id] = TouchPoint(x=x, y=y, active=True)

    def touch_move(self, pointer_id: int, x: int, y: int) -> None:
        if pointer_id not in self.points:
            self.touch_down(pointer_id, x, y)
            return
        self.client.control.touch(x, y, action=const.ACTION_MOVE, touch_id=pointer_id)
        self.points[pointer_id].x = x
        self.points[pointer_id].y = y
        self.points[pointer_id].active = True

    def touch_up(self, pointer_id: int) -> None:
        point = self.points.get(pointer_id)
        if point is None or not point.active:
            return
        self.client.control.touch(point.x, point.y, action=const.ACTION_UP, touch_id=pointer_id)
        point.active = False

    def tap(self, x: int, y: int, pointer_id: int=0, press_ms: int=40) -> None:
        self.touch_down(pointer_id, x, y)
        time.sleep(max(0, press_ms) / 1000.0)
        self.touch_up(pointer_id)

    def send_legacy_script(self, script: str) -> None:
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            op = parts[0]
            if op == 'd' and len(parts) >= 4:
                pointer_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                self.touch_down(pointer_id, x, y)
            elif op == 'm' and len(parts) >= 4:
                pointer_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                self.touch_move(pointer_id, x, y)
            elif op == 'u' and len(parts) >= 2:
                pointer_id = int(parts[1])
                self.touch_up(pointer_id)
            elif op == 'c':
                continue

    def stop(self) -> None:
        for pointer_id in list(self.points):
            self.touch_up(pointer_id)
        self.client.stop()
