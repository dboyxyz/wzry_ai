# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import os
from dataclasses import dataclass
import cv2
import mss
import numpy as np
from PIL import Image
os.environ.setdefault('DISPLAY', ':1')

@dataclass
class CaptureConfig:
    top: int = 65
    left: int = 60
    width: int = 960
    height: int = 432

    @classmethod
    def from_env(cls) -> 'CaptureConfig':
        return cls(top=int(os.environ.get('WZRY_CAPTURE_TOP', '65')), left=int(os.environ.get('WZRY_CAPTURE_LEFT', '60')), width=int(os.environ.get('WZRY_CAPTURE_WIDTH', '960')), height=int(os.environ.get('WZRY_CAPTURE_HEIGHT', '432')))

    def as_monitor(self) -> dict[str, int]:
        return {'top': self.top, 'left': self.left, 'width': self.width, 'height': self.height}

class MSSCapture:

    def __init__(self, config: CaptureConfig | None=None):
        self.config = config or CaptureConfig.from_env()
        self._sct = mss.MSS()

    def grab_bgr(self) -> np.ndarray:
        frame = np.array(self._sct.grab(self.config.as_monitor()))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def grab_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.grab_bgr(), cv2.COLOR_BGR2RGB)

    def grab_pil(self) -> Image.Image:
        return Image.fromarray(self.grab_rgb())

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> 'MSSCapture':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
