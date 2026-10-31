import threading
import time

import cv2

from config import CONFIG


class CameraSource:
    """One camera, captured on a background thread so the UI never waits for a frame.

    Frames come out at the camera's full aspect ratio, never cropped or stretched. The capture size
    request picks the camera *mode*: on the Mac FaceTime HD and iPhone Continuity cameras the 16:9 modes
    (e.g. 1280x720, 1920x1080) are a top/bottom crop of the 4:3 sensor, so a 4:3 request shows much more
    of the upper body. Large frames are then downscaled (aspect-preserving) to `processing_max_width`
    on the capture thread, so pose detection and the preview use the very same image.
    """

    def __init__(self, index: int):
        self.index = index
        self.cap = None
        self.frame_size = None   # (width, height) actually delivered by the camera
        self.new_frame = False   # whether the last read() returned a frame not seen before
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._seq = 0
        self._read_seq = 0
        self._thread = None
        self._running = False

    def open(self) -> bool:
        if self.cap is not None:
            self.release()
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        if CONFIG.frame_width > 0 and CONFIG.frame_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG.target_fps)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            print(f"Camera {self.index}: capturing {self.frame_size[0]}x{self.frame_size[1]}")
            self._store(frame)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"camera-{self.index}", daemon=True)
        self._thread.start()
        return True

    @staticmethod
    def fit_for_processing(frame):
        """Downscale (never crop) frames wider than processing_max_width."""
        max_width = CONFIG.processing_max_width
        height, width = frame.shape[:2]
        if max_width <= 0 or width <= max_width:
            return frame
        scale = max_width / width
        return cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)

    def _store(self, frame):
        frame = self.fit_for_processing(frame)
        with self._lock:
            self._frame, self._frame_time = frame, time.monotonic()
            self._seq += 1

    def _capture_loop(self):
        cap = self.cap
        while self._running and cap is not None:
            ok, frame = cap.read()  # blocks until the camera delivers (on this thread, not the UI thread)
            if ok and frame is not None:
                self._store(frame)
            else:
                time.sleep(0.01)

    def read(self):
        """Latest frame without waiting. Fails when no frame has arrived for camera_frame_timeout_seconds."""
        if self.cap is None:
            return False, None
        with self._lock:
            frame, frame_time, seq = self._frame, self._frame_time, self._seq
        self.new_frame = seq != self._read_seq
        self._read_seq = seq
        if frame is None or time.monotonic() - frame_time > CONFIG.camera_frame_timeout_seconds:
            return False, None
        return True, frame

    def release(self):
        """Stop the capture thread first (so no read() is in flight), then release the device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._frame = None
