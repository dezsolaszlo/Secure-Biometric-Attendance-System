import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThread, QTimer, pyqtSignal

_HERE = Path(__file__).parent
MODEL_PATH = _HERE.parent / "models" / "facenet512.onnx"

MATCH_THRESHOLD = 0.50

def _load_haar_cascade() -> cv2.CascadeClassifier:
    file_name = "haarcascade_frontalface_default.xml"
    src = Path(cv2.data.haarcascades) / file_name

    if str(src).isascii():
        cascade = cv2.CascadeClassifier(str(src))
        if not cascade.empty():
            return cascade

    try:
        dst = Path(tempfile.gettempdir()) / file_name
        dst.write_bytes(src.read_bytes())
        cascade = cv2.CascadeClassifier(str(dst))
        if not cascade.empty():
            return cascade
    except Exception:
        pass
    return cv2.CascadeClassifier()

class FaceEmbedder:
    INPUT_SIZE = 160

    def __init__(self):
        self._session = self._input = None
        self.available = False

        if not MODEL_PATH.exists():
            print(f"Missing {MODEL_PATH.name}! Run export_model.py!")
            return

        try:
            import onnxruntime as ort
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            self._session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
            self._input = self._session.get_inputs()[0].name

            #Warm up, so the first face doesn't lag
            self._session.run(None, {self._input: np.zeros((1, self.INPUT_SIZE, self.INPUT_SIZE, 3), np.float32)})
            self.available = True
            print("ONNX model loaded and warmed up")
        except Exception as e:
            print(f"Load failed: {e}")


    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if not self.available or face_bgr is None or not face_bgr.size:
            return None
        try:
            # Resize, convert to RGB, normalize to [-1 ,1]
            rgb = cv2.cvtColor(cv2.resize(face_bgr, (self.INPUT_SIZE, self.INPUT_SIZE)), cv2.COLOR_BGR2RGB)
            norm = (rgb.astype(np.float32) - 127.5) / 128.0
            vec = self._session.run(None, {self._input: norm[None]})[0][0]
            mag = np.linalg.norm(vec)
            if mag > 0:
                return vec / mag
            else:
                return vec
        except Exception as e:
            print(f"Embed error: {e}")
            return None


class _InferenceSignals(QObject):
    done = pyqtSignal(list)
    free = pyqtSignal()

class InferenceWorker(QRunnable):
    """Processes a single video frame: detects faces, tracks them, and identifies them."""

    COOLDOWN_FRAMES = 10
    MATCH_DIST = 120
    MIN_CONFIRM = 5

    def __init__(self, frame, embedder, cascade, roster_embs, roster_names, roster_ids, tracks):
        super().__init__()
        self.frame = frame
        self.embedder = embedder
        self.cascade = cascade

        self.embs = roster_embs
        self.names = roster_names
        self.ids = roster_ids

        self.tracks = tracks
        self.signals = _InferenceSignals()

    def run(self) -> None:
        try:
            self.signals.done.emit(self._process())
        except Exception as e:
            print(f"[InferenceWorker] {e}")
            self.signals.done.emit([])
        finally:
            self.frame = None
            self.signals.free.emit()

    def _process(self) -> list:
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor = 1.1, minNeighbors = 5, minSize=(60,60)
        ) if not self.cascade.empty() else []

        for t in self.tracks:
            t["age"] += 1
        active = [t for t in self.tracks if t["age"] <= 5]

        updated, matched = [], set()
        for (x, y, w, h) in boxes:
            cx = x + w // 2
            cy = y + h // 2

            best_idx, best_dist = None, self.MATCH_DIST
            for i, t in enumerate(active):
                if i in matched:
                    continue
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_dist:
                    best_dist, best_idx = d, i

            if best_idx is not None:
                t = active[best_idx]
                matched.add(best_idx)
                t.update(x=x, y=y, w=w, h=h, cx=cx, cy=cy, age=0)
            else:
                t = {"x":x, "y":y, "w":w, "h":h, "cx":cx, "cy":cy, "age": 0,
                     "name": "Unknown", "label": "Unknown", "profile_id": None, "cooldown": 0, "history": deque(maxlen=10)}

            if t["name"] == "Unknown":
                if t["cooldown"] > 0:
                    t["cooldown"] -= 1
                else:
                    name, pid = self._match(x, y, w, h)
                    t["history"].append(name)
                    voted = Counter(t["history"]).most_common(1)[0][0]

                    t["name"] = voted
                    t["label"] = voted
                    t["profile_id"] = pid if voted != "Unknown" else None

                    if name == "Unknown":
                        t["cooldown"] = self.COOLDOWN_FRAMES

            updated.append(t)

        self.tracks[:] = updated
        return [t for t in updated if t["age"] == 0]

    def _match(self, x, y, w, h):
        vec = self.embedder.embed(self.frame[y:y+h, x:x+w])
        if vec is None or not self.embs:
            return "Unknown", None

        dist = [1.0 - float(np.dot(vec, e)) for e in self.embs]
        best_idx = int(np.argmin(dist))

        if dist[best_idx] < MATCH_THRESHOLD:
            return self.names[best_idx], self.ids[best_idx]
        return "Unknown", None


class CaptureThread(QThread):
    """Pulls frames from the webcam and broadcasts to the UI"""
    frame_ready = pyqtSignal(object)
    camera_lost = pyqtSignal()
    fps_updated = pyqtSignal(float)

    def __init__(self, cam_index: int = 0, parent = None):
        super().__init__(parent)
        self.cam_index = cam_index
        self._running = False
        self._busy = False

    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    def stop(self) -> None:
        self._running = False
        self.wait()

    def run(self) -> None:
        self._running = True

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            return self.camera_lost.emit()

        cap.set(cv2.CAP_PROP_FPS, 30)
        frames, fps_time = 0, time.perf_counter()

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.camera_lost.emit()
                break

            frames += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                self.fps_updated.emit(frames / (now - fps_time))
                frames, fps_time = 0, now

            if not self._busy:
                self.frame_ready.emit(frame)

        cap.release()


class ProfileWatcher(QTimer):
    """Polls the db every 5s to pick up new faces"""
    profiles_updated = pyqtSignal(list)

    def __init__(self, parent = None):
        super().__init__(parent)
        self.timeout.connect(self._poll)
        self.start(5000)

    def _poll(self) -> None:
        try:
            from core import database as db
            if new_profiles := db.drain_new_profiles():
                self.profiles_updated.emit(new_profiles)
        except Exception as e:
            print(f"[ProfileWatcher] {e}")

class CameraWatchdog(QTimer):
    """Raises an alarm if the capture thread stall for 3 secs"""
    camera_lost = pyqtSignal()
    camera_restored = pyqtSignal()

    STALL_TIMEOUT = 3.0

    def __init__(self, capture_thread: CaptureThread, parent=None):
        super().__init__(None)
        self._last_frame = time.monotonic()
        self._ok = True
        capture_thread.frame_ready.connect(self._ping)
        capture_thread.camera_lost.connect(self._fail)
        self.timeout.connect(self._check)
        self.start(2000)

    def _ping(self, _) -> None:
        self._last_frame = time.monotonic()
        if not self._ok:
            self._ok = True
            self.camera_restored.emit()

    def _fail(self) -> None:
        if self._ok:
            self._ok = False
            self.camera_lost.emit()

    def _check(self) -> None:
        if self._ok and (time.monotonic() - self._last_frame) > self.STALL_TIMEOUT:
            self._fail()
