import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStatusBar, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QDateEdit, QFileDialog, QMessageBox, QLineEdit,
    QFormLayout, QGroupBox, QSizePolicy, QTextEdit, QSplitter, QDialog,
    QDialogButtonBox, QComboBox, QInputDialog, QAbstractItemView, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon, QFontMetrics
from PyQt6.QtCore import Qt, QSize, QDate, QThreadPool, pyqtSignal, pyqtSlot, QObject, QRect

from core import database as db
from core.engine import (
    FaceEmbedder, CaptureThread, InferenceWorker,
    ProfileWatcher, CameraWatchdog, _load_haar_cascade
)

# App Style

_DARK_QSS = """
QMainWindow, QWidget, QDialog { background: #121212; color: #e0e0e0; }
QTabWidget::pane { border: 1px solid #2a2a2a; }
QTabBar::tab { background: #1e1e1e; color: #aaa; padding: 8px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #00e5ff; border-bottom: 2px solid #00e5ff; }
QTabBar::tab:hover { background: #252525; }
QPushButton { background: #1e88e5; color: white; border: none; border-radius: 4px; padding: 6px 16px; font-size: 13px; }
QPushButton:hover { background: #1976d2; }
QPushButton:disabled { background: #333; color: #666; }
QPushButton#dangerBtn { background: #c62828; }
QPushButton#dangerBtn:hover { background: #b71c1c; }
QPushButton#secondaryBtn { background: #37474f; }
QPushButton#secondaryBtn:hover { background: #455a64; }
QLineEdit, QTextEdit, QDateEdit, QComboBox { background: #1e1e1e; border: 1px solid #333; border-radius: 3px; padding: 4px 8px; color: #e0e0e0; }
QTableWidget { background: #1a1a1a; gridline-color: #2a2a2a; alternate-background-color: #1e1e1e; }
QHeaderView::section { background: #1e1e1e; color: #aaa; padding: 4px; border: none; border-bottom: 1px solid #333; }
QListWidget { background: #1a1a1a; border: 1px solid #2a2a2a; }
QListWidget::item:selected { background: #1e88e5; color: white; }
QGroupBox { border: 1px solid #2a2a2a; border-radius: 4px; margin-top: 8px; padding-top: 8px; color: #aaa; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00e5ff; }
QStatusBar { background: #1e1e1e; color: #aaa; font-size: 12px; }
"""

_CYAN = QColor(0, 229, 255)
_RED = QColor(239, 83, 80)
_GREEN = QColor(102, 187, 106)
_AMBER = QColor(255, 167, 38)
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Core Engine

class AttendanceEngine(QObject):
    frame_with_tracks = pyqtSignal(object, list)
    checkin_logged = pyqtSignal(str, str)
    fps_updated = pyqtSignal(float)
    camera_lost = pyqtSignal()
    camera_restored = pyqtSignal()
    roster_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.embedder = FaceEmbedder()
        self._cascade = _load_haar_cascade()

        # State tracking
        self._tracks = []
        self._marked_today = set()
        self._marked_class_today = set()
        self._confirm_count = {}
        self._last_frame = None

        #Roster Memory
        self.known_embs = []
        self.known_names = []
        self.known_ids = []
        self.profile_map = {}
        self._load_roster()

        # Threading
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        self.capture = CaptureThread()
        self.watchdog = CameraWatchdog(self.capture)
        self.profile_watcher = ProfileWatcher()

        # Internal signals
        self.capture.frame_ready.connect(self._on_frame)
        self.capture.fps_updated.connect(self.fps_updated.emit)
        self.capture.camera_lost.connect(self.camera_lost.emit)
        self.watchdog.camera_restored.connect(self.camera_restored.emit)
        self.profile_watcher.profiles_updated.connect(self._on_new_profiles)

        self.capture.start()


    def _register_profile(self, profile: dict) -> None:
        emb = profile.get("embedding")
        if emb:
            self.known_embs.append(np.array(emb, dtype=np.float32))
            self.known_names.append(profile.get("full_name", "Unknown"))
            self.known_ids.append(profile["id"])
            self.profile_map[profile.get("full_name", "")] = profile

    def _load_roster(self) -> None:
        try:
            for p in db.load_all_profiles():
                self._register_profile(p)
        except Exception as e:
            print(f"Roster load failed: {e}")

    def add_profile(self, profile: dict) -> None:
        cnp = profile.get("cnp")
        if cnp:
            try:
                db_profile = db.lookup_profile_by_cnp(cnp)
                if db_profile:
                    self._register_profile(db_profile)
            except Exception as e:
                print(f"[Engine.add_profile] {e}")
        self.roster_changed.emit(len(self.known_names), db.count_profiles())

    @pyqtSlot(list)
    def _on_new_profiles(self, new_profiles: list) -> None:
        for p in new_profiles:
            self._register_profile(p)
        self.roster_changed.emit(len(self.known_names), db.count_profiles())

    @pyqtSlot(object)
    def _on_frame(self, frame) -> None:
        self._last_frame = frame
        self.capture.set_busy(True)

        worker = InferenceWorker(
            frame = frame,
            embedder = self.embedder,
            cascade = self._cascade,
            roster_embs = self.known_embs,
            roster_names = self.known_names,
            roster_ids = self.known_ids,
            tracks = self._tracks,
        )

        worker.signals.done.connect(self._on_inference_done)
        worker.signals.free.connect(lambda: self.capture.set_busy(False))

        self.thread_pool.start(worker)

    @pyqtSlot(list)
    def _on_inference_done(self, tracks: list) -> None:
        self._tracks = tracks
        if self._last_frame is not None:
            self.frame_with_tracks.emit(self._last_frame, tracks)

        try:
            active_classes = db.get_active_classes()
        except Exception:
            active_classes = []

        class_enrolled = {}
        for cls in active_classes:
            cid = cls["id"]
            try:
                class_enrolled[cid] = db.get_class_student_ids(cid)
            except Exception:
                class_enrolled[cid] = set()

        seen_names = set()

        for track in tracks:
            name = track.get("name", "Unknown")
            if name == "Unknown":
                continue

            seen_names.add(name)
            self._confirm_count[name] = self._confirm_count.get(name, 0) + 1

            if self._confirm_count[name] < InferenceWorker.MIN_CONFIRM:
                continue

            person = self.profile_map.get(name)
            if not person:
                continue

            pid = person["id"]

            # Gen Attendance
            if name not in self._marked_today:
                if db.log_attendance(pid):
                    self._marked_today.add(name)
                    ts = datetime.now().strftime("%H:%M")
                    self.checkin_logged.emit(name, ts)

            # Class Attendance
            for cls in active_classes:
                cid = cls["id"]
                if pid in class_enrolled.get(cid, set()):
                    key = (cid, pid)
                    if key not in self._marked_class_today:
                        if db.log_class_attendance(cid, pid):
                            self._marked_class_today.add(key)
                            print(f"[ClassAttendance] {name} -> {cls['name']}")

        for name in list(self._confirm_count):
            if name not in seen_names:
                self._confirm_count[name] = 0

    def stop(self) -> None:
        self.capture.stop()
        self.thread_pool.waitForDone(2000)


# LogIn Widgets

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Attendance System - Login")
        self.setMinimumSize(400, 320)
        self._role = None
        self._data = None

        icon_path = Path(__file__).parent / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Attendance System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00e5ff;"
                            "padding: 20px 0 10px 0; background: #0d1a2e;")
        layout.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_teacher_tab(), "  Teacher / Admin  ")
        self._tabs.addTab(self._build_student_tab(), "  Student  ")
        layout.addWidget(self._tabs)

    def _build_teacher_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 16, 24, 16)

        self._is_first_run = (db.count_teachers() == 0)
        if self._is_first_run:
            info = QLabel("No admin account exist yet.\n Create the first administrator account:")
            info.setStyleSheet("color: #ffa726; font-size: 12px;")
            layout.addWidget(info)

        form = QFormLayout()
        self._t_username = QLineEdit(placeholderText="Username")
        self._t_password = QLineEdit(placeholderText="Password")
        self._t_password.setEchoMode(QLineEdit.EchoMode.Password)

        form.addRow("Username:", self._t_username)
        form.addRow("Password:", self._t_password)

        if self._is_first_run:
            self._t_fullname = QLineEdit(placeholderText="Display name (e.g. Prof. Smith)")
            form.addRow("Full Name:", self._t_fullname)
        else:
            self._t_fullname = None

        layout.addLayout(form)

        btn_label = "Create Account & Login" if self._is_first_run else "Login"
        login_btn = QPushButton(btn_label)
        login_btn.setFixedHeight(38)
        login_btn.clicked.connect(self._on_teacher_login)
        self._t_password.returnPressed.connect(login_btn.click)
        layout.addWidget(login_btn)

        self._teacher_error = QLabel("", alignment=Qt.AlignmentFlag.AlignCenter)
        self._teacher_error.setStyleSheet("color: #ef5350;")
        layout.addWidget(self._teacher_error)
        layout.addStretch()
        return widget

    def _on_teacher_login(self) -> None:
        username = self._t_username.text().strip()
        password = self._t_password.text()

        if not username or not password:
            self._teacher_error.setText("Username and Password are required.")
            return

        if self._is_first_run:
            full_name = (self._t_fullname.text().strip() if self._t_fullname else "") or username
            if not db.create_teacher(username, password, full_name):
                self._teacher_error.setText("Failed to create account.")
                return

        teacher = db.authenticate_teacher(username, password)
        if not teacher:
            self._teacher_error.setText("Invalid username or password.")
            return

        self._role = "teacher"
        self._data = teacher
        self.accept()

    def _build_student_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 16, 24, 16)

        info = QLabel("Enter your CNP to view your attendance records:")
        info.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(info)

        self._s_cnp = QLineEdit(placeholderText="13-digit CNP")
        self._s_cnp.setMaxLength(13)
        layout.addWidget(self._s_cnp)

        btn = QPushButton("View My Attendance")
        btn.setFixedHeight(38)
        btn.clicked.connect(self._on_student_login)
        self._s_cnp.returnPressed.connect(btn.click)
        layout.addWidget(btn)

        self._student_error = QLabel("", alignment=Qt.AlignmentFlag.AlignCenter)
        self._student_error.setStyleSheet("color: #ef5350;")
        layout.addWidget(self._student_error)
        layout.addStretch()
        return widget

    def _on_student_login(self) -> None:
        cnp = self._s_cnp.text().strip()
        if not cnp:
            self._student_error.setText("Please enter your CNP.")
            return

        profile = db.lookup_profile_by_cnp(cnp)
        if not profile:
            self._student_error.setText("CNP not found. Please contact your teacher.")
            return

        self._role = "student"
        self._data = profile
        self.accept()

    def result_data(self) -> tuple:
        return self._role, self._data

# UI Utils

def _build_readonly_table(headers: list, stretch_col: int = -1) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)

    for col in range(len(headers)):
        mode = QHeaderView.ResizeMode.Stretch if col == stretch_col else QHeaderView.ResizeMode.ResizeToContents
        table.horizontalHeader().setSectionResizeMode(col, mode)
    return table

class _ClassEditDialog(QDialog):
    def __init__(self, name: str = "", description: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Details")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText("e.g. Mathematics 10A")
        self._desc_edit = QLineEdit(description)
        self._desc_edit.setPlaceholderText("Optional description")
        form.addRow("Class name:", self._name_edit)
        form.addRow("Class description:", self._desc_edit)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self) -> None:
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Required", "Class name cannot be empty.")
            return
        self.accept()

    def values(self)  -> tuple[str, str]:
        return self._name_edit.text().strip(), self._desc_edit.text().strip()


class _DrumRoller(QWidget):
    valueChanged =pyqtSignal(int)

    ROW_H = 36
    ROWS = 5
    ACCENT = "#1976d2"

    def __init__(self, lo: int, hi: int, initial: int = 0, parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self._val = max(lo, min(hi, initial))
        self._drag_offset = 0.0
        self._dragging = False

        self.setFixedSize(72, self.ROW_H  * self.ROWS)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def value(self) -> int:
        return self._val

    def setValue(self, v: int, reset_drag: bool = True):
        v = max(self._lo, min(self._hi, v))
        if v != self._val:
            self._val = v
            if reset_drag:
                self._drag_offset = 0.0
            self.update()
            self.valueChanged.emit(self._val)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        mid = h // 2
        half = self.ROWS // 2

        # Center section highlighted indicator
        p.fillRect(QRect(4, mid - self.ROW_H // 2, w - 8, self.ROW_H), QColor(self.ACCENT))

        # Visible sliding
        for offset in range(-half, half+1):
            frac = offset - self._drag_offset
            val = self._val + offset - round(self._drag_offset)
            y = mid + int(frac * self.ROW_H) - self.ROW_H // 2

            if y + self.ROW_H < 0 or y > h:
                continue

            dist = abs(frac)
            is_mid = dist < 0.5
            in_range = self._lo <= val <=  self._hi

            #Dynamic fading
            alpha = max(30, int(255 * (1.0 - min(dist, 2) / 2.0))) if in_range else 0
            color = QColor("white") if is_mid else QColor(200, 200, 200, alpha)

            p.setPen(color)
            p.setFont(QFont("Arial", 15 if is_mid else 12,
                            QFont.Weight.Bold if is_mid else QFont.Weight.Normal))
            p.drawText(0, y, w, self.ROW_H, Qt.AlignmentFlag.AlignCenter, f"{val:02d}")

        #Top / bottom edge
        for i in range(self.ROW_H):
            a = int(200 * (1 - i / self.ROW_H))
            p.fillRect(0, i, w, 1, QColor(18, 18, 18, a))
            p.fillRect(0, h - 1 - i, w, 1, QColor(18, 18, 18, a))

        p.end()

    #User Interection Handlings
    def wheelEvent(self, e):
        self.setValue(self._val + (-1 if e.angleDelta().y() > 0 else 1))

    def mousePressEvent(self ,e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_y = e.position().y()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._drag_offset = -(e.position().y() - self._start_y) / self.ROW_H
            self.update()

    def mouseReleaseEvent(self, e):
        if self._dragging:
            self._dragging = False
            steps = round(self._drag_offset)
            self._drag_offset = 0.0
            self.setValue(self._val + steps)


class _TimePickerPopup(QDialog):
    time_confirmed = pyqtSignal(int, int)

    def __init__(self, hour: int = 8, minute: int = 0, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet(
            "QDialog { background: #121212; border: 1px solid #333; border-radius: 8px; }"
        )
        self.setFixedWidth(185)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._h_drum = _DrumRoller(0, 23, hour, self)
        self._m_drum = _DrumRoller(0, 59, minute, self)

        sep = QLabel(":")
        sep.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        sep.setFixedWidth(18)

        drums = QHBoxLayout()
        drums.addWidget(self._h_drum)
        drums.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)
        drums.addWidget(self._m_drum)
        layout.addLayout(drums)

        btn = QPushButton("Confirm")
        btn.setStyleSheet(
            "QPushButton { background: #1976d2; color: white; border-radius: 4px;"
            "padding: 6px; font-weight: bold; }"
            "QPushButton:hover { background: #1565c0; }"
        )
        btn.clicked.connect(self._confirm)
        layout.addWidget(btn)

    def _confirm(self):
        self.time_confirmed.emit(self._h_drum.value(), self._m_drum.value())
        self.accept()

class _TimePickerButton(QPushButton):
    timeChanged = pyqtSignal(int, int)

    def __init__(self, placeholder: str = "-- : --", parent=None):
        super().__init__(placeholder, parent)
        self._hour = self._minute = None
        self.setStyleSheet(
            "QPushButton {background: #1e1e1e; color: #e0e0e0; border: 1px solid #444;"
            "border-radius: 4px; padding: 6px 12px; font-size: 14px; }"
            "QPushButton:hover { border-color: #1976d2; color: white; }"
        )
        self.clicked.connect(self._open)

    def _open(self):
        h = self._hour if self._hour is not None else 8
        m = self._minute if self._minute is not None else 0
        popup = _TimePickerPopup(h, m, self)
        popup.time_confirmed.connect(self.setTime)
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.exec()

    def setTime(self, h: int, m: int):
        self._hour, self._minute = h, m
        self.setText(f"{h:02d}:{m:02d}")
        self.timeChanged.emit(h, m)

    def hasValue(self) -> bool:
        return self._hour is not None

    def value(self) -> str:
        return f"{self._hour:02d}:{self._minute:02d}" if self.hasValue() else ""

    def totalMinutes(self) -> int | None:
        return (self._hour * 60 + self._minute) if self.hasValue() else None

class _AddScheduleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Schedule Slot")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._day_combo = QComboBox()
        self._day_combo.addItems(_DAY_NAMES)

        self._start_btn = _TimePickerButton("Pick start time")
        self._end_btn = _TimePickerButton("Pick end time")

        #Auto fill end = start + 2h
        self._start_btn.timeChanged.connect(self._auto_end)

        form.addRow("Day:", self._day_combo)
        form.addRow("Start Time:", self._start_btn)
        form.addRow("End Time:", self._end_btn)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _auto_end(self, h: int, m: int):
        self._end_btn.setTime((h + 2) % 24, m)

    def _on_accept(self) -> None:
        if not self._start_btn.hasValue() or not self._end_btn.hasValue():
            QMessageBox.warning(self, "Missing Time",
                                "Please pick both a start and an end time.")
            return
        s = self._start_btn.totalMinutes()
        e = self._end_btn.totalMinutes()
        if s == e:
            QMessageBox.warning(self, "Invalid",
                                "Start and end time cannot be the same.")
        self.accept()


    def values(self) -> tuple[int, str, str]:
        return (
            self._day_combo.currentIndex(),
            self._start_btn.value(),
            self._end_btn.value(),
        )


class _EnrollStudentDialog(QDialog):
    def __init__(self, all_profiles: list, enrolled_ids: set, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enroll Students")
        self.setMinimumSize(380, 400)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select students to enroll (students already enrolled are hidden):"))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._profile_ids: list[int] = []

        for p in all_profiles:
            if p["id"] not in enrolled_ids:
                item = QListWidgetItem(f"{p.get('full_name', '?')} ({p.get('cnp', '')})")
                self._list.addItem(item)
                self._profile_ids.append(p["id"])

        layout.addWidget(self._list)
        if self._list.count() == 0:
            layout.addWidget(QLabel("All enrolled profiles are already in this class."))

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_ids(self) -> list[int]:
        return [self._profile_ids[i] for i in range(self._list.count()) if self._list.item(i).isSelected()]

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(480, 360))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("Waiting for camera...")
        self.setStyleSheet("background: #0d0d0d; color: #555; font-size: 14px;")

    def update_frame(self, frame, tracks):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)

        if tracks:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setFont(QFont("Arial", max(8, w // 80)))

            for t in tracks:
                known = t.get("name", "Unknown") != "Unknown"
                col = _GREEN if known else _RED
                bx, by, bw, bh = t["x"], t["y"], t["w"], t["h"]

                # Bounding box
                painter.setPen(QPen(col, 2))
                painter.drawRect(bx, by, bw, bh)

                # Name tag background
                tag_h = max(20, h // 28)
                painter.fillRect(bx, by + bh, bw, tag_h, col)

                # Name tag text
                painter.setPen(QColor(255,255, 255))
                painter.drawText(bx + 4, by + bh + tag_h - 4, t.get("label", "Unknown"))

            painter.end()

        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.setPixmap(scaled)

    def show_message(self, text):
        self.clear()
        self.setText(text)

# Teacher UI Tabs

class ClassesTab(QWidget):
    def __init__(self, teacher_id: int,  parent=None):
        super().__init__(parent)
        self._teacher_id = teacher_id
        self._current_class_id = None

        # LEft panel
        left = QWidget()
        left.setFixedWidth(220)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        hdr = QLabel("My Classes")
        hdr.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 13px;")
        left_layout.addWidget(hdr)

        self._class_list = QListWidget()
        self._class_list.currentItemChanged.connect(self._on_class_selected)
        left_layout.addWidget(self._class_list)

        left_btns = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._on_new_class)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit_class)

        self._del_btn = QPushButton("Delete")
        self._del_btn.setObjectName("dangerBtn")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._on_delete_class)

        left_btns.addWidget(new_btn)
        left_btns.addWidget(self._edit_btn)
        left_btns.addWidget(self._del_btn)
        left_layout.addLayout(left_btns)

        # Right panel
        self._right = QWidget()
        right_layout = QVBoxLayout(self._right)

        self._class_header = QLabel("Select a class")
        self._class_header.setStyleSheet("color: #555; font-size: 14px;")
        right_layout.addWidget(self._class_header)

        sched_box = QGroupBox("Weekly Schedule")
        sched_vbox = QVBoxLayout(sched_box)
        self._sched_table = _build_readonly_table(["Day", "Start", "End"], stretch_col=0)
        self._sched_table.setMaximumHeight(160)
        self._sched_ids = []
        sched_vbox.addWidget(self._sched_table)

        sched_btns = QHBoxLayout()
        add_sched_btn = QPushButton("Add Slot")
        add_sched_btn.clicked.connect(self._on_add_schedule)

        rm_sched_btn = QPushButton("Remove")
        rm_sched_btn.setObjectName("dangerBtn")
        rm_sched_btn.clicked.connect(self._on_remove_schedule)

        sched_btns.addWidget(add_sched_btn)
        sched_btns.addWidget(rm_sched_btn)
        sched_btns.addStretch()
        sched_vbox.addLayout(sched_btns)
        right_layout.addWidget(sched_box)

        # Students Sub-Box
        stu_box = QGroupBox("Enrolled Students")
        stu_vbox = QVBoxLayout(stu_box)
        self._stu_table = _build_readonly_table(["Name", "CNP"], stretch_col=0)
        self._stu_ids = []
        stu_vbox.addWidget(self._stu_table)

        stu_btns = QHBoxLayout()
        enroll_btn = QPushButton("Enroll Student")
        enroll_btn.clicked.connect(self._on_enroll_student)

        rm_stu_btn = QPushButton("Remove")
        rm_stu_btn.setObjectName("dangerBtn")
        rm_stu_btn.clicked.connect(self._on_remove_student)

        stu_btns.addWidget(enroll_btn)
        stu_btns.addWidget(rm_stu_btn)
        stu_btns.addStretch()
        stu_vbox.addLayout(stu_btns)
        right_layout.addWidget(stu_box)

        # Report Sub-Box
        report_box = QGroupBox("Attendance Report")
        report_vbox = QVBoxLayout(report_box)
        self._report_table = _build_readonly_table(["Student", "Sessions Held", "Present", "Absent", "Rate%"], stretch_col=0)
        report_vbox.addWidget(self._report_table)

        ref_report_btn = QPushButton("Refresh Report")
        ref_report_btn.setObjectName("secondaryBtn")
        ref_report_btn.clicked.connect(self._on_refresh_report)
        ref_report_btn.setFixedWidth(130)
        report_vbox.addWidget(ref_report_btn, alignment=Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(report_box)

        # Main Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._refresh_class_list()

    def _refresh_class_list(self) -> None:
        self._class_list.blockSignals(True)
        self._class_list.clear()

        for cls in db.list_classes(teacher_id=self._teacher_id):
            item = QListWidgetItem(cls["name"])
            item.setData(Qt.ItemDataRole.UserRole, cls["id"])
            self._class_list.addItem(item)

            if self._current_class_id == cls["id"]:
                self._class_list.setCurrentItem(item)

        self._class_list.blockSignals(False)

    def _on_class_selected(self, current, _previous) -> None:
        if current is None:
            self._current_class_id = None
            self._class_header.setText("Select a class")
            self._class_header.setStyleSheet("color: #555; font-size: 14px;")
            self._edit_btn.setEnabled(False)
            self._del_btn.setEnabled(False)
            self._sched_table.setRowCount(0)
            self._stu_table.setRowCount(0)
            self._report_table.setRowCount(0)
            return

        cid = current.data(Qt.ItemDataRole.UserRole)
        self._current_class_id = cid
        cls = db.get_class(cid)

        name = cls["name"] if cls else "?"
        desc = cls.get("description", "") if cls else ""
        self._class_header.setText(f"{name} - {desc}" if desc else name)
        self._class_header.setStyleSheet("color: #00e5ff; font-size: 14px; font-weight: bold;")

        self._edit_btn.setEnabled(True)
        self._del_btn.setEnabled(True)

        self._refresh_schedule(cid)
        self._refresh_students(cid)
        self._refresh_attendance_report(cid)


    # CRUD

    def _on_new_class(self) -> None:
        dlg = _ClassEditDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, desc = dlg.values()
            self._current_class_id = db.create_class(self._teacher_id, name, desc)
            self._refresh_class_list()

    def _on_edit_class(self) -> None:
        if not self._current_class_id: return
        cls = db.get_class(self._current_class_id)

        dlg = _ClassEditDialog(cls["name"], cls.get("description", ""), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, desc = dlg.values()
            db.update_class(self._current_class_id, name, desc)
            self._refresh_class_list()

    def _on_delete_class(self) -> None:
        if not self._current_class_id: return
        cls = db.get_class(self._current_class_id)
        name = cls["name"] if cls else "this class"

        ans = QMessageBox.question(
            self, "Delete Class",
            f"Delete '{name}' and all its schedules and attendance records?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            db.delete_class(self._current_class_id)
            self._current_class_id = None
            self._refresh_class_list()
            self._on_class_selected(None, None)

    def _refresh_schedule(self, class_id: int) -> None:
        schedules = db.get_class_schedule(class_id)
        self._sched_ids = [s["id"] for s in schedules]
        self._sched_table.setRowCount(0)

        for r, s in enumerate(schedules):
            self._sched_table.insertRow(r)
            self._sched_table.setItem(r, 0, QTableWidgetItem(_DAY_NAMES[s["day_of_week"]]))
            self._sched_table.setItem(r, 1, QTableWidgetItem(s["start_time"]))
            self._sched_table.setItem(r, 2, QTableWidgetItem(s["end_time"]))

    def _on_add_schedule(self) -> None:
        if not self._current_class_id: return
        dlg = _AddScheduleDialog(parent=self)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            day, start, end = dlg.values()
            if db.add_class_schedule(self._current_class_id, day, start, end) is None:
                QMessageBox.information(self, "Duplicate", "A schedule slot for that day/time already exists.")
            self._refresh_schedule(self._current_class_id)

    def _on_remove_schedule(self) -> None:
        if not self._current_class_id: return
        row = self._sched_table.currentRow()

        if 0 <= row < len(self._sched_ids):
            db.remove_class_schedule(self._sched_ids[row])
            self._refresh_schedule(self._current_class_id)

    def _refresh_students(self, class_id: int) -> None:
        enrolled_ids = db.get_class_student_ids(class_id)
        profile_map = {p["id"]: p for p in db.load_all_profiles()}
        self._stu_ids = list(enrolled_ids)
        self._stu_table.setRowCount(0)

        sorted_ids = sorted(self._stu_ids, key=lambda i: profile_map.get(i, {}).get("full_name", ""))
        for r, pid in enumerate(sorted_ids):
            p = profile_map.get(pid, {})
            self._stu_table.insertRow(r)
            self._stu_table.setItem(r, 0, QTableWidgetItem(p.get("full_name", f"id:{pid}")))
            self._stu_table.setItem(r, 1, QTableWidgetItem(p.get("cnp", "")))

    def _on_enroll_student(self) -> None:
        if not self._current_class_id: return

        enrolled_ids = db.get_class_student_ids(self._current_class_id)
        all_profiles = db.load_all_profiles()
        dlg = _EnrollStudentDialog(all_profiles, enrolled_ids, parent=self)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            for pid in dlg.selected_ids():
                db.enroll_student_in_class(self._current_class_id, pid)
            self._refresh_students(self._current_class_id)

    def _on_remove_student(self) -> None:
        if not self._current_class_id: return
        row = self._stu_table.currentRow()

        if 0 <= row < len(self._stu_ids):
            name = self._stu_table.item(row, 0).text()
            ans = QMessageBox.question(self, "Remove Student", f"Remove '{name}' from this class?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans == QMessageBox.StandardButton.Yes:
                db.unenroll_student_from_class(self._current_class_id, self._stu_ids[row])
                self._refresh_students(self._current_class_id)

    def _on_refresh_report(self) -> None:
        if self._current_class_id:
            self._refresh_attendance_report(self._current_class_id)

    def _refresh_attendance_report(self, class_id: int) -> None:
        self._report_table.setRowCount(0)
        try:
            summary = db.get_class_attendance_summary(class_id)
            sessions_held = db.get_class_session_count(class_id)
            enrolled_ids = db.get_class_student_ids(class_id)
            profile_map = {p["id"]: p for p in db.load_all_profiles()}
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        present_map = {row["profile_id"]: row["present_count"] for row in summary}
        sorted_ids = sorted(enrolled_ids, key=lambda i: profile_map.get(i, {}).get("full_name", ""))

        for r, pid in enumerate(sorted_ids):
            p = profile_map.get(pid, {})
            name = p.get("full_name", f"id:{pid}")
            present = present_map.get(pid, 0)
            absent = max(0, sessions_held - present)
            pct = f"{100 * present / sessions_held:.0f}%" if sessions_held > 0 else "-"

            self._report_table.insertRow(r)
            self._report_table.setItem(r, 0, QTableWidgetItem(name))
            self._report_table.setItem(r, 1, QTableWidgetItem(str(sessions_held)))

            pi = QTableWidgetItem(str(present))
            pi.setForeground(_GREEN if present > 0 else QColor(200, 200, 200))
            self._report_table.setItem(r, 2, pi)

            ai = QTableWidgetItem(str(absent))
            ai.setForeground(_RED if absent > 0 else QColor(200, 200, 200))
            self._report_table.setItem(r, 3, ai)

            pct_i = QTableWidgetItem(pct)
            if sessions_held > 0:
                ratio = present / sessions_held
                pct_i.setForeground(_GREEN if ratio >= 0.75 else (_AMBER if ratio >= 0.5 else _RED))
            self._report_table.setItem(r, 4, pct_i)


class AttendanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video = VideoWidget()

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background: #161616;")
        side_layout = QVBoxLayout(sidebar)

        title = QLabel("Recent Check-ins")
        title.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 13px;")
        side_layout.addWidget(title)

        self.recent_list = QListWidget()
        self.recent_list.setStyleSheet("border: none; QListWidget::item { padding: 4px 2px; border-bottom: 1px solid #222; }")
        side_layout.addWidget(self.recent_list)

        self.today_label = QLabel("Today: 0 people")
        self.today_label.setStyleSheet("color: #aaa; font-size: 12px;")
        side_layout.addWidget(self.today_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout =  QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._today_count = 0

    def update_frame(self, frame, tracks):
        self.video.update_frame(frame, tracks)

    def show_camera_message(self, text):
        self.video.show_message(text)

    def add_checkin(self, name, time_str):
        item = QListWidgetItem(f"{time_str} {name}")
        item.setForeground(_GREEN)
        self.recent_list.insertItem(0, item)

        while self.recent_list.count() > 50:
            self.recent_list.takeItem(self.recent_list.count() - 1)

        self._today_count += 1
        self.today_label.setText(f"Today: {self._today_count} people")

    def refresh_today_count(self):
        self._today_count = db.count_today_attendance()
        self.today_label.setText(f"Today: {self._today_count} people")


class AdminTab(QWidget):
    enrolled = pyqtSignal(dict)
    SUPPORTED_FORMATS = "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_embedding = None
        self._pending_format = None
        self._pending_warnings = []

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.preview = QLabel("No document uploaded yet. \nClick 'Upload Document' to begin.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #0d0d0d; color: #555;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.preview, stretch=1)

        self.upload_btn = QPushButton("Upload Document")
        self.upload_btn.setFixedHeight(38)
        self.upload_btn.clicked.connect(self._on_upload)
        left_layout.addWidget(self.upload_btn)

        self.file_label = QLabel("")
        self.file_label.setStyleSheet("color: #555; font-size: 11px;")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.file_label)

        # Right panel
        right_box = QGroupBox("Extracted Profile")
        form = QFormLayout(right_box)

        self.cnp_field = QLineEdit()
        self.name_field = QLineEdit()
        self.dob_field = QLineEdit()
        self.format_field = QLineEdit()
        self.format_field.setReadOnly(True)
        self.warnings_field = QTextEdit()
        self.warnings_field.setReadOnly(True)
        self.warnings_field.setMinimumHeight(80)

        form.addRow("CNP:", self.cnp_field)
        form.addRow("Full Name:", self.name_field)
        form.addRow("Birth Date:", self.dob_field)
        form.addRow("Format:", self.format_field)
        form.addRow("Warnings:", self.warnings_field)

        btn_row = QHBoxLayout()

        self.enroll_btn = QPushButton("Enroll")
        self.enroll_btn.setEnabled(False)
        self.enroll_btn.clicked.connect(self._on_enroll)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.clicked.connect(self._on_clear)

        btn_row.addWidget(self.enroll_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()

        right = QWidget()
        right_vbox = QVBoxLayout(right)
        right_vbox.addWidget(right_box, stretch=1)
        right_vbox.addLayout(btn_row)

        self.status_label = QLabel("Upload a scanned ID card image to begin.")
        self.status_label.setStyleSheet("color: #aaa;")

        #Main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.status_label)

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ID Card Image", "", self.SUPPORTED_FORMATS)
        if not path: return

        self.upload_btn.setEnabled(False)
        self.enroll_btn.setEnabled(False)
        self.file_label.setText(Path(path).name)
        self._set_status(f"Processing {Path(path).name} ...")

        try:
            raw = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            frame = None

        if frame is None:
            self._set_status(f"Could not read image: {path}", error=True)
            self.upload_btn.setEnabled(True)
            return

        self._show_preview(frame)

        #Run the heavy scanner in bckground so the UI doesn't freeze
        import threading
        threading.Thread(target=self._run_scan, args=(frame,), daemon=True).start()

    def _run_scan(self, frame):
        try:
            from scanner.scanner import scan_image
            self._scan_result = scan_image(frame)
        except Exception as e:
            self._scan_result = None
            self._scan_error = str(e)

        from PyQt6.QtCore import QMetaObject, Qt as Q2
        QMetaObject.invokeMethod(self, "_apply_scan_result", Q2.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _apply_scan_result(self):
        self.upload_btn.setEnabled(True)
        result = getattr(self, "_scan_result", None)

        if result is None:
            self._set_status(f"Scan error: {getattr(self, '_scan_error', 'unknown')}", error=True)
            return

        self.cnp_field.setText(result.cnp or "")
        self.name_field.setText(result.full_name or "")
        self.dob_field.setText(result.birth_date or "")
        self.format_field.setText(result.card_format or "")

        self._pending_format = result.card_format
        self._pending_warnings = list(result.warnings or [])
        self._pending_embedding = result.embedding

        warn = list(self._pending_warnings)
        if result.embedding is None:
            warn.append("No face embedding - person will not be recognised by camera")

        self.warnings_field.setPlainText("\n".join(warn) if warn else "None")
        self.warnings_field.setStyleSheet("color: #ffa726;" if warn else "color: #66bb6a;")
        self.enroll_btn.setEnabled(True)

        if not result.cnp:
            self._set_status("CNP not found - fill it in manually, then click Enroll.", error=True)
        elif not result.success:
            self._set_status("Partial extraction - check the fields, then click Enroll.")
        else:
            self._set_status("Extraction complete - review the fields then click Enroll.")


    def _on_enroll(self):
        cnp = self.cnp_field.text().strip()
        name = self.name_field.text().strip()
        dob = self.dob_field.text().strip()

        if not cnp: return self._set_status("CNP is required.", error=True)
        if not name: return self._set_status("Full name is required.", error=True)

        profile = {
            "cnp": cnp,
            "full_name": name,
            "birth_date": dob,
            "embedding": self._pending_embedding,
            "card_format": self._pending_format,
            "warnings": self._pending_warnings
        }

        try:
            ok = db.enroll_profile(profile)
        except Exception as e:
            return self._set_status(f"Enroll error: {e}", error=True)

        if ok:
            msg = f"{name} enrolled successfully." + (" (no face embedding)" if not self._pending_embedding else "")
            self._set_status(msg)
            self.enrolled.emit(profile)
            self._on_clear()
        else:
            self._set_status("Enroll failed", error=True)

    def _on_clear(self):
        for f in [self.cnp_field, self.name_field, self.dob_field, self.format_field]:
            f.clear()
        self.warnings_field.clear()
        self._pending_embedding = self._pending_format = None
        self._pending_warnings = []

        self.enroll_btn.setEnabled(False)
        self.file_label.setText("")
        self.preview.setText("No document uploaded yet.\nClick 'Upload Document' to begin.")
        self._set_status("Cleared.")

    def _show_preview(self, frame):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.preview.setPixmap(pixmap)

    def _set_status(self, text, error=False):
        col = "#ef5350" if error else "#aaa"
        self.status_label.setStyleSheet(f"color: {col}; font-size: 12px;")
        self.status_label.setText(text)

class LogTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.date_from = QDateEdit(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")

        self.date_to = QDateEdit(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")

        sb = QPushButton("Search")
        sb.setFixedWidth(90)
        sb.clicked.connect(self.refresh)

        bar.addWidget(QLabel("From:"))
        bar.addWidget(self.date_from)
        bar.addWidget(QLabel("To:"))
        bar.addWidget(self.date_to)
        bar.addWidget(sb)
        bar.addStretch()
        layout.addLayout(bar)

        self.table = _build_readonly_table(["Timestamp", "Name", "Class"], stretch_col=1)
        layout.addWidget(self.table)

        self.row_count_label = QLabel("0 records")
        self.row_count_label.setStyleSheet("color: #666;")
        layout.addWidget(self.row_count_label)

        self.refresh()

    def refresh(self):
        d_from = self.date_from.date().toString("yyyy-MM-dd")
        d_to = self.date_to.date().toString("yyyy-MM-dd")

        try:
            rows = db.query_attendance_with_class(d_from, d_to)
        except Exception as e:
            QMessageBox.warning(self, "Query Error", str(e))
            return

        self.table.setRowCount(0)
        for r, (ts, name, cls_str) in enumerate(rows):
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(ts)))
            self.table.setItem(r, 1, QTableWidgetItem(str(name)))
            self.table.setItem(r, 2, QTableWidgetItem(str(cls_str)))

        self.row_count_label.setText(f"{self.table.rowCount()} records")


class TeacherAccountsTab(QWidget):
    def __init__(self, current_teacher_id: int, parent=None):
        super().__init__(parent)
        self._current_teacher_id = current_teacher_id

        layout = QVBoxLayout(self)

        hdr = QLabel("Teacher Accounts")
        hdr.setStyleSheet("font-size: 15px; font-weight: bold; color: #00e5ff;")
        layout.addWidget(hdr)

        self._table = _build_readonly_table(["Username", "Full Name", "Created"], stretch_col=1)
        layout.addWidget(self._table)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Teacher")
        add_btn.clicked.connect(self._on_add)

        del_btn = QPushButton("Delete")
        del_btn.setObjectName("dangerBtn")
        del_btn.clicked.connect(self._on_delete)

        pw_btn = QPushButton("Change Password")
        pw_btn.setObjectName("secondaryBtn")
        pw_btn.clicked.connect(self._on_change_password)

        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        btns.addWidget(pw_btn)
        btns.addStretch()
        layout.addLayout(btns)

        self._teacher_ids = []
        self._refresh()

    def _refresh(self) -> None:
        teachers = db.list_teachers()
        self._teacher_ids = [t["id"] for t in teachers]
        self._table.setRowCount(0)

        for r, t in enumerate(teachers):
            self._table.insertRow(r)
            self._table.setItem(r, 0, QTableWidgetItem(t["username"]))
            self._table.setItem(r, 1, QTableWidgetItem(t["full_name"]))
            self._table.setItem(r, 2, QTableWidgetItem(t["created_at"][:10]))

            if t["id"] == self._current_teacher_id:
                for c in range(3):
                    self._table.item(r, c).setForeground(_CYAN)


    def _on_add(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("New Teacher Account")
        dlg.setMinimumWidth(320)

        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        u_edit = QLineEdit()
        u_edit.setPlaceholderText("Username")
        fn_edit = QLineEdit()
        fn_edit.setPlaceholderText("Full name")
        pw_edit = QLineEdit()
        pw_edit.setPlaceholderText("Password")
        pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        pw2_edit = QLineEdit()
        pw2_edit.setPlaceholderText("Confirm password")
        pw2_edit.setEchoMode(QLineEdit.EchoMode.Password)

        form.addRow("Username:", u_edit)
        form.addRow("Full name:", fn_edit)
        form.addRow("Password:", pw_edit)
        form.addRow("Confirm:", pw2_edit)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        def _accept():
            if not u_edit.text().strip() or not fn_edit.text().strip():
                return QMessageBox.warning(dlg, "Required", "Username and full name required.")
            if not pw_edit.text():
                return QMessageBox.warning(dlg, "Required", "Password is required.")
            if pw_edit.text() != pw2_edit.text():
                return QMessageBox.warning(dlg, "Mismatch", "Passwords do no match.")
            dlg.accept()

        btns.accepted.connect(_accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            if db.create_teacher(u_edit.text().strip(), pw_edit.text(), fn_edit.text().strip()):
                self._refresh()
            else:
                QMessageBox.warning(self, "Error", "Username already taken.")


    def _on_delete(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._teacher_ids):
            tid = self._teacher_ids[row]
            if tid == self._current_teacher_id:
                return QMessageBox.warning(self, "Cannot Delete", "You cannot delete your own account while logged in.")

            name = self._table.item(row, 1).text()
            ans = QMessageBox.question(
                self, "Delete Teacher",
                f"Delete teacher account '{name}'?\nThis does NOT delete their classes or attendance data.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ans == QMessageBox.StandardButton.Yes:
                db.delete_teacher(tid)
                self._refresh()

    def _on_change_password(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._teacher_ids):
            name = self._table.item(row, 1).text()
            new_pw, ok = QInputDialog.getText(self, "Change Password", f"New password for {name:}", QLineEdit.EchoMode.Password)
            if ok and new_pw.strip():
                db.change_teacher_password(self._teacher_ids[row], new_pw.strip())
                QMessageBox.information(self, "Done", "Password updated.")


# Student UI

class StudentWindow(QMainWindow):
    logout_requested = pyqtSignal()

    def __init__(self, profile: dict, parent=None):
        super().__init__(parent)
        self._profile_id = profile["id"]
        self.setWindowTitle("My Attendance")
        self.setMinimumSize(800, 560)

        icon_path = Path(__file__).parent / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header
        hrow = QHBoxLayout()
        greet = QLabel(f"Welcome, {profile.get('full_name', 'Student')}")
        greet.setStyleSheet("font-size: 18px; font-weight: bold; color: #00e5ff;")
        hrow.addWidget(greet)
        hrow.addStretch()
        hrow.addWidget(QLabel(f"CNP: {profile.get('cnp', '')}"))

        lo_btn = QPushButton("Log Out")
        lo_btn.setObjectName("secondaryBtn")
        lo_btn.clicked.connect(self._on_logout)
        hrow.addWidget(lo_btn)
        main_layout.addLayout(hrow)

        #Summary Table
        sum_box = QGroupBox("My Classes")
        sum_lay = QVBoxLayout(sum_box)
        self._summary_table = _build_readonly_table(["Class", "Total Sessions", "Present", "Absent", "Attendance %"], stretch_col=0)
        self._summary_table.setMaximumHeight(220)
        self._summary_table.currentCellChanged.connect(self._on_class_selected)
        sum_lay.addWidget(self._summary_table)

        ref_btn = QPushButton("Refresh")
        ref_btn.setObjectName("secondaryBtn")
        ref_btn.setFixedWidth(100)
        ref_btn.clicked.connect(self._refresh_summary)
        sum_lay.addWidget(ref_btn, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(sum_box)

        # Detail Table
        self._detail_box = QGroupBox("Session Detail - select a class above")
        det_lay = QVBoxLayout(self._detail_box)
        self._detail_table = _build_readonly_table(["Date", "Start", "End", "Status"], stretch_col=3)
        det_lay.addWidget(self._detail_table)
        main_layout.addWidget(self._detail_box)

        self._class_ids = []
        self._refresh_summary()

    def _refresh_summary(self):
        try:
            summary = db.get_student_summary(self._profile_id)
        except Exception as e:
            return QMessageBox.warning(self, "Error", str(e))

        self._class_ids = [s["class_id"] for s in summary]
        self._summary_table.setRowCount(0)

        for r, s in enumerate(summary):
            self._summary_table.insertRow(r)
            total, present = s["total_sessions"], s["present_count"]
            absent = total - present
            pct = f"{100 * present / total:.0f}%" if total > 0 else "-"

            self._summary_table.setItem(r, 0, QTableWidgetItem(s["class_name"]))
            self._summary_table.setItem(r, 1, QTableWidgetItem(str(total)))

            pi = QTableWidgetItem((str(present)))
            pi.setForeground(_GREEN if present > 0 else QColor(200, 200, 200))
            self._summary_table.setItem(r, 2, pi)

            ai = QTableWidgetItem(str(absent))
            ai.setForeground(_RED if absent > 0 else QColor(200, 200, 200))
            self._summary_table.setItem(r, 3, ai)

            pct_i = QTableWidgetItem(pct)
            if total > 0:
                pct_i.setForeground(_GREEN if (present/total) >= 0.75 else (_AMBER if (present/total) >= 0.5 else _RED))
            self._summary_table.setItem(r, 4, pct_i)

    def _on_class_selected(self, row, *_):
        if 0 <= row < len(self._class_ids):
            cid = self._class_ids[row]
            cls_name = self._summary_table.item(row, 0).text()
            self._detail_box.setTitle(f"Session Detail - {cls_name}")

            try:
                sessions = db.get_student_attendance_for_class(self._profile_id, cid)
            except Exception as e:
                return QMessageBox.warning(self, "Error", str(e))

            self._detail_table.setRowCount(0)
            for r, s in enumerate(sessions):
                self._detail_table.insertRow(r)
                self._detail_table.setItem(r, 0, QTableWidgetItem(s["session_date"]))
                self._detail_table.setItem(r, 1, QTableWidgetItem(s["start_time"]))
                self._detail_table.setItem(r, 2, QTableWidgetItem(s["end_time"]))

                status = QTableWidgetItem("Present" if s["present"] else "Absent")
                status.setForeground(_GREEN if s["present"] else _RED)
                self._detail_table.setItem(r, 3, status)

    def _on_logout(self):
        self.hide()
        self.logout_requested.emit()
        self.close()


# Main Window Controllers

class MainWindow(QMainWindow):
    logout_requested = pyqtSignal()

    def __init__(self, teacher: dict, engine: AttendanceEngine, parent=None):
        super().__init__(parent)
        self._engine = engine

        name = teacher.get("full_name", teacher.get("username", ""))
        self.setWindowTitle(f"Attendance System - {name}")
        self.setMinimumSize(1100, 700)

        icon_path = Path(__file__).parent / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._build_ui(teacher)
        self._build_status_bar()
        self._connect_engine()

        self.admin_tab.enrolled.connect(self._on_enrolled)
        self.attendance_tab.refresh_today_count()

    def _build_ui(self, teacher):
        self.attendance_tab = AttendanceTab()
        self.admin_tab = AdminTab()
        self.log_tab = LogTab()
        self.classes_tab = ClassesTab(teacher_id=teacher["id"])
        self.accounts_tab = TeacherAccountsTab(current_teacher_id=teacher["id"])

        self._tabs = QTabWidget()
        self._tabs.addTab(self.attendance_tab, "Live Attendance")
        self._tabs.addTab(self.admin_tab, "Admin Enroll")
        self._tabs.addTab(self.log_tab, "Attendance Log")
        self._tabs.addTab(self.classes_tab, "Classes")
        self._tabs.addTab(self.accounts_tab, "Accounts")

        self._tabs.currentChanged.connect(lambda i: self.log_tab.refresh() if i == 2 else None)
        self.setCentralWidget(self._tabs)

    def _build_status_bar(self):
        bar = QStatusBar()

        self._fps_label = QLabel("FPS: —")
        self._fps_label.setStyleSheet("color: #4fc3f7")

        self._cam_label = QLabel("Camera: OK")
        self._cam_label.setStyleSheet("color: #66bb6a;")

        mod_ready = self._engine.embedder.available
        self._model_label = QLabel("Model: ready" if mod_ready else "Model: NOT LOADED")
        self._model_label.setStyleSheet("color: #66bb6a;" if mod_ready else "color: #ffa726;")

        self._roster_label = QLabel(f"Roster: {len(self._engine.known_names)} recognisable / {db.count_profiles()} enrolled")


        lo_btn = QPushButton("Logout")
        lo_btn.setObjectName("secondaryBtn")
        lo_btn.setFixedHeight(24)
        lo_btn.clicked.connect(self._on_logout)

        bar.addWidget(self._fps_label)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._cam_label)
        bar.addPermanentWidget(self._roster_label)
        bar.addPermanentWidget(QLabel(" | "))
        bar.addPermanentWidget(self._model_label)
        bar.addPermanentWidget(QLabel(" | "))
        bar.addPermanentWidget(lo_btn)

        self.setStatusBar(bar)

    def _connect_engine(self):
        e = self._engine
        e.frame_with_tracks.connect(self.attendance_tab.update_frame)
        e.checkin_logged.connect(self.attendance_tab.add_checkin)
        e.fps_updated.connect(lambda fps: self._fps_label.setText(f"FPS: {fps:.1f}"))
        e.camera_lost.connect(self._on_camera_lost)
        e.camera_restored.connect(lambda: self._cam_label.setText("Camera: OK") or self._cam_label.setStyleSheet("color: #66bb6a;"))
        e.roster_changed.connect(lambda r, t: self._roster_label.setText(f"Roster: {r} recognisable / {t} enrolled"))


    def _disconnect_engine(self):
        e = self._engine
        for signal in [e.frame_with_tracks, e.checkin_logged, e.fps_updated, e.camera_lost, e.camera_restored, e.roster_changed]:
            try: signal.disconnect()
            except Exception: pass


    def _on_camera_lost(self):
        self._cam_label.setText("Camera: LOST")
        self._cam_label.setStyleSheet("color: #ef5350;")
        self.attendance_tab.show_camera_message("Camera lost - check connection")

    @pyqtSlot(dict)
    def _on_enrolled(self, profile: dict):
        self._engine.add_profile(profile)
        self.log_tab.refresh()

    def _on_logout(self):
        self._disconnect_engine()
        self.hide()
        self.logout_requested.emit()
        self.close()

    def closeEvent(self, event):
        # we let it kill the background engine, not the Window
        event.accept()

class AppController(QObject):
    def __init__(self, app):
        super().__init__()
        self._app = app
        self._window = None
        self._engine = AttendanceEngine()
        self.show_login()

    def show_login(self):
        if self._window is not None:
            try:
                self._window.logout_requested.disconnect()
            except Exception:
                pass
            self._window.hide()
            self._window = None

        dlg = LoginDialog()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._engine.stop()
            self._app.quit()
            return

        role, data = dlg.result_data()
        if role == "teacher":
            self._window = MainWindow(teacher=data, engine=self._engine)
        elif role == "student":
            self._window = StudentWindow(profile=data)

        self._window.logout_requested.connect(self.show_login)
        self._window.show()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Attendance System")
    app.setStyleSheet(_DARK_QSS)
    controller = AppController(app)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

