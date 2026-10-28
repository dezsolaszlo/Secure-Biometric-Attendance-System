import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

from .store import ProfileStore

_HERE = Path(__file__).parent
DB_PATH = _HERE.parent / 'profiles.db'
KEY_PATH = _HERE.parent / 'profiles.key'

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DUP_MINS = 5

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles (id),
    checked_in_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_att_profile ON attendance(profile_id);
CREATE INDEX IF NOT EXISTS idx_att_time ON attendance(checked_in_at);

CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL REFERENCES teachers (id),
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS class_schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes (id) ON DELETE CASCADE,
    day_of_week INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    UNIQUE (class_id, day_of_week, start_time)
);

CREATE TABLE IF NOT EXISTS class_enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes (id) ON DELETE CASCADE,
    profile_id INTEGER NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    enrolled_at TEXT NOT NULL,
    UNIQUE (class_id, profile_id)
);

CREATE TABLE IF NOT EXISTS class_attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes (id) ON DELETE CASCADE,
    profile_id INTEGER NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    session_date TEXT NOT NULL,
    marked_at TEXT NOT NULL,
    UNIQUE (class_id, profile_id, session_date)
);
CREATE INDEX IF NOT EXISTS idx_ca_class ON class_attendance(class_id);
CREATE INDEX IF NOT EXISTS idx_ca_profile ON class_attendance(profile_id);
CREATE INDEX IF NOT EXISTS idx_ca_date ON class_attendance(session_date);
"""

def _init_db(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    #WAL mode = simultaneous read/write
    #foreign_keys=ON: makes ON DELETE CASCADE work
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)


@contextmanager
def db_session():
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def store_session():
    store = ProfileStore(db_path=str(DB_PATH), key_path=str(KEY_PATH))
    _init_db(store._db)
    try:
        yield store
    finally:
        store.close()

#Password

def _hash_password(pw: str) -> str:
     salt = secrets.token_hex(16)
     h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 260000)
     return f"{salt}:{h.hex()}"

def _verify_pass(pw: str, stored: str) -> bool:
    try:
        salt, h_hex = stored.split(":", 1)
        h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 260000)
        return secrets.compare_digest(h_hex, h.hex())
    except Exception:
        return False


#Teachers

def count_teachers() -> int:
    with db_session() as db:
        return db.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]

def create_teacher(username: str, password: str, fullname: str) -> bool:
    try:
        with db_session() as db:
            db.execute(
                "INSERT INTO teachers (username, password_hash, full_name, created_at) VALUES (?, ?, ?, ?)",
                (username.strip(), _hash_password(password), fullname.strip(), _now())
            )
            db.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def authenticate_teacher(username: str, password: str) -> dict | None:
    with db_session() as db:
        row = db.execute("SELECT * FROM teachers WHERE username = ?", (username.strip(),)).fetchone()
        return dict(row) if row and _verify_pass(password, row["password_hash"]) else None

def list_teachers() -> list[dict]:
    with db_session() as db:
        return [dict(r) for r in db.execute(
            "SELECT id, username, full_name, created_at FROM teachers ORDER BY full_name"
        )]

def delete_teacher(teacher_id: int) -> bool:
    with db_session() as db:
        cur = db.execute("DELETE FROM teachers WHERE id = ?", (teacher_id,))
        db.commit()
        return cur.rowcount > 0

def change_teacher_password(teacher_id: int, new_pass: str) -> bool:
    with db_session() as db:
        cur = db.execute(
            "UPDATE teachers SET password_hash = ? WHERE id = ?",
            (_hash_password(new_pass), teacher_id),
        )
        db.commit()
        return cur.rowcount > 0

#Classes

def create_class(teacher_id: int, name: str, desc: str = "") -> int:
    with db_session() as db:
        cur = db.execute(
            "INSERT INTO classes (teacher_id, name, description, created_at) VALUES (?,?,?,?)",
            (teacher_id, name.strip(), desc.strip(), _now())
        )
        db.commit()
        return cur.lastrowid

def update_class(class_id: int, name: str, desc: str) -> bool:
    with db_session() as db:
        cur = db.execute(
            "UPDATE classes SET name = ?, description = ? WHERE id = ?",
            (name.strip(), desc.strip(), class_id),
        )
        db.commit()
        return cur.rowcount > 0

def delete_class(class_id: int) -> bool:
    with db_session() as db:
        cur = db.execute(
            "DELETE FROM classes WHERE id = ?",
            (class_id,)
        )
        db.commit()
        return cur.rowcount > 0

def list_classes(teacher_id: int | None = None) -> list[dict]:
    with db_session() as db:
        if teacher_id is not None:
            return [dict(r) for r in db.execute(
                "SELECT * FROM classes WHERE teacher_id = ? ORDER BY name",
                (teacher_id,)
            )]
        return [dict(r) for r in db.execute("SELECT * FROM classes ORDER BY name")]

def get_class(class_id: int) -> dict | None:
    with db_session() as db:
        row = db.execute("SELECT * FROM classes WHERE id = ?", (class_id,)).fetchone()
        return dict(row) if row else None

#Schedules

def add_class_schedule(class_id: int, day_of_week: int, start_time: str, end_time:str) -> int | None:
    with db_session() as db:
        cur = db.execute(
            "INSERT OR IGNORE INTO class_schedules (class_id, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?)",
            (class_id, day_of_week, start_time, end_time),
        )
        db.commit()
        return cur.lastrowid if cur.rowcount > 0 else None

def remove_class_schedule(schedule_int: int) -> bool:
    with db_session() as db:
        cur = db.execute("DELETE FROM class_schedules WHERE id = ?", (schedule_int,))
        db.commit()
        return cur.rowcount > 0

def get_class_schedule(class_id: int) -> list[dict]:
    with db_session() as db:
        return[dict(r) for r in db.execute(
            "SELECT * FROM class_schedules WHERE class_id = ? ORDER BY day_of_week, start_time",
            (class_id,)
        )]

def get_active_classes(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now()
    time_str = now.strftime("%H:%M")
    with db_session() as db:
        return [dict(r) for r in db.execute(
            """
            SELECT c.*, s.id AS schedule_id, s.day_of_week, s.start_time, s.end_time
            FROM class_schedules s JOIN classes c ON c.id = s.class_id
            WHERE s.day_of_week = ? AND s.start_time <= ? AND s.end_time > ?
            """,
            (now.weekday(), time_str, time_str),
        )]

# Enrollments

def enroll_student_in_class(class_id: int, profile_id: int) -> bool:
    try:
        with db_session() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO class_enrollments (class_id, profile_id, enrolled_at) VALUES (?, ?, ?)",
                (class_id, profile_id, _now()),
            )
            db.commit()
            return True
    except Exception:
        return False

def unenroll_student_from_class(class_id: int, profile_id: int) -> bool:
    with db_session() as db:
        cur = db.execute(
            "DELETE FROM class_enrollments WHERE class_id = ? AND profile_id = ?",
            (class_id, profile_id)
        )
        db.commit()
        return cur.rowcount > 0

def get_class_student_ids(class_id: int) -> set[int]:
    with db_session() as db:
        return {r["profile_id"] for r in db.execute(
            "SELECT profile_id FROM class_enrollments WHERE class_id = ?",
            (class_id,)
        )}

def get_student_classes(profile_id: int) -> list[dict]:
    with db_session() as db:
        return [dict(r) for r in db.execute(
            "SELECT c.* FROM class_enrollments e JOIN classes c ON c.id = e.class_id WHERE e.profile_id = ? ORDER BY c.name",
            (profile_id,)
        )]

# Class attendance

def log_class_attendance(class_id: int, profile_id: int, session_date: str | None = None) -> bool:
    session_date = session_date or date.today().isoformat()
    with db_session() as db:
        cur = db.execute(
            "INSERT OR IGNORE INTO class_attendance (class_id, profile_id, session_date, marked_at) VALUES (?, ?, ?, ?)",
            (class_id, profile_id, session_date, _now()),
        )
        db.commit()
        return cur.rowcount > 0

def get_expected_session_dates(class_id: int, up_to: date | None = None) -> list[str]:
    up_to = up_to or date.today()
    with db_session() as db:
        cls = db.execute("SELECT created_at FROM classes WHERE id = ?", (class_id,)).fetchone()
        if not cls: return []

        days = {r["day_of_week"] for r in db.execute(
            "SELECT day_of_week FROM class_schedules WHERE class_id = ?", (class_id,)
        )}
        if not days: return []

        created = date.fromisoformat(cls["created_at"][:10])

    result, current = [], created
    while current <= up_to:
        if current.weekday() in days:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result

def get_class_attendance_report(class_id: int) -> list[dict]:
    with store_session() as session:
        name = {p["id"]: p.get("full_name", "Unknown") for p in session.list_all()}
        present = {(r["profile_id"], r["session_date"]) for r in session._db.execute(
            "SELECT profile_id, session_date FROM class_attendance WHERE class_id = ?",
            (class_id,)
        )}

    expected_dates = get_expected_session_dates(class_id)
    enrolled_ids = get_class_student_ids(class_id)

    report = [
        {
            "profile_id": pid,
            "full_name": name.get(pid, f"id:{pid}"),
            "session_date": sd,
            "present": 1 if (pid, sd) in present else 0,
        }
        for sd in reversed(expected_dates)
        for pid in enrolled_ids
    ]
    report.sort(key=lambda r: r["full_name"])
    report.sort(key=lambda r: r["session_date"], reverse=True)
    return report

def get_student_attendance_for_class(profile_id: int, class_id: int) -> list[dict]:
    with db_session() as db:
        schedule = {r["day_of_week"]: (r["start_time"], r["end_time"]) for r in db.execute(
            "SELECT day_of_week, start_time, end_time FROM class_schedules WHERE class_id = ?",
            (class_id,)
        )}
        present_dates = {r["session_date"] for r in db.execute(
            "SELECT session_date FROM class_attendance WHERE class_id = ? AND profile_id = ?",
            (class_id,profile_id),
        )}

        result = []
        for sd in reversed(get_expected_session_dates(class_id)):
            start, end = schedule.get(date.fromisoformat(sd).weekday(), ("?","?"))
            result.append({"session_date": sd, "present": 1 if sd in present_dates else 0,
                           "start_time": start, "end_time": end})

        return result

def get_student_summary(profile_id: int) -> list[dict]:
    with db_session() as db:
        classes = db.execute(
            "SELECT c.id, c.name FROM class_enrollments e JOIN classes c on c.id = e.class_id WHERE e.profile_id = ? ORDER BY c.name",
            (profile_id,)
        ).fetchall()

        return [
            {
                "class_id": c["id"],
                "class_name": c["name"],
                "total_sessions": len(get_expected_session_dates(c["id"])),
                "present_count": db.execute(
                    "SELECT COUNT(*) FROM class_attendance WHERE class_id = ? AND profile_id = ?",
                    (c["id"],profile_id),
                ).fetchone()[0],
            }
            for c in classes
        ]


def get_class_attendance_summary(class_id: int) -> list[dict]:
    with db_session() as db:
        return [dict(r) for r in db.execute(
            "SELECT profile_id, COUNT(*) AS present_count FROM class_attendance WHERE class_id = ? GROUP BY profile_id",
            (class_id,)
        )]

def get_class_session_count(class_id: int) -> int:
    return len(get_expected_session_dates(class_id))

# Profiles

def enroll_profile(profile_data: dict, overwrite: bool = True) -> bool:
    with store_session() as store: return store.enroll(profile_data, overwrite=overwrite)

def load_all_profiles() -> list[dict]:
    with store_session() as store: return store.list_all()

def drain_new_profiles() -> list[dict]:
    with store_session() as store: return store.drain_inbox()

def lookup_profile_by_cnp(cnp: str) -> dict | None:
    with store_session() as store: return store.lookup(cnp)

def is_enrolled(cnp: str) -> bool:
    with store_session() as store: return store.is_enrolled(cnp)

def count_profiles() -> int:
    with store_session() as store: return store.count()

def delete_profile(cnp: str) -> bool:
    with store_session() as store: return store.delete(cnp)

# General attandence

def log_attendance(profile_id: int) -> bool:
    with store_session() as store:
        cur = store._db.execute(
            """
            INSERT INTO attendance (profile_id, checked_in_at)
            SELECT ?, datetime('now')
            WHERE NOT EXISTS (
                SELECT 1 FROM attendance
                WHERE profile_id = ? AND checked_in_at >= datetime('now', ?)
            )
            """,
            (profile_id, profile_id, f"-{_DUP_MINS} minutes"),
        )
        store._db.commit()
        return cur.rowcount > 0

def _query_attendance(date_from: str, date_to: str, with_class: bool = False) -> list[tuple]:
    with store_session() as store:
        names = {p["id"]: p.get("full_name", "Unknown") for p in store.list_all()}
        rows = store._db.execute(
            "SELECT profile_id, checked_in_at FROM attendance WHERE checked_in_at BETWEEN ? AND ? ORDER BY checked_in_at DESC",
            (date_from, date_to + " 23:59:59"),
        ).fetchall()

        result = []
        for row in rows:
            pid, ts = row["profile_id"], row["checked_in_at"]
            name = names.get(pid, f"id:{pid}")

            if with_class:
                cls_rows = store._db.execute(
                    "SELECT c.name FROM class_attendance ca JOIN classes c ON c.id = ca.class_id WHERE ca.profile_id = ? AND ca.session_date = ? ORDER BY c.name",
                    (pid, ts[:10]),
                ).fetchall()
                result.append((ts, name, ", ".join(r["name"] for r in cls_rows) or "-"))
            else:
                result.append((ts, name))
        return result

def query_attendance(date_from: str, date_to: str) -> list[tuple]:
    return _query_attendance(date_from, date_to)

def query_attendance_with_class(date_from: str, date_to: str) -> list[tuple]:
    return _query_attendance(date_from, date_to, with_class=True)

def count_today_attendance() -> int:
    with store_session() as store:
        return store._db.execute(
            "SELECT COUNT(DISTINCT profile_id) FROM attendance WHERE date(checked_in_at) = date('now')"
        ).fetchone()[0]

def recent_attendance(limit: int = 20) -> list:
    return _query_attendance("2000-01-01", datetime.now().strftime("%Y-%m-%d"))[:limit]