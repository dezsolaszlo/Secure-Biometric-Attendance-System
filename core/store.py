import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from cryptography.fernet import Fernet

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
KEY_FILE = os.path.join(CURRENT_DIR, "profiles.key")
DATABASE = os.path.join(CURRENT_DIR, "profiles.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cnp_hash        TEXT    UNIQUE  NOT NULL,
    cnp_enc         BLOB    NOT NULL,
    name_enc        BLOB,
    birth_date      TEXT,
    card_format     TEXT,
    embedding_enc   BLOB,
    source_file     TEXT,
    enrolled_at     TEXT    NOT NULL,
    warnings        TEXT
);
"""

SCHEMA_INBOX = """
CREATE TABLE IF NOT EXISTS inbox(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cnp_hash    TEXT NOT NULL
);
"""

def _get_encryption_key(path: str) -> bytes:
    if os.path.exists(path):
        with open(path, "rb") as file:
            return file.read().strip()

    fresh_key = Fernet.generate_key()
    with open(path, "wb") as file:
        file.write(fresh_key)

    print("New encryption key was generated!")
    return fresh_key


class ProfileStore:

    def __init__(self, db_path: str = DATABASE, key_path: str = KEY_FILE):
        self._fernet = Fernet(_get_encryption_key(key_path))
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(SCHEMA)
        self._db.execute(SCHEMA_INBOX)
        self._db.commit()

    def _encrypt(self, value: str | None) -> bytes | None:
        if not value:
            return None

        raw_bytes = str(value).encode()

        return self._fernet.encrypt(raw_bytes)

    def _decrypt(self, encrypted: bytes | None) -> str | None:
        if not encrypted:
            return None
        token = bytes(encrypted)
        return self._fernet.decrypt(token).decode()

    def _hash(self, cnp: str) -> str:
        return hashlib.sha256(cnp.encode()).hexdigest()



    def _unpack_row(self, row: sqlite3.Row) -> dict:
        plaintext_cnp = self._decrypt(row["cnp_enc"])
        plaintext_name = self._decrypt(row["name_enc"])

        embedding = self._decrypt(row["embedding_enc"])
        if embedding:
            embedding_list = json.loads(embedding)
        else:
            embedding_list = None

        warnings = row["warnings"]
        if warnings:
            warning_list = json.loads(warnings)
        else:
            warning_list = []

        return {
            "id": row["id"],
            "cnp": plaintext_cnp,
            "full_name": plaintext_name,
            "birth_date": row["birth_date"],
            "card_format": row["card_format"],
            "embedding": embedding_list,
            "source_file": row["source_file"],
            "enrolled_at": row["enrolled_at"],
            "warnings": warning_list,
        }

    def is_enrolled(self, cnp: str) -> bool:
        target_hash = self._hash(cnp)

        query = "SELECT 1 FROM profiles WHERE cnp_hash = ?"
        mathcing_row = self._db.execute(query, (target_hash,)).fetchone()

        if mathcing_row:
            return True
        else:
            return False



    def enroll(self, profile_data: dict, overwrite: bool = False) -> bool:
        cnp = profile_data.get("cnp")
        if not cnp:
            return False

        embedding = profile_data.get("embedding")
        if embedding:
            packed_embedding = json.dumps(embedding)
        else:
            packed_embedding = None

        enrollment_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        warnings_list = profile_data.get("warnings") or []

        db_payload = (
            self._hash(cnp),
            self._encrypt(cnp),
            self._encrypt(profile_data.get("full_name")),
            profile_data.get("birth_date"),
            profile_data.get("card_format"),
            self._encrypt(packed_embedding),
            profile_data.get("source_file"),
            enrollment_time,
            json.dumps(warnings_list),
        )

        try:
            if overwrite:
                query = """
                    INSERT INTO profiles VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cnp_hash) DO UPDATE SET
                        cnp_enc=excluded.cnp_enc,
                        name_enc=excluded.name_enc,
                        birth_date=excluded.birth_date,
                        card_format=excluded.card_format,
                        embedding_enc=excluded.embedding_enc,
                        source_file=excluded.source_file,
                        enrolled_at=excluded.enrolled_at,
                        warnings=excluded.warnings
                """

                self._db.execute(query, db_payload)
            else:
                query = "INSERT INTO profiles VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                self._db.execute(query, db_payload)

            target_hash = self._hash(cnp)
            queue_query = "INSERT INTO inbox (cnp_hash) VALUES (?)"
            self._db.execute(queue_query, (target_hash,))

            self._db.commit()
            return True

        except sqlite3.IntegrityError:
            return False


    def drain_inbox(self) -> list[dict]:
        query = """
            SELECT p.* FROM profiles p
            INNER JOIN inbox i ON p.cnp_hash = i.cnp_hash
        """
        encrypted_rows = self._db.execute(query).fetchall()

        flush_query = "DELETE FROM inbox"
        self._db.execute(flush_query)
        self._db.commit()

        newly_enrolled_profiles = []
        for row in encrypted_rows:
            profile = self._unpack_row(row)
            newly_enrolled_profiles.append(profile)

        return newly_enrolled_profiles


    def lookup(self, cnp: str) -> dict | None:
        target_hash = self._hash(cnp)

        query = "SELECT * FROM profiles WHERE cnp_hash = ?"
        db_row = self._db.execute(query, (target_hash,)).fetchone()

        if db_row:
            return self._unpack_row(db_row)
        else:
            return None

    def list_all(self) -> list[dict]:
        query = "SELECT * FROM profiles ORDER BY enrolled_at"
        encrypted_rows = self._db.execute(query).fetchall()

        decrypted_roster = []
        for row in encrypted_rows:
            clean_profile = self._unpack_row(row)
            decrypted_roster.append(clean_profile)

        return decrypted_roster

    def delete(self, cnp: str) -> bool:
        target_hash = self._hash(cnp)
        query = "DELETE FROM profiles WHERE cnp_hash = ?"
        cursor = self._db.execute(query, (target_hash,))
        self._db.commit()

        return cursor.rowcount > 0

    def count(self) -> int:
        query = "SELECT COUNT(*) FROM profiles"
        result = self._db.execute(query).fetchone()

        total_profiles = result[0]
        return total_profiles

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()