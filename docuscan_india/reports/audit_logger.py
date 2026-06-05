import os
import sqlite3
import hashlib
from datetime import datetime
import json
from typing import List, Dict, Any
from utils.document_packet import DocumentPacket
from utils.logger import get_logger

logger = get_logger("audit_logger")

class AuditLogger:
    def __init__(self, db_dir: str = None):
        """Initializes SQLite database and tables."""
        if db_dir is None:
            # Set default path: data/db/audit.db relative to the app directory
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_dir = os.path.join(app_dir, "data", "db")
        else:
            self.db_dir = db_dir

        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "audit.db")
        self._init_db()

    def _init_db(self):
        """Creates the audit log table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    verdict TEXT NOT NULL,
                    extracted_fields TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"Audit database initialized successfully at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")

    def _compute_sha256(self, file_path: str) -> str:
        """Computes the SHA-256 hash of a file."""
        if not os.path.exists(file_path):
            return "UNKNOWN"
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            return "HASH_ERROR"

    def log(self, packet: DocumentPacket) -> bool:
        """Persists a DocumentPacket's audit record to the SQLite database."""
        try:
            # 1. Compute Hash
            doc_hash = self._compute_sha256(packet.image_path)
            
            # 2. Determine Verdict
            # If there's any validation FAIL or Critical Risk, the verdict is REJECTED/FAIL
            verdict = "PASS"
            if packet.fraud_risk_score >= 50:
                verdict = "FAIL"
            else:
                for res in packet.validation_results:
                    if res.status == "FAIL":
                        verdict = "FAIL"
                        break
                    elif res.status == "WARN":
                        verdict = "WARN"

            # 3. Serialize extracted fields (simplified key-value for database queries)
            fields_dict = {k: v.value for k, v in packet.extracted_fields.items()}
            fields_json = json.dumps(fields_dict)

            # 4. Insert log
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (
                    timestamp, document_hash, file_path, document_type, 
                    risk_score, verdict, extracted_fields, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                doc_hash,
                packet.image_path,
                packet.document_type.value,
                packet.fraud_risk_score,
                verdict,
                fields_json,
                "1.0.0"  # Pipeline version
            ))
            conn.commit()
            conn.close()
            logger.info(f"Audit log saved successfully. Hash: {doc_hash[:10]}... Verdict: {verdict}")
            return True
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            return False

    def fetch_all(self) -> List[Dict[str, Any]]:
        """Utility to fetch all logs, useful for reporting / UI statistics."""
        logs = []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC")
            rows = cursor.fetchall()
            for r in rows:
                logs.append(dict(r))
            conn.close()
        except Exception as e:
            logger.error(f"Failed to fetch audit logs: {e}")
        return logs
