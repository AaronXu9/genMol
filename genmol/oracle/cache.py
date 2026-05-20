"""SQLite cache for oracle evaluations.

Key: (smiles_canonical, receptor_sha1, oracle_name, oracle_version).
Value: score + metadata JSON + timestamp.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


class OracleCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS oracle_cache (
                smiles TEXT, receptor_hash TEXT,
                oracle_name TEXT, oracle_version TEXT,
                score REAL, metadata TEXT, ts INTEGER,
                PRIMARY KEY (smiles, receptor_hash, oracle_name, oracle_version)
            )"""
        )
        self._conn.commit()

    def get(
        self, smiles: str, receptor_hash: str, oracle_name: str, oracle_version: str
    ) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT score, metadata FROM oracle_cache WHERE smiles=? AND receptor_hash=? "
            "AND oracle_name=? AND oracle_version=?",
            (smiles, receptor_hash, oracle_name, oracle_version),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {"score": row[0], "metadata": json.loads(row[1]) if row[1] else {}}

    def put(
        self,
        smiles: str,
        receptor_hash: str,
        oracle_name: str,
        oracle_version: str,
        score: float,
        metadata: dict,
    ):
        import time

        self._conn.execute(
            "INSERT OR REPLACE INTO oracle_cache VALUES (?,?,?,?,?,?,?)",
            (
                smiles,
                receptor_hash,
                oracle_name,
                oracle_version,
                float(score),
                json.dumps(metadata or {}),
                int(time.time()),
            ),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
