import sqlite3
import os
import json
from typing import Dict, Optional
from app.config import BASE_DIR


DB_PATH = os.path.join(BASE_DIR, "database", "audit_copilot.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entreprises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raison_sociale TEXT NOT NULL,
            ice TEXT UNIQUE,
            rc TEXT,
            if_val TEXT,
            activite TEXT,
            dirigeants TEXT,
            capital REAL,
            adresse TEXT,
            exercice TEXT,
            banques TEXT,
            fiscalite TEXT,
            forme_juridique TEXT,
            date_creation TEXT,
            secteur TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        migrations = [
            "ALTER TABLE entreprises ADD COLUMN if_val TEXT",
            "ALTER TABLE entreprises ADD COLUMN activite TEXT",
            "ALTER TABLE entreprises ADD COLUMN fiscalite TEXT",
            "ALTER TABLE entreprises ADD COLUMN date_creation TEXT",
            "ALTER TABLE entreprises ADD COLUMN secteur TEXT",
            "ALTER TABLE entreprises ADD COLUMN capital REAL",
        ]
        for migration in migrations:
            try:
                cursor.execute(migration)
            except sqlite3.OperationalError:
                pass  

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entreprise_id INTEGER,
            type TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entreprise_id) REFERENCES entreprises(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entreprise_id INTEGER,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entreprise_id) REFERENCES entreprises(id)
        )
        """)

        conn.commit()


def save_entreprise(data: Dict) -> int:
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        ice = data.get("ice", "").strip() or None

        if ice:
            cursor.execute("SELECT id FROM entreprises WHERE ice = ?", (ice,))
            existing = cursor.fetchone()
            if existing:
                return existing[0]

        cursor.execute("""
        INSERT INTO entreprises
        (raison_sociale, ice, rc, if_val, activite, dirigeants, capital,
         adresse, exercice, banques, fiscalite, forme_juridique, date_creation, secteur)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("raison_sociale", ""),
            ice,
            data.get("rc", ""),
            data.get("if_val", ""),
            data.get("activite", ""),
            json.dumps(data.get("dirigeants", [])),
            data.get("capital", 0),
            data.get("adresse", ""),
            data.get("exercice", ""),
            json.dumps(data.get("banques", [])),
            data.get("fiscalite", ""),
            data.get("forme_juridique", ""),
            data.get("date_creation", ""),
            data.get("secteur", "")
        ))
        conn.commit()
        return cursor.lastrowid


def get_entreprise_by_ice(ice: str) -> Optional[Dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM entreprises WHERE ice = ?", (ice,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "raison_sociale": row[1],
                "ice": row[2],
                "rc": row[3],
                "if_val": row[4],
                "activite": row[5],
                "dirigeants": json.loads(row[6]) if row[6] else [],
                "capital": row[7],
                "adresse": row[8],
                "exercice": row[9],
                "banques": json.loads(row[10]) if row[10] else [],
                "fiscalite": row[11],
                "forme_juridique": row[12],
                "date_creation": row[13],
                "secteur": row[14]
            }
        return None


def save_document(entreprise_id: int, doc_type: str, path: str) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (entreprise_id, type, path) VALUES (?, ?, ?)",
            (entreprise_id, doc_type, path)
        )
        conn.commit()
        return cursor.lastrowid


def save_analyse(entreprise_id: int, analyse_type: str, content: Dict) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analyses (entreprise_id, type, content) VALUES (?, ?, ?)",
            (entreprise_id, analyse_type, json.dumps(content))
        )
        conn.commit()
        return cursor.lastrowid