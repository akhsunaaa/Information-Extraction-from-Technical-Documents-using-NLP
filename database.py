import sqlite3
import os
from contextlib import contextmanager

DB_PATH = "pharmaceutical_nlp.db"

class DatabaseManager:
    """
    Manages the SQLite database connection and operations for the NLP Extractor.
    Implements a context manager for safe database connections.
    """
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_database()
        
    @contextmanager
    def get_connection(self):
        """Context manager to ensure connections are closed safely."""
        conn = sqlite3.connect(self.db_path)
        # Enable foreign key support in SQLite
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _initialize_database(self):
        """Creates tables if they do not exist with appropriate schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Chemicals Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chemicals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    cas_number TEXT,
                    hazard_level TEXT,
                    molecular_formula TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Documents Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT,
                    error_message TEXT
                )
            ''')
            
            # Tests Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    target_substance TEXT,
                    standard_id TEXT,
                    description TEXT,
                    document_id INTEGER,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            ''')
            
            # Requirements Table (Apparatus, Reagents)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS requirements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    quantity_string TEXT,
                    unit TEXT,
                    ratio_per_unit REAL,
                    critical_flag BOOLEAN DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
                )
            ''')
            
            # Procedures Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS procedures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    step_number INTEGER,
                    instruction_text TEXT NOT NULL,
                    substeps TEXT,
                    estimated_duration INTEGER,
                    FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
                )
            ''')
            
            # Test-Chemicals Mapping (Junction Table)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_chemicals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    chemical_id INTEGER NOT NULL,
                    role TEXT,
                    FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE,
                    FOREIGN KEY (chemical_id) REFERENCES chemicals(id) ON DELETE CASCADE
                )
            ''')
            
            # Preset C: Confidence Scores Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS confidence_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
                )
            ''')
            
            # Preset C: Processing Logs Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    processing_time_sec REAL,
                    llm_used BOOLEAN DEFAULT 0,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            ''')
            
            # View for test summaries
            cursor.execute('''
                CREATE VIEW IF NOT EXISTS vw_test_summary AS
                SELECT 
                    t.id, t.name, t.target_substance,
                    COUNT(DISTINCT r.id) as requirement_count,
                    COUNT(DISTINCT p.id) as step_count,
                    COUNT(DISTINCT tc.chemical_id) as chemical_count
                FROM tests t
                LEFT JOIN requirements r ON t.id = r.test_id
                LEFT JOIN procedures p ON t.id = p.test_id
                LEFT JOIN test_chemicals tc ON t.id = tc.test_id
                GROUP BY t.id
            ''')
            
            # Creating indexes for faster lookup
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tests_name ON tests(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chem_name ON chemicals(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_req_testid ON requirements(test_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_proc_testid ON procedures(test_id)')

    # --- CRUD Operations ---
    
    def add_document(self, filename, file_hash, file_size, status="PROCESSING"):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents (filename, file_hash, file_size, processing_status) 
                    VALUES (?, ?, ?, ?)
                ''', (filename, file_hash, file_size, status))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # File already exists
            return self.get_document_by_hash(file_hash)[0]
            
    def get_document_by_hash(self, file_hash):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, filename, processing_status FROM documents WHERE file_hash = ?', (file_hash,))
            return cursor.fetchone()

    def update_document_status(self, doc_id, status, error=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE documents SET processing_status = ?, error_message = ? WHERE id = ?
            ''', (status, error, doc_id))

    def add_chemical(self, name, cas=None):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO chemicals (name, cas_number) VALUES (?, ?)', (name, cas))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # If chemical exists, return its ID
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM chemicals WHERE name = ?', (name,))
                return cursor.fetchone()[0]

    def add_test(self, name, document_id, description="", target_substance="Unknown"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tests (name, target_substance, document_id, description) 
                VALUES (?, ?, ?, ?)
            ''', (name, target_substance, document_id, description))
            return cursor.lastrowid

    def add_requirement(self, test_id, req_type, name, quantity_string="", unit="", ratio=1.0):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO requirements (test_id, type, name, quantity_string, unit, ratio_per_unit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (test_id, req_type, name, quantity_string, unit, ratio))

    def add_procedure_step(self, test_id, step_number, instruction):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO procedures (test_id, step_number, instruction_text)
                VALUES (?, ?, ?)
            ''', (test_id, step_number, instruction))

    def link_test_chemical(self, test_id, chemical_id, role="reagent"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO test_chemicals (test_id, chemical_id, role)
                VALUES (?, ?, ?)
            ''', (test_id, chemical_id, role))

    # --- Fetch Operations ---

    def get_all_chemicals(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM chemicals ORDER BY name')
            return cursor.fetchall()
            
    def get_all_apparatuses(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT name FROM requirements WHERE type="Apparatus" ORDER BY name')
            return cursor.fetchall()
            
    def get_all_substances(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT target_substance FROM tests WHERE target_substance IS NOT NULL')
            return [row[0] for row in cursor.fetchall()]

    def get_tests_by_substance(self, substance):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, description FROM tests WHERE target_substance = ?', (substance,))
            return cursor.fetchall()

    def get_test_details(self, test_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Fetch requirements
            cursor.execute('SELECT type, name, quantity_string, unit, ratio_per_unit FROM requirements WHERE test_id = ?', (test_id,))
            requirements = cursor.fetchall()
            # Fetch procedures
            cursor.execute('SELECT step_number, instruction_text FROM procedures WHERE test_id = ? ORDER BY id', (test_id,))
            procedures = cursor.fetchall()
            return {"requirements": requirements, "procedures": procedures}

    def get_all_tests(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM tests ORDER BY name')
            return cursor.fetchall()

    def get_all_documents(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, filename, upload_date, processing_status, file_size FROM documents ORDER BY upload_date DESC')
            return cursor.fetchall()

    def delete_document(self, doc_id):
        """Will cascade delete all tests, requirements, etc. associated with this doc."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

    def get_statistics(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            cursor.execute('SELECT COUNT(*) FROM documents')
            stats['documents'] = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM tests')
            stats['tests'] = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM chemicals')
            stats['chemicals'] = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM requirements WHERE type="Apparatus"')
            stats['equipment'] = cursor.fetchone()[0]
            return stats
