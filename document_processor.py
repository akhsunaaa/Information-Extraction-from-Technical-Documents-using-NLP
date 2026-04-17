import re
import hashlib
import fitz  # PyMuPDF
import pdfplumber

class ASTMProcessor:
    """
    Handles file ingestion, text extraction using PyMuPDF (with pdfplumber fallback),
    and fast regex-based document type detection.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath

    def _compute_hash(self):
        """Computes SHA-256 hash of the file content for deduplication."""
        hasher = hashlib.sha256()
        try:
            with open(self.filepath, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Hashing error: {e}")
            return None

    def read_text_pymupdf(self):
        """Fast extraction using PyMuPDF."""
        text = ""
        page_count = 0
        try:
            doc = fitz.open(self.filepath)
            page_count = doc.page_count
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            text = f"PyMuPDF error: {e}"
        return text, page_count

    def read_text_pdfplumber(self):
        """Fallback extraction using pdfplumber."""
        text = ""
        page_count = 0
        try:
            with pdfplumber.open(self.filepath) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            text = f"pdfplumber error: {e}"
        return text, page_count

    def extract_text(self):
        """
        Main extraction method. Uses PyMuPDF first, falls back to pdfplumber 
        if the extracted text is too short or fails entirely.
        """
        success = False
        filename = getattr(self.filepath, 'name', str(self.filepath).split('/')[-1])
        file_hash = self._compute_hash()
        
        # If it's a plain text file, just read it directly
        if str(self.filepath).endswith('.txt'):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {
                    "success": True, 
                    "text": text, 
                    "page_count": 1, 
                    "file_hash": file_hash, 
                    "filename": filename
                }
            except Exception as e:
                return {"success": False, "text": str(e), "page_count": 0, "file_hash": file_hash, "filename": filename}
        
        # Default to PDF parsing
        text, page_count = self.read_text_pymupdf()
        
        # If text is too short, PyMuPDF might have failed structurally. Try pdfplumber.
        if len(text.strip()) < 100:
            fallback_text, page_count = self.read_text_pdfplumber()
            if len(fallback_text.strip()) > len(text.strip()):
                text = fallback_text

        if len(text.strip()) > 50:
            success = True
            
        return {
            "success": success,
            "text": text.strip(),
            "page_count": page_count,
            "file_hash": file_hash,
            "filename": filename
        }

    def detect_document_type(self, text):
        """
        Uses standard regex definitions to classify the document type.
        Focuses on ASTM and ISO standard headers often found in the beginning page.
        """
        text_start = text[:2000].upper() # Usually standard is declared early
        
        astm_pattern = re.compile(r'ASTM\s+[A-Z]\d+(-\d+)?')
        iso_pattern = re.compile(r'ISO\s+\d+(:\d+)?')
        bs_pattern = re.compile(r'BS\s+EN\s+\d+')
        ep_pattern = re.compile(r'EUROPEAN\s+PHARMACOPOEIA|PH\.\s*EUR\.')
        usp_pattern = re.compile(r'USP\s*\d+|UNITED\s*STATES\s*PHARMACOPEIA')

        if astm_pattern.search(text_start):
            return "ASTM Standard"
        elif iso_pattern.search(text_start):
            return "ISO Standard"
        elif bs_pattern.search(text_start):
            return "British Standard"
        elif ep_pattern.search(text_start):
            return "European Pharmacopoeia (Ph. Eur.)"
        elif usp_pattern.search(text_start):
            return "United States Pharmacopeia (USP)"
        else:
            return "General Lab Procedure"
