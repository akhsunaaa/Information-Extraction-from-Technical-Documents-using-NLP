# Information Extraction from Technical Documents using NLP

**Course:** Natural Language Processing (PBL)  
**Student Name:** Anushka Jain 
**PRN:** 1032220621
**Date:** 13 April 2026  

## 🧬 Overview

This project is a production-ready Web Application that parses complex pharmaceutical technical documents (like ASTM or EPA standards) and extracts structured information using semantic NLP techniques. 

It implements a hybrid NLP architecture, utilizing Zero-Shot sequence classification (`sentence-transformers`) for document segmentation and an elite Generative AI engine (`Google Gemini 2.5 Flash`) to natively distill recursive laboratory methodologies into highly structured relational databases.

### Key Features
* **Dual-Engine Natural Language Processing:** Bypasses destructive text chunking by passing full documents sequentially into a Trillion-parameter LLM, mathematically forced to output into strict JSON topologies regardless of methodology variation.
* **Semantic Classifier & Embeddings:** Employs the `all-MiniLM-L6-v2` Sentence-BERT model to semantically classify section blocks via high-dimensional vector embeddings and Cosine Similarity.
* **Custom Horizontal State Navigation:** Eradicates the native Streamlit UI limits via CSS payloads, offering a custom floating pink-gradient top-bar navigated dynamically.
* **Secure Enterprise Connection:** Safe UI-mounted connection portal bridging the Python execution core with Google's Cloud Intelligence API without risking hard-coded credentials.

## 🚀 Installation Instructions

### 1. Create a Virtual Environment

It's highly recommended to use a standard Python Virtual Environment (`venv`):

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is activated, locate the root folder containing `requirements.txt` and install the modules:

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy Language Model

The Fallback NLP Engine relies on spaCy's English core ruleset:

```bash
python -m spacy download en_core_web_sm
```

## 🖥 Usage Instructions

Run the main Streamlit application locally:

```bash
streamlit run app.py
```
This will automatically spool up the local HTTP server and initialize the `pharmaceutical_nlp.db` SQLite database.

### Navigating the Framework UI
- **Global Extractor Orbit:** Main dashboard tracking live intelligence scaling, engine telemetry, and API connection.
- **Upload Document:** Provide standard laboratory procedure PDFs (e.g., `method_353-2.pdf`) for instantaneous chunking and semantic extraction.
- **Tests:** Cross-reference extracted documents to securely isolated analytical sub-routines and required chemical ratios.
- **Documents:** Relational database tree containing metadata on uploaded methodology standards.
- **Inventory:** Master aggregate catalog mapping every uniquely identified chemical and piece of apparatus processing logic across the system history.

## 🖼 System Snapshots

- `dashboard.jpg`: The customized Global Extraction UI demonstrating the embedded pink top-bar navigation and LLM telemetry.
- `extraction_results.jpg`: Semantic GenAI cleanly segregating complex data blocks from a dense laboratory specification sheet.
- `test_selector.jpg`: The SQLite cross-referencing capabilities separating distinct procedures and reagents mathematically.
