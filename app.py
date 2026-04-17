import streamlit as st
import pandas as pd
import os
import time

from database import DatabaseManager
from document_processor import ASTMProcessor
from semantic_classifier import SemanticSectionDetector
from nlp_extractor import TechnicalEntityExtractor
from rag_assistant import RagAssistant

from dotenv import load_dotenv
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Valence Pharma NLP", page_icon="💊", layout="wide", initial_sidebar_state="collapsed")

# Handle Logo Click Hard-Reset
if "reset" in st.query_params:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.query_params.clear()
    import time; time.sleep(0.1) # Debounce
    st.rerun()

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #ffe4f0, #ffffff);
}

body::after {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    opacity: 0.2;
}

    /* Main container styling */
    .reportview-container .main .block-container{
        padding-top: 5rem;
        padding-bottom: 5rem;
    }
    
    /* Elevate Streamlit's native collapse control... actually the user wants it DISABLED permanently! */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Custom Floating Top Navigation */
    .custom-topbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 70px;
    z-index: 999999 !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 40px;

    background: linear-gradient(135deg, #ff7eb3, #ff4da6, #ff007f);
    box-shadow: 0 6px 25px rgba(255, 0, 127, 0.4);
    border-bottom: 1px solid rgba(255,255,255,0.2);

    backdrop-filter: blur(12px);
    }

    .custom-topbar .logo h1 {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #ffe3f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 15px rgba(255,255,255,0.5);
}

    .custom-topbar .llm-status {
    color: #ffffff;
    background: rgba(255,255,255,0.2);
    padding: 8px 18px;
    border-radius: 30px;
    font-weight: 700;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.3);
    display: flex;           /* 🔥 THIS fixes it */
    align-items: center;     /* vertical alignment */
    gap: 8px;                /* spacing between "LLM" and dot */
}

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;   /* ensures proper sizing */
    }
    .status-green {
    background-color: #48bb78;
    box-shadow: 0 0 8px #48bb78;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 5px #48bb78; }
    50% { box-shadow: 0 0 15px #48bb78; }
    100% { box-shadow: 0 0 5px #48bb78; }
}

    .status-red {
        background-color: #f56565;
        box-shadow: 0 0 10px #f56565;
    }
    
    /* Sleek gradient background for sidebar making it a floating island */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Ensure the backdrop also disconnects from the edge */
    [data-testid="stSidebar"] > div:first-child {
        display: none !important;
    }

    /* Horizontal Nav Links */
    .nav-links {
        display: flex;
        gap: 20px;
        align-items: center;
        pointer-events: auto;
    }

    .nav-links a {
    text-decoration: none;
    color: #ffffff;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    padding: 8px 16px;
    border-radius: 20px;
    transition: all 0.25s ease;
}

    .nav-links a:hover {
    background: rgba(255,255,255,0.25);
    box-shadow: 0 4px 15px rgba(255,255,255,0.3);
}

.nav-links a.active {
    background: white;
    color: #ff007f !important;
    font-weight: 700;
    box-shadow: 0 4px 20px rgba(255,255,255,0.5);
}

    /* Metric card UI */
    .metric-card {
    background: linear-gradient(135deg, #ffffff, #ffe6f2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;

    box-shadow: 0 10px 25px rgba(255, 0, 127, 0.15);
    border: 1px solid rgba(255, 0, 127, 0.1);

    transition: all 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 15px 35px rgba(255, 0, 127, 0.25);
}
    
    /* Typography improvements */
    h1, h2, h3 {
    color: #d1006b !important;
    letter-spacing: -0.3px;
}
    
    /* Footer Styling */
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(135deg, #ff7eb3, #ff4da6);
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 0.9em;
    border-top: 1px solid var(--faded-text-10);
    z-index: 100;
    backdrop-filter: blur(5px);
    box-shadow: 0 -4px 15px rgba(255, 0, 127, 0.2);
}
    
    /* Header Container */
    .global-header {
        text-align: center;
        padding-bottom: 20px;
        margin-bottom: 20px;
        border-bottom: 2px solid #eaeaea;
    }
    .global-header h1 {
        margin: 0;
        font-size: 2.5rem;
        color: #2c3e50;
    }
    .global-header p {
        color: #7f8c8d;
        margin-top: 5px;
    }

    .stButton button {
    background: linear-gradient(135deg, #ff4da6, #ff007f);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4);
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 0, 127, 0.6);
}

/* Inputs & Selectboxes */
.stTextInput input, 
.stSelectbox div[data-baseweb="select"],
.stFileUploader {
    border-radius: 12px !important;
    border: 1px solid #ffd1e8 !important;
    background: #fff0f6 !important;
    color: #4a0033 !important;
    box-shadow: 0 2px 8px rgba(255, 0, 127, 0.1);
}

/* Focus glow */
.stTextInput input:focus {
    border: 1px solid #ff4da6 !important;
    box-shadow: 0 0 10px rgba(255, 0, 127, 0.3);
}

/* Tables */
.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #ffd1e8;
    box-shadow: 0 8px 20px rgba(255, 0, 127, 0.1);
}

/* Header */
.stDataFrame thead {
    background: linear-gradient(135deg, #ff7eb3, #ff4da6);
    color: white;
}

/* Expander */
details {
    background: #fff0f6;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #ffd1e8;
    margin-bottom: 10px;
}

/* Smooth transitions everywhere */
* {
    transition: all 0.2s ease-in-out;
}


</style>
""", unsafe_allow_html=True)

# --- Initialization & State ---
@st.cache_resource
def init_db():
    db = DatabaseManager()
    # Inject demonstration data if DB is empty
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tests")
        if cursor.fetchone()[0] == 0:
            # Create a mock document
            doc_id = db.add_document("mock_standard.pdf", "mockhash123", 1024, "COMPLETED")
            
            # Chemicals
            chem1 = db.add_chemical("Acetonitrile", "75-05-8")
            chem2 = db.add_chemical("Water for Injection", "7732-18-5")
            chem3 = db.add_chemical("Phosphoric Acid", "7664-38-2")
            
            # Test 1
            t1 = db.add_test("High-Performance Liquid Chromatography (HPLC) Assay", doc_id, "Determines concentration of active ingredient.")
            db.link_test_chemical(t1, chem1, "Mobile Phase")
            db.link_test_chemical(t1, chem2, "Solvent")
            db.add_requirement(t1, "Reagent", "Acetonitrile", "500 mL", "mL", 500)
            db.add_requirement(t1, "Apparatus", "HPLC Module", "1 unit", "unit", 1)
            db.add_procedure_step(t1, 1, "Prepare the mobile phase by mixing 500 mL Acetonitrile and 500 mL Water.")
            db.add_procedure_step(t1, 2, "Set flow rate to 1.0 mL/min.")
            
            # Test 2
            t2 = db.add_test("Dissolution Testing", doc_id, "Evaluates the rate of drug release.")
            db.link_test_chemical(t2, chem2, "Dissolution Medium")
            db.add_requirement(t2, "Reagent", "Water for Injection", "900 mL", "mL", 900)
            db.add_procedure_step(t2, 1, "Place 900 mL water into each dissolution vessel.")
            db.add_procedure_step(t2, 2, "Maintain temperature at 37°C.")
            
            # Test 3
            t3 = db.add_test("Content Uniformity", doc_id, "Ensures consistent active ingredient amounts across batches.")
            db.link_test_chemical(t3, chem3, "Buffer")
            db.add_requirement(t3, "Reagent", "Phosphoric Acid", "5 mL", "mL", 5)
            db.add_procedure_step(t3, 1, "Randomly select 10 units.")
            db.add_procedure_step(t3, 2, "Assay each unit individually using the specified volumetric buffer.")
            
    return db

@st.cache_resource
def load_models():
    # Load heavy NLP models only once
    extractor = TechnicalEntityExtractor()
    classifier = SemanticSectionDetector()
    rag = RagAssistant()
    return extractor, classifier, rag

db = init_db()

# Provide a loading state for models
with st.spinner("Loading NLP Models (spaCy, Sentence-Transformers, Llama)..."):
    try:
        extractor, classifier, rag = load_models()
        models_loaded = True
        llm_loaded = rag.is_healthy()
    except Exception as e:
        st.error(f"Error loading models: {e}\nPlease ensure you run: python -m spacy download en_core_web_sm")
        models_loaded = False
        llm_loaded = False

# Handle Navigation Link Clicks
if "nav" in st.query_params:
    nav = st.query_params["nav"]
    if nav == "upload": st.session_state.page = "Upload Document"
    elif nav == "test": st.session_state.page = "Test Selector"
    elif nav == "manager": st.session_state.page = "Document Manager"
    elif nav == "inventory": st.session_state.page = "Inventory"
    elif nav == "about": st.session_state.page = "About Me"
    
    st.query_params.clear()
    import time; time.sleep(0.1)
    st.rerun()

page = st.session_state.get("page", None)

# Dynamic LLM Engine Monitor 
if rag.is_healthy():
    engine_name = getattr(rag, 'nlp_engine', "Offline")
else:
    engine_name = "Offline"

# Inject the fixed floating navbar
dot_color = "status-green" if llm_loaded else "status-red"
st.markdown(f"""
<div class="custom-topbar">
    <a href="/?reset=1" target="_self" style="text-decoration: none;">
        <div class="logo">
            <h1>Valence</h1>
        </div>
    </a>
    <div class="nav-links">
        <a href="/?nav=upload" class="{'active' if page=='Upload Document' else ''}" target="_self">Upload</a>
        <a href="/?nav=test" class="{'active' if page=='Test Selector' else ''}" target="_self">Tests</a>
        <a href="/?nav=manager" class="{'active' if page=='Document Manager' else ''}" target="_self">Documents</a>
        <a href="/?nav=inventory" class="{'active' if page=='Inventory' else ''}" target="_self">Inventory</a>
        <a href="/?nav=about" class="{'active' if page=='About Me' else ''}" target="_self">About</a>
    </div>
    <div class="llm-status" title="Active Engine: {engine_name}">
    <span>LLM</span>
    <span class="status-dot {dot_color}"></span>
</div>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_ci(d, key, default=None):
    """Case-insensitive dictionary getter to handle LLMs that mutate key capitalization."""
    if not isinstance(d, dict): return default
    for k, v in d.items():
        if k.lower() == key.lower():
            return v
    return default

def extract_and_save_data(text, doc_filename, doc_hash, file_size):
    doc_id = db.add_document(doc_filename, doc_hash, file_size, status="PROCESSING")
    
    # 1. Semantic Segmentation
    sections = classifier.segment_document(text)
    
    reagents_text = "\n".join(sections.get("Reagents", []))
    apparatus_text = "\n".join(sections.get("Apparatus", []))
    procedure_text = "\n".join(sections.get("Procedure", []))
    scope_text = "\n".join(sections.get("Scope", []))
    
    # NEW PRESET C: Ask Llama/Gemini to natively parse the FULL unfiltered document
    if rag.is_healthy():
        engine_str = getattr(rag, 'nlp_engine', "Neural Engine")
        with st.spinner(f"{engine_str} is reading the document and extracting entities..."):
            # Pass the raw, unfiltered text entirely to the LLM. 
            # The legacy semantic distillator was destructively truncating the QC and Reagents sections.
            extracted_json = rag.parse_document_entities(text)
            
            if extracted_json:
                substance = get_ci(extracted_json, "TargetSubstance", "Unknown")
                tests_array = get_ci(extracted_json, "Tests", [])
                
                if not tests_array:
                    db.update_document_status(doc_id, "ERROR - NO TESTS EXTRACTED")
                    return sections, [], [], []
                
                extracted_reagents = []
                extracted_apparatus = []
                extracted_procedures = []
                
                for test_data in tests_array:
                    test_name = get_ci(test_data, "TestName", f"Standard Assessment - {doc_filename}")
                    
                    # 1. Add Test with Target Substance
                    test_id = db.add_test(test_name, doc_id, "Extracted by LLM Intelligence", target_substance=substance)
                    
                    # 2. Add Reagents
                    for r in get_ci(test_data, "Reagents", []):
                        c_name = get_ci(r, "name", "Unknown Chemical")
                        qty = get_ci(r, "quantity", "Unknown")
                        cid = db.add_chemical(c_name)
                        db.link_test_chemical(test_id, cid, "Reagent")
                        db.add_requirement(test_id, "Reagent", c_name, qty)
                        extracted_reagents.append(r)
                        
                    # 3. Add Apparatus
                    for eq in get_ci(test_data, "Apparatus", []):
                        db.add_requirement(test_id, "Apparatus", eq, "1 unit")
                        extracted_apparatus.append(eq)
                        
                    # 4. Add Translated Procedures
                    for i, p in enumerate(get_ci(test_data, "Procedures", [])):
                        db.add_procedure_step(test_id, i + 1, p)
                        extracted_procedures.append(p)
                
                db.update_document_status(doc_id, "COMPLETED")
                return sections, extracted_reagents, extracted_apparatus, extracted_procedures
            else:
                db.update_document_status(doc_id, "ERROR - LLM FAILED")
                return sections, [], [], []
    else:
        st.error("LLM Engine Offline. No extraction performed.")
        return sections, [], [], []
    
    return sections, [], [], []

# --- Pages ---

if page is None:
    st.markdown("<h1 style='margin-top: 20px;'>📊 Global Extractor Orbit</h1>", unsafe_allow_html=True)
    st.write("Welcome to Valence. A fully autonomous NLP intelligence framework processing standard methodologies.")
    
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>{stats['documents']}</h3><p>Documents</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>{stats['tests']}</h3><p>Tests Identified</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>{stats['chemicals']}</h3><p>Unique Chemicals</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>{stats['equipment']}</h3><p>Equipment Pieces</p></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader(f"🧠 Active Neural Engine: {engine_name}")
    
    if not os.environ.get("GEMINI_API_KEY"):
        st.markdown("**⚡ Boost to Enterprise Speed**")
        st.markdown("<span style='font-size: 0.9em; color: gray;'>Local Llama feeling slow? Instantly plug in Google's trillion-parameter AI API (Completely free for limits up to 15 PDFs/min). Get a free key <a href='https://aistudio.google.com/app/apikey' target='_blank'>here</a>.</span>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            api_key_input = st.text_input("🔑 Gemini API Key", type="password", key="gemini_key_input")
        with c2:
            st.write("")
            st.write("")
            if st.button("🔌 Connect API Engine", use_container_width=True):
                if api_key_input:
                    with open(".env", "a") as f:
                        f.write(f"\nGEMINI_API_KEY={api_key_input.strip()}\n")
                    os.environ["GEMINI_API_KEY"] = api_key_input.strip()
                    st.cache_resource.clear()
                    st.rerun()
    
    st.markdown("---")
    st.subheader("Recent Documents")
    docs = db.get_all_documents()
    if docs:
        st.dataframe(pd.DataFrame(docs, columns=["ID", "Filename", "Date Uploaded", "Status", "Size (bytes)"]).head(5))

elif page == "Upload Document":
    st.title("📄 Upload Standard Document")
    st.write("Upload a Pharmaceutical Standard (ASTM, ISO, etc.) in PDF or TXT format.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Save temp file
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        processor = ASTMProcessor(temp_path)
        
        if st.button("Process Document"):
            if not models_loaded:
                st.warning("NLP models failed to load. Cannot process.")
            else:
                progress_bar = st.progress(0)
                st.info("Extracting Text...")
                res = processor.extract_text()
                progress_bar.progress(30)
                
                if res['success']:
                    text = res['text']
                    doc_type = processor.detect_document_type(text)
                    st.success(f"Successfully extracted {len(text)} characters. Type: {doc_type}")
                    
                    st.info("Running Semantic Classification & NLP Extraction...")
                    sections, chemicals, equipment, procedures = extract_and_save_data(text, res['filename'], res['file_hash'], file_details['FileSize'])
                    progress_bar.progress(100)
                    
                    st.success("Data successfully saved to database!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("🧪 Extracted Chemicals"):
                            st.json(chemicals)
                        with st.expander("⚙️ Extracted Equipment"):
                            st.json(equipment)
                    with col2:
                        with st.expander("📋 Extracted Procedures"):
                            st.json(procedures)
                else:
                    st.error("Failed to extract text from document.")
                    
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

elif page == "Test Selector":
    st.title("🧪 Test Selector")
    
    substances = db.get_all_substances()
    
    if not substances:
        st.warning("No tests found in database.")
    else:
        selected_substance = st.selectbox("Select Target Substance (What do you want to test?)", substances)
        
        tests = db.get_tests_by_substance(selected_substance)
        if tests:
            st.write(f"Available tests for **{selected_substance}**:")
            for t in tests:
                with st.expander(f"Test: {t[1]}"):
                    st.write(f"**Description:** {t[2]}")
                    details = db.get_test_details(t[0])
                    
                    st.subheader("Requirements")
                    req_df = pd.DataFrame(details["requirements"], columns=["Type", "Name", "Quantity", "Unit", "Ratio"])
                    st.dataframe(req_df, use_container_width=True)
                    
                    st.subheader("Simplified Procedures")
                    for p in details["procedures"]:
                        st.write(f"**{p[0]}.** {p[1]}")
        else:
            st.info("No tests linked to this substance.")

elif page == "Document Manager":
    st.title("📂 Document Manager")
    
    docs = db.get_all_documents()
    if docs:
        df = pd.DataFrame(docs, columns=["ID", "Filename", "Date Uploaded", "Status", "Size (bytes)"])
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Manage")
        doc_dict = {f"{d[0]} - {d[1]}": d[0] for d in docs}
        selected_doc = st.selectbox("Select Document to Delete", list(doc_dict.keys()))
        
        if st.button("Delete Document"):
            db.delete_document(doc_dict[selected_doc])
            st.success("Document deleted (cascade delete applied to dependent tests).")
            st.rerun()
    else:
        st.info("No documents uploaded yet.")

elif page == "Inventory":
    st.title("🗄️ Global Inventory")
    st.write("A master catalog of all distinct chemicals and apparatuses identified across all processed standards.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 All Chemicals")
        chemicals = db.get_all_chemicals()
        if chemicals:
            chem_df = pd.DataFrame(chemicals, columns=["Database ID", "Chemical Name"])
            st.dataframe(chem_df, use_container_width=True, hide_index=True)
        else:
            st.info("No chemicals extracted yet.")
            
    with col2:
        st.subheader("⚙️ All Apparatuses")
        apparatuses = db.get_all_apparatuses()
        if apparatuses:
            app_df = pd.DataFrame(apparatuses, columns=["Equipment Name"])
            st.dataframe(app_df, use_container_width=True, hide_index=True)
        else:
            st.info("No equipment extracted yet.")

elif page == "About Me":
    st.title("🎓 About the Developer")
    st.markdown("""
    Welcome to the **Pharma NLP System** project submission.
    
    * **Student Name:** *Anushka Jain*
    * **Student ERP:** *1032220621*
    * **Course:** NLP Project Based Learning (PBL)
    * **Institution:** MIT World Peace University
    
    ### Project Overview
    This application orchestrates an end-to-end local intelligence pipeline designed to rip apart deeply nested pharmaceutical PDFs. Using chunked **Semantic Paragraph Classification**, **Generative LLM Context Slicing**, and complex SQL routing, it mathematically constructs dynamic laboratory dashboards with zero human intervention.
    """)

# Global Footer
st.markdown("""
<div class="footer">
    Valence Intelligence Core — Powered by Sentence-Transformers & Llama 3.2 Frameworks © 2026
</div>
""", unsafe_allow_html=True)
