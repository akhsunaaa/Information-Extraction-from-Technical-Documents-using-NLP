import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

class RagAssistant:
    """
    Manages the RAG pipeline using local chroma DB and local Ollama inference instance.
    Utilizes Sentence-Transformers MiniLM for embeddings (100% local, no API).
    """
    def __init__(self, persist_directory="./chroma_db", model_name="llama3.2"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # We wrap in Try/Except because the Vector DB might be locked if accessed aggressively 
        # or initialized poorly.
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="pharma_standards"
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = None
            
        # Check for Gemini API Key locally
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.use_gemini = bool(self.gemini_key)
        
        # Instantiate LLM Model (Dynamic Routing)
        if self.use_gemini:
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=self.gemini_key, temperature=0.1)
                self.nlp_engine = "Google Gemini 2.5 Flash"
            except Exception as e:
                print(f"Error binding to Google Gemini: {e}")
                self.llm = None
        else:
            try:
                self.llm = OllamaLLM(model=model_name, temperature=0.1, num_predict=4000, repeat_penalty=1.2)
                self.nlp_engine = f"Local Ollama ({model_name})"
            except Exception as e:
                print(f"Error binding to Ollama: {e}")
                self.llm = None
            
        self.prompt_template = """
You are an expert World-Class NLP AI and Pharmaceutical Quality Control Inspector.
Use the provided pieces of the standard laboratory document (context) to definitively answer the user's question.
If you do not know the answer based strictly on the text provided, simply declare "I cannot find the answer in the provided document." Do not try to make up an answer or pull from external laboratory knowledge. Use bullet-points where appropriate for clarity.

Document Context: {context}

User Question: {question}

Expert Answer:"""
        self.QA_CHAIN_PROMPT = PromptTemplate(
            template=self.prompt_template, 
            input_variables=["context", "question"]
        )

    def is_healthy(self):
        """Returns True if the system can bind to Ollama."""
        return self.llm is not None

    def index_document(self, doc_id, filename, text_chunks):
        """
        Takes segmented documents (or large paragraphs) and turns them into 
        searchable vector coordinates.
        """
        if not self.vector_store:
            return False
            
        docs = []
        for i, chunk in enumerate(text_chunks):
            # We ignore very short chunks to save db space
            if len(chunk.strip()) > 50:
                doc = Document(
                    page_content=chunk,
                    metadata={"doc_id": doc_id, "filename": filename, "chunk_id": i}
                )
                docs.append(doc)
                
        if docs:
            self.vector_store.add_documents(docs)
            return True
        return False
        
    def ask_question(self, query):
        """
        Queries the vector store for top 4 most relevant chunks, passes them to 
        local Llama 3.2, and yields an intelligent answer.
        """
        if not self.vector_store or not self.llm:
            return "RAG System is Offline. Please ensure Ollama is installed and running."
            
        try:
            # Manually retrieve from VectorDB instead of relying on legacy chains
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            
            # Format context
            context_str = "\n\n".join([d.page_content for d in docs])
            
            # Generate Prompt
            prompt_value = self.QA_CHAIN_PROMPT.format(context=context_str, question=query)
            
            # Execute LLM Answer
            result = self.llm.invoke(prompt_value)
            
            return {"result": result, "source_documents": docs}
            
        except Exception as e:
            return {"result": f"Error communicating with local LLM. Did you run 'ollama run llama3.2'? Log: {e}", "source_documents": []}

    def parse_document_entities(self, full_text):
        """
        Forces the local LLM to extract the document natively into a strict JSON object.
        Replaces the old regex approach.
        """
        if not self.llm:
            return None
            
        prompt = f"""
You are an expert analytical chemist and world-class NLP data extraction architect.
I am handing you a complex laboratory standard document. Your task is to mathematically decipher the text, segregating the "Core Analytical Method" from the minor variations and Quality Control (Calibration/QC) routines.
You must synthesize the "Big Picture" and extract the core chemistry cleanly.

FORMAT YOUR ENTIRE RESPONSE AS A STRICT JSON OBJECT exactly like this pattern:
{{
  "TargetSubstance": "string (Name of the core target analyte)",
  "Tests": [
    {{
      "TestName": "string (Name of the Analytical Method or QC Routine)",
      "Reagents": [
        {{"name": "string (Name of Chemical)", "quantity": "string (Amount)"}}
      ],
      "Apparatus": [
        "string (Name of Equipment)"
      ],
      "Procedures": [
        "string (Key scientific step 1)",
        "string (Key scientific step 2)"
      ]
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- READ THE PROVIDED DOCUMENT! Do not invent answers. Extract the REAL chemical names, real equipment, and real procedure steps from the text!
- SYNTHESIZE AND DISTILL! Discard endless repetitive boilerplate. Extract the TRUE chemical procedures.
- Segregate the main analytical method from calibration/QC methods. Map them as separate objects in the `Tests` array!
- NEVER repeat the exact same procedure step multiple times. Avoid endless semantic loops!
- If a section is completely missing, output an empty list: []
- Do NOT make up steps if they don't exist in the document!

Document Text:
{full_text[:30000]}  # Expanded context to allow dense procedure paragraphs to fit into Llama 3B window
"""
        try:
            # We strictly request JSON format from the LLM
            result = self.llm.invoke(prompt)
            
            # Accommodate LangChain's different output classes (AIMessage vs String)
            result_text = result.content if hasattr(result, "content") else result
            
            import json
            import re
            
            # Clean up the response in case the LLM wrapped it in markdown
            json_pattern = re.search(r'\{.*\}', result_text.replace('\n', ' '), re.IGNORECASE)
            if json_pattern:
                clean_json = json_pattern.group(0)
                data = json.loads(clean_json)
                return data
            else:
                print("LLM failed to return valid JSON.")
                return None
                
        except Exception as e:
            print(f"Error parsing JSON from LLM: {e}")
            return None
