from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class SemanticSectionDetector:
    """
    Uses semantic similarity (Sentence-BERT) to classify sections even when 
    section headers use different wording (e.g., “Apparatus” vs “Equipment Required”).
    We use all-MiniLM-L6-v2 for a good balance of speed and accuracy.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Define categories and example phrases for zero-shot classification via similarity
        self.section_definitions = {
            "Scope": [
                "This standard defines the scope",
                "The scope of this test method covers",
                "This test method describes the procedure for",
                "Application and scope",
                "1. Scope"
            ],
            "Apparatus": [
                "Apparatus and equipment required",
                "The following equipment is needed",
                "Instrumentation",
                "Glassware and measuring devices",
                "Equipment list"
            ],
            "Reagents": [
                "Reagents and materials",
                "Chemicals used in this procedure",
                "Purity of reagents",
                "Solutions and solvents required",
                "Reagents and reference standards"
            ],
            "Procedure": [
                "Procedure and instructions",
                "Step-by-step method",
                "Test procedure",
                "Preparation of the sample",
                "Conducting the assay"
            ],
            "Calculation": [
                "Calculations and results",
                "Equations to determine the outcome",
                "Calculate the percentage",
                "Statistical analysis and data processing"
            ],
            "Report": [
                "Reporting the results",
                "Final report format",
                "Information to be reported",
                "Report the following test data"
            ]
        }
        
        # Pre-compute embeddings for all example phrases to save computation time later
        self.category_embeddings = {}
        for category, phrases in self.section_definitions.items():
            # torch tensor of embeddings for each phrase in this category
            self.category_embeddings[category] = self.model.encode(phrases, convert_to_tensor=True)

    def classify_section(self, text_chunk, threshold=0.45):
        """
        Classify a single paragraph/chunk into a section type.
        Returns dict with section_type, confidence, matched_phrase, raw_text.
        """
        if not text_chunk or len(text_chunk.strip()) < 10:
            return {"section_type": "Unknown", "confidence": 0.0, "matched_phrase": None, "raw_text": text_chunk}
            
        chunk_embedding = self.model.encode(text_chunk, convert_to_tensor=True)
        
        best_category = "Unknown"
        highest_score = 0.0
        best_phrase = None
        
        for category, embeddings in self.category_embeddings.items():
            # Calculate cosine similarities
            cos_scores = util.cos_sim(chunk_embedding, embeddings)[0]
            max_score_idx = torch.argmax(cos_scores).item()
            score = cos_scores[max_score_idx].item()
            
            if score > highest_score:
                highest_score = score
                best_category = category
                best_phrase = self.section_definitions[category][max_score_idx]
                
        if highest_score >= threshold:
            return {
                "section_type": best_category, 
                "confidence": highest_score, 
                "matched_phrase": best_phrase, 
                "raw_text": text_chunk
            }
        else:
            return {
                "section_type": "Unknown", 
                "confidence": highest_score, 
                "matched_phrase": best_phrase, 
                "raw_text": text_chunk
            }

    def segment_document(self, full_text):
        """
        Splits text into paragraphs, classifies each, and groups them by section.
        Returns dict mapping section_type to list of raw text paragraphs.
        """
        # Split full text into paragraphs robustly, handling single newlines
        lines = full_text.split('\n')
        paragraphs = []
        current_para = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
            else:
                current_para.append(line_str)
        if current_para:
            paragraphs.append(" ".join(current_para))
            
        paragraphs = [p for p in paragraphs if len(p) > 20]
        
        segmented = {
            "Scope": [],
            "Apparatus": [],
            "Reagents": [],
            "Procedure": [],
            "Calculation": [],
            "Report": [],
            "Unknown": []
        }
        
        current_section = "Unknown"
        
        for paragraph in paragraphs:
            # We look at the first 200 characters of a paragraph to determine section changes
            classification = self.classify_section(paragraph[:200], threshold=0.50)
            
            # If we detect a strong signal for a new section, we switch to it
            if classification["section_type"] != "Unknown":
                current_section = classification["section_type"]
                
            segmented[current_section].append(paragraph)
            
        return segmented
