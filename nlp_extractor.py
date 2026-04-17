import re
import spacy

class TechnicalEntityExtractor:
    """
    NLP Extractor using spaCy and regular expressions to parse technical documents.
    Specifically designed for Pharmaceutical technical texts.
    """
    
    def __init__(self):
        # Load the small English spaCy model.
        # It's fast and sufficient for general NER combined with our rule-based methods.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model isn't downloaded yet. We instruct the user to download it in README.
            # But we can try to fall back or raise a clear error.
            raise OSError("Please download the spacy model using: python -m spacy download en_core_web_sm")

        # Compile common chemical and pharmaceutical regex patterns
        # Matches formats like "Sodium Chloride"
        self.chemical_pattern = re.compile(r'\b([A-Z][a-z]+(\s+acid|\s+chloride|\s+sulfate|\s+hydroxide|\s+oxide|\s+nitrate|\s+phosphate|\s+carbonate))\b')
        
    def extract_chemicals(self, text):
        """
        Extract chemicals using 3 methods:
        1. spaCy NER (looking for PRODUCT or ORG which sometimes catches chemicals)
        2. Regex patterns
        3. Part-of-Speech and noun-chunks
        Returns a deduplicated list of dicts.
        """
        chemicals = {}
        
        doc = self.nlp(text)
        
        # Method 1: spaCy NER
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'CHEMICAL', 'SUBSTANCE'] and len(ent.text.strip()) > 3: 
                # Avoid capturing standard organizational text
                if not any(word in ent.text.lower() for word in ['astm', 'usa', 'copyright', 'committee', 'standard']):
                    name = ent.text.strip().lower()
                    chemicals[name] = {"name": ent.text.strip(), "method": "spaCy NER", "confidence": 0.85}
                
        # Method 2: Regex
        matches = self.chemical_pattern.finditer(text)
        for match in matches:
            name = match.group(0).strip()
            # Ignore simple acronyms that are too short to be unique chemicals
            if len(name) > 2 and name.lower() not in chemicals:
                chemicals[name.lower()] = {"name": name, "method": "Regex", "confidence": 0.95}

        # Method 3: Noun chunks filtered by keywords
        keywords = ["solution", "buffer", "reagent", "acid", "solvent", "catalyst"]
        generic_modifiers = ['specific', 'speciﬁc', 'possible', 'various', 'such', 'these', 'those', 'general', 'typical', 'any', 'other', 'all']
        for chunk in doc.noun_chunks:
            name_lower = chunk.text.lower()
            if any(kw in name_lower for kw in keywords):
                name = chunk.text.strip()
                
                # Check if the chunk is JUST the keyword itself
                is_just_kw = any(name_lower == kw or name_lower == kw + 's' for kw in keywords)
                
                if len(name) > 3 and name_lower not in chemicals and not is_just_kw:
                     if not any(word in name_lower for word in ['astm', 'usa', 'copyright', 'committee', 'standard'] + generic_modifiers):
                        chemicals[name_lower] = {"name": name, "method": "Noun-Chunk", "confidence": 0.70}

        # Filter out common false positives
        blacklist = {'astm', 'usa', 'copyright', 'ihs', 'mdt', 'appendix', 'selection', 'sample', 'utc', 'p&w', 'iso', 'table', 'figure', 'section', 'part'}
        filtered_chemicals = {k: v for k, v in chemicals.items() if not any(b in k.lower() for b in blacklist)}

        return list(filtered_chemicals.values())

    def extract_equipment(self, text):
        """
        Uses keyword lists grouped by category to identify apparatus and equipment.
        """
        categories = {
            "Measurement": ["balance", "scale", "spectrophotometer", "chromatograph", "hplc", "gc", "thermometer", "meter", "gauge", "sensor", "detector"],
            "Glassware": ["beaker", "flask", "pipette", "burette", "cylinder", "dish", "vial", "crucible", "tube", "glassware"],
            "Electronics": ["stirrer", "hot plate", "incubator", "centrifuge", "oven", "furnace", "sonicator", "pump"],
            "General": ["filter", "spatula", "desiccator", "hood", "syringe", "distillation", "column", "resin", "apparatus"]
        }
        
        equipment_found = []
        text_lower = text.lower()
        
        for category, keywords in categories.items():
            for kw in keywords:
                # Find all occurrences
                starts = [m.start() for m in re.finditer(rf'\b{re.escape(kw)}\b', text_lower)]
                for start in starts:
                    # Get surrounding context (up to 50 chars)
                    context_start = max(0, start - 25)
                    context_end = min(len(text), start + len(kw) + 25)
                    context = text[context_start:context_end].replace("\n", " ").strip()
                    
                    equipment_found.append({
                        "name": kw.title(),
                        "category": category,
                        "keyword_matched": kw,
                        "full_description": context
                    })
                    
        # Basic deduplication
        unique_equipment = {e["name"]: e for e in equipment_found}
        return list(unique_equipment.values())

    def extract_quantities(self, text):
        """
        Uses unit patterns to extract quantities.
        Example: "10 mL", "5.5 g", "2 hours"
        """
        # (value) (unit)
        pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(mL|L|g|mg|kg|mol|M|N|°C|min|hours?|hrs?|sec|%)\b', re.IGNORECASE)
        
        unit_types = {
            "ml": "volume", "l": "volume",
            "g": "mass", "mg": "mass", "kg": "mass",
            "mol": "concentration", "m": "concentration", "n": "concentration",
            "°c": "temperature",
            "min": "time", "hour": "time", "hours": "time", "hr": "time", "hrs": "time", "sec": "time",
            "%": "percentage"
        }
        
        quantities = []
        for match in pattern.finditer(text):
            value = float(match.group(1))
            unit = match.group(2)
            unit_type = unit_types.get(unit.lower(), "unknown")
            
            # Simple context extraction
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].replace("\n", " ")
            
            quantities.append({
                "value": value,
                "unit_type": unit_type,
                "unit": unit,
                "context": context.strip()
            })
            
        return quantities

    def extract_procedure_steps(self, text):
        """
        Uses regex patterns to identify ordered procedural steps.
        Falls back to sentence splitting if no numbers are found.
        """
        steps = []
        
        # Look for patterns like "1. ", "1.1 ", "10.1.1 "
        step_pattern = re.compile(r'^\s*(?:Step\s+)?(\d+(?:\.\d+)*)[\.\)]?\s+(.*)', re.MULTILINE | re.IGNORECASE)
        
        matches = list(step_pattern.finditer(text))
        
        if matches:
            for match in matches:
                step_num_str = match.group(1)
                instruction_str = match.group(2).strip()
                
                # Filter out obvious false positives like table decimals (0.055) 
                # or single word items like Roman Numerals ("IV") or short numbers ("18")
                if step_num_str.startswith('0') or len(instruction_str) < 5 or instruction_str.replace('.', '').isdigit():
                    continue
                    
                # We save it as string since it can be "10.1"
                steps.append({
                    "step_number": step_num_str,
                    "instruction": instruction_str,
                    "pattern_used": "Numbered list"
                })
        else:
            # Fallback: Treat each sentence as a step if no strict numbering is found
            doc = self.nlp(text)
            for i, sent in enumerate(doc.sents):
                if len(sent.text.strip()) > 15: # Ignore very short sentences
                    steps.append({
                        "step_number": i + 1,
                        "instruction": sent.text.strip().replace('\n', ' '),
                        "pattern_used": "Sentence Split"
                    })
                    
        return steps
