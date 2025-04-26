from typing import List, Dict, Any
import logging
from transformers import pipeline
import torch
import spacy
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_pipeline = None
        self.intent_classifier = None
        self.nlp = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the NER and intent classification models"""
        try:
            # Initialize NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=self.device
            )
            
            # Initialize intent classification pipeline
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            # Initialize spaCy for additional entity extraction
            self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("Entity extraction models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing entity extraction models: {str(e)}")
            raise

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using both BERT and spaCy"""
        try:
            # Get entities from BERT
            bert_entities = self.ner_pipeline(text)
            
            # Get entities from spaCy
            doc = self.nlp(text)
            spacy_entities = [
                {
                    "entity": ent.label_,
                    "word": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy"
                }
                for ent in doc.ents
            ]
            
            # Combine and deduplicate entities
            combined_entities = []
            seen_entities = set()
            
            for ent in bert_entities:
                key = (ent["word"], ent["entity"])
                if key not in seen_entities:
                    combined_entities.append({
                        "entity": ent["entity"],
                        "word": ent["word"],
                        "score": ent["score"],
                        "source": "bert"
                    })
                    seen_entities.add(key)
            
            for ent in spacy_entities:
                key = (ent["word"], ent["entity"])
                if key not in seen_entities:
                    combined_entities.append(ent)
                    seen_entities.add(key)
            
            return combined_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def classify_intent(self, text: str, possible_intents: List[str] = None) -> Dict[str, float]:
        """Classify the intent of the text"""
        try:
            if not possible_intents:
                possible_intents = [
                    "question",
                    "statement",
                    "request",
                    "command",
                    "greeting",
                    "farewell"
                ]
            
            # Use zero-shot classification for intent
            result = self.intent_classifier(
                text,
                candidate_labels=possible_intents,
                multi_label=True
            )
            
            # Return intents and their probabilities
            return dict(zip(result["labels"], result["scores"]))
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return {}

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze a message for both entities and intent"""
        try:
            entities = self.extract_entities(message)
            intent = self.classify_intent(message)
            
            # Group entities by type
            entities_by_type = defaultdict(list)
            for entity in entities:
                entities_by_type[entity["entity"]].append(entity["word"])
            
            return {
                "entities": dict(entities_by_type),
                "intent": intent,
                "raw_entities": entities
            }
            
        except Exception as e:
            logger.error(f"Error analyzing message: {str(e)}")
            return {
                "entities": {},
                "intent": {},
                "raw_entities": []
            } 