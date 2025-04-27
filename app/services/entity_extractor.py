from typing import List, Dict, Any
import logging
from transformers import pipeline
import torch
import spacy
from collections import defaultdict

logger = logging.getLogger(__name__)

def aggregate_session_entities_intents(messages: List[Dict]) -> Dict[str, Any]:
    entities_by_type = {}
    intents_agg = {}
    for msg in messages:
        analysis = msg.get('analysis', {})
        for ent_type, ents in analysis.get('entities', {}).items():
            if ent_type not in entities_by_type:
                entities_by_type[ent_type] = set()
            entities_by_type[ent_type].update(ents)
        for intent, score in analysis.get('intent', {}).items():
            intents_agg[intent] = intents_agg.get(intent, 0) + score
    entities_by_type = {k: list(v) for k, v in entities_by_type.items()}
    if intents_agg:
        num_msgs = len(messages)
        intents_agg = {k: v / num_msgs for k, v in intents_agg.items()}
    return {"entities": entities_by_type, "intents": intents_agg}

class EntityExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=self.device
        )
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        bert_entities = self.ner_pipeline(text)
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

    def classify_intent(self, text: str) -> Dict[str, float]:
        """Classify the intent of a message."""
        if not text or not text.strip():
            return {}

        result = self.intent_classifier(
            text,
            candidate_labels=["question", "statement", "request", "command", "greeting", "farewell"],
            hypothesis_template="This example is {}."
        )
        
        return {label: score for label, score in zip(result['labels'], result['scores'])}

    def analyze_message(self, message: str) -> Dict[str, Any]:
        entities = self.extract_entities(message)
        intent = self.classify_intent(message)
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity["entity"]].append(entity["word"])
        return {
            "entities": dict(entities_by_type),
            "intent": intent,
            "raw_entities": entities
        }
