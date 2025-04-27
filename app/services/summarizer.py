from typing import List, Dict, Any
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, vector_store=None):
        self.model_name = "Falconsai/text_summarization"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        self.vector_store = vector_store  # Optionally inject

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            
            # Format the message
            message_parts = [f"{role}: {content}"]
            
            # Add analysis if present
            if 'analysis' in msg and isinstance(msg['analysis'], dict):
                analysis = msg['analysis']
                entities = analysis.get('entities', {})
                intent = analysis.get('intent', {})
                
                analysis_parts = []
                # Add entities
                for entity_type, values in entities.items():
                    if values:
                        analysis_parts.append(f"{entity_type}: {', '.join(values)}")
                # Add intents with high confidence
                for intent_type, score in intent.items():
                    if score > 0.3:
                        analysis_parts.append(f"{intent_type}: {score:.2f}")
                
                if analysis_parts:
                    message_parts.append(f"[Analysis: {' | '.join(analysis_parts)}]")
            
            formatted_messages.append(" ".join(message_parts))
        
        return "\n".join(formatted_messages)

    def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a summary of the conversation."""
        try:
            # Convert messages to text format
            conversation_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in messages
            ])
            
            # Generate summary
            summary = self.summarizer(
                conversation_text,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            
            # Store summary in vector store with serialized metadata
            if self.vector_store:
                # Convert analysis data to string if present
                serialized_analysis = {}
                for msg in messages:
                    if 'analysis' in msg and isinstance(msg['analysis'], dict):
                        serialized_analysis[msg['role']] = json.dumps(msg['analysis'])
                
                # Get conversation_id from first message
                conversation_id = None
                for msg in messages:
                    if 'conversation_id' in msg:
                        conversation_id = msg['conversation_id']
                        break
                
                if not conversation_id:
                    logger.warning("No conversation_id found in messages, generating a new one")
                    conversation_id = str(uuid.uuid4())
                
                metadata = {
                    "type": "summary",
                    "conversation_id": conversation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_count": len(messages),
                    "analysis": json.dumps(serialized_analysis) if serialized_analysis else "{}"
                }
                self.vector_store.store_summary(summary, metadata)
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating conversation summary."

    def generate_key_points(self, messages: List[Dict[str, Any]], num_points: int = 3) -> List[str]:
        if not messages:
            return []
        text_to_analyze = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        summary = self.summarizer(
            text_to_analyze,
            max_length=200,
            min_length=100,
            do_sample=False
        )[0]["summary_text"]
        key_points = summary.split(". ")
        key_points = [point.strip() + "." for point in key_points[:num_points]]
        return key_points
