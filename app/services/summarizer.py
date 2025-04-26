from typing import List, Dict, Any
import logging
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self):
        """Initialize the summarizer with a pre-trained model."""
        try:
            # Use a more appropriate model for conversation summarization
            self.model_name = "facebook/bart-large-cnn"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing summarizer: {str(e)}")
            raise

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation messages into a coherent text"""
        try:
            formatted_messages = []
            current_speaker = None
            current_content = []
            
            for msg in messages:
                if not isinstance(msg, dict):
                    logger.warning(f"Skipping invalid message format: {msg}")
                    continue
                    
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                
                # If speaker changes, add previous content and start new
                if role != current_speaker and current_speaker is not None:
                    formatted_messages.append(f"{current_speaker}: {' '.join(current_content)}")
                    current_content = []
                
                current_speaker = role
                current_content.append(content)
                
                # Add analysis if available
                if 'analysis' in msg and isinstance(msg['analysis'], dict):
                    analysis = msg['analysis']
                    entities = analysis.get('entities', {})
                    intent = analysis.get('intent', {})
                    
                    # Format entities
                    entity_text = []
                    if 'PERSON' in entities:
                        entity_text.append(f"People mentioned: {', '.join(entities['PERSON'])}")
                    if 'ORG' in entities:
                        entity_text.append(f"Organizations mentioned: {', '.join(entities['ORG'])}")
                    
                    # Format intent
                    intent_text = []
                    for intent_type, score in intent.items():
                        if score > 0.3:  # Only include significant intents
                            intent_text.append(f"{intent_type}: {score:.2f}")
                    
                    if entity_text or intent_text:
                        current_content.append(f"[Analysis: {' | '.join(entity_text + intent_text)}]")
            
            # Add the last message
            if current_speaker and current_content:
                formatted_messages.append(f"{current_speaker}: {' '.join(current_content)}")
            
            return "\n\n".join(formatted_messages)
        except Exception as e:
            logger.error(f"Error formatting conversation: {str(e)}")
            return ""

    def _calculate_summary_length(self, text: str) -> tuple[int, int]:
        """Calculate appropriate summary lengths based on input text"""
        try:
            if not text:
                return 50, 25
                
            # Count tokens in the input text
            tokens = self.tokenizer.encode(text, truncation=True, max_length=1024)
            input_length = len(tokens)
            
            # Calculate max and min lengths based on input length
            if input_length < 100:
                max_length = min(30, input_length // 2)
                min_length = min(15, max_length // 2)
            elif input_length < 500:
                max_length = min(100, input_length // 3)
                min_length = min(50, max_length // 2)
            else:
                max_length = 150
                min_length = 75
                
            return max_length, min_length
        except Exception as e:
            logger.error(f"Error calculating summary length: {str(e)}")
            return 100, 50  # Default values if calculation fails

    def _post_process_summary(self, summary: str, messages: List[Dict[str, Any]]) -> str:
        """Post-process the summary to ensure accuracy and coherence"""
        try:
            if not summary:
                return "Unable to generate summary at this time."
                
            # Extract key information from the conversation
            user_name = None
            assistant_name = None
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    if 'my name is' in content:
                        user_name = content.split('my name is')[1].split()[0].strip()
                elif msg.get('role') == 'assistant':
                    content = msg.get('content', '').lower()
                    if 'alias' in content:
                        assistant_name = 'ALIAS'
            
            # Fix any name/role confusions
            if user_name and assistant_name:
                summary = summary.replace(f"{assistant_name} is", f"I am")
                summary = summary.replace(f"{user_name} is", f"You are")
            
            # Ensure proper sentence endings
            summary = summary.strip()
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
            
            return summary
        except Exception as e:
            logger.error(f"Error post-processing summary: {str(e)}")
            return summary

    def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a high-quality summary of the conversation"""
        try:
            if not messages:
                return "No messages to summarize."
                
            # Format the conversation
            conversation_text = self._format_conversation(messages)
            if not conversation_text:
                return "Unable to format conversation for summary."
            
            # Calculate appropriate lengths
            max_length, min_length = self._calculate_summary_length(conversation_text)
            
            # Generate summary with improved parameters
            try:
                summary_result = self.summarizer(
                    conversation_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                if not summary_result or not isinstance(summary_result, list) or not summary_result:
                    return "Unable to generate summary at this time."
                    
                summary = summary_result[0].get('summary_text', '')
                if not summary:
                    return "Generated summary is empty."
            except Exception as e:
                logger.error(f"Error in summarization pipeline: {str(e)}")
                return "Error during summarization process."
            
            # Post-process the summary
            summary = self._post_process_summary(summary, messages)
            
            logger.debug(f"Generated summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary at this time."

    def generate_key_points(self, messages: List[Dict[str, Any]], num_points: int = 3) -> List[str]:
        """Extract key points from the conversation"""
        try:
            if not messages:
                return []

            # Format messages into a single text
            text_to_analyze = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])

            # Generate a longer summary for key points extraction
            summary = self.summarizer(
                text_to_analyze,
                max_length=200,
                min_length=100,
                do_sample=False
            )[0]["summary_text"]

            # Split summary into sentences and take the first num_points
            key_points = summary.split(". ")
            key_points = [point.strip() + "." for point in key_points[:num_points]]

            return key_points
        except Exception as e:
            logger.error(f"Error generating key points: {str(e)}")
            return [] 