import requests
import logging
from typing import List, Dict, Any
import json
import traceback

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, model_name: str = "llama3"):
        try:
            logger.debug(f"Initializing OllamaService with model: {model_name}")
            self.model_name = model_name
            self.base_url = "http://localhost:11434"
            self.system_prompt = """You are a helpful assistant that can answer questions and help with tasks. Act as though you are a personal assistant with a human tone"""
            logger.debug("OllamaService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OllamaService: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_response(self, message: str, context: str = "", conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate a response using Ollama with context from vector store"""
        try:
            # Prepare the messages array
            messages = [{"role": "system", "content": self.system_prompt}]
            logger.debug("Added system prompt to messages")
            
            # Add context if available
            if context:
                messages.append({"role": "system", "content": f"Context from previous conversations:\n{context}"})
                logger.debug("Added context from previous conversations")
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
                logger.debug(f"Added {len(conversation_history)} messages from conversation history")
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            logger.debug("Added current message to messages")
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1,
                    "top_k": 50,
                    "num_predict": 1000,
                    "stop": ["</s>", "User:", "Assistant:"]
                }
            }
            
            logger.debug(f"Sending request to Ollama with {len(messages)} messages")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            if "message" in result and "content" in result["message"]:
                logger.debug("Successfully extracted response content")
                return result["message"]["content"]
            else:
                logger.error(f"Unexpected response format: {json.dumps(result, indent=2)}")
                raise ValueError("Unexpected response format from Ollama")
                
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to Ollama service. Is Ollama running?")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def list_available_models(self) -> List[str]:
        """List all available Ollama models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
            
    def set_model(self, model: str):
        """Set the model to use for generation."""
        self.model_name = model 