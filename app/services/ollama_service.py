import ollama
from typing import List, Dict, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.base_url = getattr(settings, 'OLLAMA_BASE_URL', None)

    def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        extra_context: Optional[List[Dict]] = None,
    ) -> str:
        """Generate a response using Ollama, with flexible context."""
        try:
            logger.debug(f"Generating response with model: {self.model}")

            # Build up the context for the LLM
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if extra_context:
                messages.extend(extra_context)
            if conversation_history:
                messages.extend(conversation_history)
            # Always append the current user message last
            messages.append({"role": "user", "content": message})

            logger.debug(f"Messages sent to Ollama:\n{messages}")

            response = ollama.chat(model=self.model, messages=messages)
            logger.debug("Received response from Ollama")

            # Extract the response content
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
                return response['message']['content']
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                logger.error(f"Response object: {response}")
                raise ValueError("Unexpected response format from Ollama")
        except Exception as e:
            logger.error(f"Error in Ollama service: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            raise Exception(f"Error generating response: {str(e)}")

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
        self.model = model
        logger.debug(f"Ollama model set to: {model}")
