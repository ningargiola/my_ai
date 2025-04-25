import requests
import json
import sys
from typing import Optional, List, Dict
from datetime import datetime
import socket
import subprocess

class AIChat:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_id: Optional[int] = None
        self.session = requests.Session()
        self.load_all_conversations()

    def load_all_conversations(self):
        """Load all conversations from the database."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/conversations")
            response.raise_for_status()
            self.all_conversations = response.json()
            if self.all_conversations:
                print(f"\nLoaded {len(self.all_conversations)} previous conversations.")
        except requests.exceptions.RequestException as e:
            self.all_conversations = []
            print(f"\nError loading conversations: {str(e)}")
            print("Make sure the FastAPI server is running and accessible at", self.base_url)

    def send_message(self, message: str) -> str:
        """Send a message to the AI and get the response."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/chat",
                json={
                    "message": message,
                    "conversation_id": self.conversation_id,
                    "use_history": True
                }
            )
            response.raise_for_status()
            data = response.json()
            self.conversation_id = data["conversation_id"]
            return data["response"]
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def list_conversations(self) -> List[Dict]:
        """Get list of all conversations."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/conversations")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return []

    def get_conversation(self, conversation_id: int) -> Dict:
        """Get details of a specific conversation."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/conversations/{conversation_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}

    def print_conversation_history(self, conversation: Dict):
        """Print the history of a conversation."""
        if not conversation or "messages" not in conversation:
            print("No messages in this conversation.")
            return

        print("\nConversation History:")
        for msg in conversation["messages"]:
            role = "You" if msg["role"] == "user" else "AI"
            timestamp = datetime.fromisoformat(msg["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] {role}:")
            print(msg["content"])

    def start_chat(self):
        """Start an interactive chat session."""
        print("\nWelcome to your Personal AI Assistant!")
        print("I remember all our previous conversations.")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'new' to start a new conversation.")
        print("Type 'help' to see available commands.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! Have a great day!")
                    break
                
                elif user_input.lower() == 'new':
                    self.conversation_id = None
                    print("\nStarted a new conversation!")
                    continue
                
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit/quit - End the conversation")
                    print("  new      - Start a new conversation")
                    print("  help     - Show this help message")
                    continue

                if not user_input:
                    continue

                print("\nAI: ", end="", flush=True)
                response = self.send_message(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Have a great day!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue

def check_server_connection():
    """Check if the server is running and accessible."""
    try:
        # Check if port 8000 is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            # Port is in use, check if it's our FastAPI server
            try:
                response = requests.get("http://localhost:8000/docs", timeout=2)
                if response.status_code == 200:
                    return True
                else:
                    print("Port 8000 is in use but not by a FastAPI server")
                    return False
            except requests.exceptions.RequestException:
                print("Port 8000 is in use but not responding correctly")
                return False
        else:
            print("Port 8000 is not in use")
            return False
            
    except socket.error as e:
        print(f"Socket error: {str(e)}")
        return False

def main():
    if not check_server_connection():
        print("\nPlease follow these steps:")
        print("1. First, kill any existing processes using port 8000:")
        print("   lsof -i :8000  # To see what's using the port")
        print("   kill <PID>     # To kill the process (replace <PID> with the process ID)")
        print("\n2. Then start the FastAPI server:")
        print("   uvicorn app.main:app --reload")
        print("\n3. Finally, run this chat interface again:")
        print("   python chat.py")
        sys.exit(1)

    chat = AIChat()
    chat.start_chat()

if __name__ == "__main__":
    main() 