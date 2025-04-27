# Personal AI Assistant

A powerful, extensible AI assistant built with [FastAPI](https://fastapi.tiangolo.com/), [Ollama](https://ollama.com/), and state-of-the-art NLP pipelines. This project supports real-time chat, conversation memory, semantic search, and multiple LLMs—designed for both personal use and as a robust boilerplate for advanced AI assistants.

---

## Features

- **Persistent Conversation Context:** Remembers your conversation history and context across sessions.
- **Multiple Model Support:** Seamlessly switch or chain LLMs with Ollama for different use cases.
- **REST & WebSocket API:** Flexible integration for web, mobile, or CLI clients.
- **NLP Analysis:** Entity extraction, intent classification, and real-time summarization with transformers and spaCy.
- **Secure Authentication:** Easy JWT setup for safe, multi-user access.
- **Semantic Vector Storage:** Fast, accurate retrieval of relevant chats using ChromaDB.
- **Test Coverage:** Thoroughly tested business logic and API endpoints for robust operation.

---

## Setup

1. **Install Ollama and pull your desired model:**
   ```bash
   ollama pull llama2  # or any other model you prefer
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file with your configuration:**
   ```bash
   cp .env.example .env
   ```

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Project Structure

```
.
├── app/
│   ├── api/           # API routes and endpoints
│   ├── core/          # Core functionality and configuration
│   ├── models/        # Data models and chat logic
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic & AI pipelines
│   └── utils/         # Utility functions
├── tests/             # Test files and pytest fixtures
├── .env.example       # Example environment variables
├── requirements.txt   # Project dependencies
├── README.md          # Project overview (this file)
```

---

## API Documentation

Once the server is running, visit [`http://localhost:8000/docs`](http://localhost:8000/docs) for interactive API documentation.

---

## Architecture & Developer Docs

See [architecture.md](architecture.md) for a full breakdown of:
- Codebase structure
- End-to-end workflows
- Service documentation
- Links to all module/test docs

---

## Contributing

Contributions, feature requests, and bug reports are welcome! Please open an issue or submit a pull request with clear context and sample data where possible.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
