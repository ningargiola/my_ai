# Personal AI Assistant

A personal AI assistant built using Ollama and FastAPI that maintains conversation context.

## Features

- Conversation context management
- Multiple model support through Ollama
- REST API interface
- Secure authentication
- Persistent conversation storage

## Setup

1. Install Ollama and pull your desired model:
```bash
ollama pull llama2  # or any other model you prefer
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```bash
cp .env.example .env
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
.
├── app/
│   ├── api/           # API routes and endpoints
│   ├── core/          # Core functionality and configuration
│   ├── models/        # Database models
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic
│   └── utils/         # Utility functions
├── tests/             # Test files
├── .env.example       # Example environment variables
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation. 