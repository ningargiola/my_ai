# Application Settings (`core/config.py`)

The `Settings` class manages all core configuration for your FastAPI app, leveraging [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for environment variable management. This ensures a clear, secure, and flexible way to configure your application across different environments.

---

## Configuration Fields

| Name                        | Type    | Default                    | Description                                 |
|-----------------------------|---------|----------------------------|---------------------------------------------|
| API_V1_STR                  | str     | "/api/v1"                  | Base path for API versioning                |
| PROJECT_NAME                | str     | "Personal AI Assistant"    | Name of the project                         |
| OLLAMA_BASE_URL             | str     | "http://localhost:11434"   | Base URL for the Ollama server              |
| DEFAULT_MODEL               | str     | "llama3"                   | Default AI model                            |
| VECTOR_DB_PATH              | str     | "./data/vector_db"          | Path to the vector database directory       |
| SECRET_KEY                  | str     | "your-secret-key-here"      | Secret key for auth       |
| ALGORITHM                   | str     | "HS256"                     | Algorithm for JWT signing                   |
| ACCESS_TOKEN_EXPIRE_MINUTES | int     | 30                         | Access token expiration (in minutes)        |

---

## Example `.env` file

```
API_V1_STR=/api/v1
PROJECT_NAME=Personal AI Assistant
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3
VECTOR_DB_PATH=./data/vector_db
SECRET_KEY=your-actual-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## Usage

Import and use the settings anywhere in your app:

```python
from app.core.config import settings

print(settings.OLLAMA_BASE_URL)
```

---

## Best Practices

- **Never commit your real secret keys to version control.** Use `.env.example` as a template.
- Adjust sensitive settings with environment variables in production.
- Use descriptive names and keep config centralized for easy maintenance.

---

*Update this file if you add, remove, or change configuration options in your application.*

