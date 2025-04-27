# Chat API Schemas (`schemas/chat_schemas.py`)

## Overview

This module defines the Pydantic models used for validating API requests and responses related to chat and conversation features.  
It is best practice to separate **schemas** (used for request/response validation) from your core **database models** (which may contain extra fields or logic not needed by the API).

---

## Schemas

### MessageBase

| Field   | Type   | Description                       |
|---------|--------|-----------------------------------|
| role    | str    | Message sender (user, assistant)  |
| content | str    | Message text content              |

### MessageCreate

- Inherits from: `MessageBase`
- Used for: Creating new messages (POST body)

### Message

| Field           | Type     | Description                  |
|-----------------|----------|------------------------------|
| id              | int      | Unique message identifier    |
| conversation_id | int      | Associated conversation      |
| created_at      | datetime | Creation timestamp           |
| ...             | ...      | Inherits from `MessageBase`  |

- Config: `from_attributes = True` (supports ORM mode)

---

### ConversationBase

| Field | Type   | Description      |
|-------|--------|------------------|
| title | str or None | Conversation title (optional) |

### ConversationCreate

- Inherits from: `ConversationBase`
- Used for: Creating new conversations

### Conversation

| Field         | Type        | Description                        |
|---------------|-------------|------------------------------------|
| id            | int         | Unique conversation identifier     |
| created_at    | datetime    | Creation timestamp                 |
| updated_at    | datetime or None | Last update timestamp       |
| messages      | List[Message] | List of messages in conversation  |
| ...           | ...         | Inherits from `ConversationBase`   |

- Config: `from_attributes = True` (supports ORM mode)

---

### ChatRequest

| Field           | Type         | Description                           |
|-----------------|--------------|---------------------------------------|
| message         | str          | User message to the bot               |
| conversation_id | int or None  | Conversation to attach this message   |
| use_history     | bool         | Include history in response           |

### ChatResponse

| Field           | Type                   | Description                         |
|-----------------|------------------------|-------------------------------------|
| conversation_id | int                    | The conversation ID                 |
| response        | str                    | Bot’s generated response            |
| analysis        | dict or None           | Optional analysis output            |

---

### WebSocketMessage

| Field | Type   | Description                     |
|-------|--------|---------------------------------|
| type  | str    | Message type (\"message\"/\"analysis\") |
| data  | dict   | Message or analysis payload     |

---

### AnalysisResult

| Field     | Type                   | Description                         |
|-----------|------------------------|-------------------------------------|
| entities  | List[dict]             | Detected entities                   |
| intents   | List[dict]             | Detected intents                    |
| sentiment | dict or None           | Sentiment analysis results          |
| summary   | str or None            | Summarized message/conversation     |

---

## Usage Example

```python
from schemas.chat_schemas import ChatRequest, ChatResponse

# Parsing a request body
payload = {"message": "Hello!", "use_history": True}
req = ChatRequest(**payload)
