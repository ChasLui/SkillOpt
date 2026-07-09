# Add a New Model Backend

SkillOpt supports multiple LLM backends. This guide shows how to add your own.

## Built-in: the generic OpenAI-compatible backend

Before writing a new backend, check whether your provider already speaks the
OpenAI Chat Completions protocol. Most do — in which case you can use the
built-in **`openai_compatible`** backend
(`skillopt/model/openai_compatible_backend.py`) with no code changes.

A single `base_url` + `api_key` pair lets you point SkillOpt at, for example:

| Provider | `base_url` | Example model |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5:7b` |
| vLLM / SGLang / TGI | `http://localhost:8000/v1` | your served model |
| LiteLLM proxy | `http://localhost:4000` | any proxied model |
| OpenRouter / Fireworks / xAI / … | provider base URL | provider model id |

Select it as the optimizer and/or target backend:

```python
import skillopt.model as model

# Shorthand: use it for both optimizer and target.
model.set_backend("openai_compatible")

# Point it at a provider (shared, or per-role with optimizer_*/target_*).
model.configure_openai_compatible(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-...",
    model="deepseek-chat",
)
```

Or configure it entirely through environment variables (role-specific
`OPTIMIZER_*` / `TARGET_*` variants override the shared ones):

```bash
export TARGET_BACKEND=openai_compatible
export OPENAI_COMPATIBLE_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_COMPATIBLE_API_KEY="gsk_..."
export OPENAI_COMPATIBLE_MODEL="llama-3.3-70b-versatile"
# Optional: OPENAI_COMPATIBLE_TEMPERATURE, _MAX_TOKENS, _TIMEOUT_SECONDS
```

The backend uses the official `openai` SDK, records token usage through the
shared tracker, supports tool/function calling via
`chat_target_messages(..., tools=...)`, and exposes
`count_tokens()` (tiktoken with a character-based fallback for non-OpenAI
models). Only write a brand-new backend if your provider is *not*
OpenAI-compatible.

## Backend Architecture

```
skillopt/model/
├── base.py           # Abstract base class
├── azure_openai.py   # Azure OpenAI backend
├── openai_model.py   # Direct OpenAI backend
├── claude.py         # Anthropic Claude backend
├── qwen.py           # Local Qwen (vLLM) backend
└── your_backend.py   # Your new backend
```

## Step 1: Create the Backend

Create `skillopt/model/your_backend.py`:

```python
from skillopt.model.base import ModelBackend, ModelResponse

class YourBackend(ModelBackend):
    """Your custom model backend."""
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_name = cfg.get('model_name', 'your-default-model')
        self.api_key = os.environ.get('YOUR_API_KEY', '')
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize API client."""
        # TODO: Set up your API client
        pass
    
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a completion.
        
        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            ModelResponse with content, usage, and metadata
        """
        response = await self.client.chat(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return ModelResponse(
            content=response.text,
            usage={
                'prompt_tokens': response.usage.input,
                'completion_tokens': response.usage.output,
            },
            model=self.model_name,
        )
    
    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs
    ) -> ModelResponse:
        """Generate with tool/function calling support."""
        # Optional: implement if your model supports tool use
        raise NotImplementedError("Tool use not supported")
```

## Step 2: Register the Backend

Add to `skillopt/model/__init__.py`:

```python
from .your_backend import YourBackend

BACKEND_REGISTRY = {
    # ... existing backends ...
    'your_backend': YourBackend,
}
```

## Step 3: Configure

Use your backend in any config:

```yaml
model:
  backend: your_backend
  model_name: your-model-id
  temperature: 0.7
  max_tokens: 4096
```

Set credentials via environment variable:

```bash
export YOUR_API_KEY="your-key"
```

## Required Interface

Your backend must implement these methods:

| Method | Required | Description |
|---|---|---|
| `generate()` | ✅ | Basic text generation |
| `generate_with_tools()` | Optional | Tool/function calling |
| `count_tokens()` | Optional | Token counting for context management |

## Tips

!!! tip
    - Test your backend with `python -c "from skillopt.model.your_backend import YourBackend"` first
    - Use `async` methods for all API calls — SkillOpt uses asyncio throughout
    - Implement retry logic with exponential backoff for production use
    - Add your API key to `.env.example` when submitting a PR
