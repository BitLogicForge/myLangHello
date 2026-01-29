# LLM Provider Configuration - Summary

## What Was Changed

Your application now supports **easy switching between Azure OpenAI and OpenAI** through configuration files.

## New Files

1. **`llm_config.json`** - Main configuration file for LLM providers
2. **`services/llm_factory.py`** - Factory class to create LLM instances
3. **`docs/llm_configuration_guide.md`** - Complete documentation
4. **`example_llm_switching.py`** - Usage examples

## Modified Files

1. **`services/agent_factory.py`** - Now accepts generic `BaseChatModel` instead of just `AzureChatOpenAI`
2. **`services/tools_manager.py`** - Updated to use `BaseChatModel`
3. **`services/__init__.py`** - Exports `LLMFactory`
4. **`main.py`** - Uses `LLMFactory` to create LLMs based on config

## Quick Start

### 1. Edit `llm_config.json` to choose your provider:

```json
{
  "provider": "openai", // Change to "azure" or "openai"
  "azure": {
    "model": "gpt-4.1",
    "temperature": 0.7
  },
  "openai": {
    "model": "gpt-4",
    "temperature": 0.7
  }
}
```

### 2. Set environment variables:

**For Azure:**

```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

**For OpenAI:**

```bash
OPENAI_API_KEY=your_key
```

### 3. Run your app - it automatically uses the configured provider:

```python
from main import AgentApp

app = AgentApp()  # Uses provider from config
result = app.run("Your question")
```

## Usage Methods

### Method 1: Config File (Recommended)

Edit `llm_config.json` and set the `provider` field.

### Method 2: Runtime Override

```python
app = AgentApp(llm_provider="openai")  # Override config
```

### Method 3: Parameter Override

```python
app = AgentApp(
    llm_provider="azure",
    model="gpt-4-turbo",
    temperature=0.9
)
```

### Method 4: Direct Factory Usage

```python
from services import LLMFactory

llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4",
    temperature=0.7
)
```

## Benefits

✅ **Easy switching** between Azure and OpenAI  
✅ **Configuration-driven** - no code changes needed  
✅ **Flexible** - override at runtime when needed  
✅ **Type-safe** - all LLMs use `BaseChatModel` interface  
✅ **Backward compatible** - existing code still works

## Configuration Options

### Azure OpenAI

- `model` - Azure deployment name
- `temperature`, `max_tokens`, `timeout`, `max_retries`
- `api_version`, `azure_endpoint`, `api_key`

### OpenAI

- `model` - OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
- `temperature`, `max_tokens`, `timeout`, `max_retries`
- `base_url`, `api_key`, `organization`

### Common Parameters

- `verbose`, `top_p`, `frequency_penalty`, `presence_penalty`

## Examples

See `example_llm_switching.py` for complete examples including:

- Using config file
- Overriding provider
- Overriding parameters
- Direct LLM factory usage
- Dynamic provider switching

## Documentation

Full documentation available in [`docs/llm_configuration_guide.md`](docs/llm_configuration_guide.md)

## Testing

Test your configuration:

```python
from services import LLMFactory

# Check available providers
print(LLMFactory.get_available_providers())  # ['azure', 'openai']

# Test creating an LLM
llm = LLMFactory.create_llm()
response = llm.invoke("Hello!")
print(response.content)
```

## Migration Notes

**Before:**

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(model="gpt-4", temperature=0.7)
```

**After:**

```python
from services import LLMFactory

llm = LLMFactory.create_llm()  # Provider from config
# or
llm = LLMFactory.create_llm(provider="openai", model="gpt-4")
```

---

**Need help?** Check the [complete guide](docs/llm_configuration_guide.md) or see [examples](example_llm_switching.py).
