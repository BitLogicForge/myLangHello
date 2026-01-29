# Quick Start: Switch LLM Provider in 3 Steps

## Switch to OpenAI

### Step 1: Edit `llm_config.json`

```json
{
  "provider": "openai",  // ← Change this to "openai"
  ...
}
```

### Step 2: Set Environment Variable

```bash
# Windows Command Prompt
set OPENAI_API_KEY=sk-your-openai-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"

# Or add to .env file
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Step 3: Run Your App

```python
from main import AgentApp

app = AgentApp()  # Automatically uses OpenAI now!
result = app.run("What is 2+2?")
```

✅ **Done!** Your app now uses OpenAI.

---

## Switch to Azure OpenAI

### Step 1: Edit `llm_config.json`

```json
{
  "provider": "azure",  // ← Change this to "azure"
  "azure": {
    "model": "gpt-4.1",  // ← Your Azure deployment name
    ...
  }
}
```

### Step 2: Set Environment Variables

```bash
# Windows Command Prompt
set AZURE_OPENAI_API_KEY=your-azure-api-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Windows PowerShell
$env:AZURE_OPENAI_API_KEY="your-azure-api-key"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Or add to .env file
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### Step 3: Run Your App

```python
from main import AgentApp

app = AgentApp()  # Automatically uses Azure OpenAI now!
result = app.run("What is 2+2?")
```

✅ **Done!** Your app now uses Azure OpenAI.

---

## Override Without Editing Config

Don't want to edit the config file? Override at runtime:

```python
from main import AgentApp

# Use OpenAI (ignoring config)
app = AgentApp(llm_provider="openai")

# Use Azure (ignoring config)
app = AgentApp(llm_provider="azure")

# Use specific model
app = AgentApp(llm_provider="openai", model="gpt-4-turbo")
```

---

## Test Your Configuration

```python
from services import LLMFactory

# Test creating LLM
llm = LLMFactory.create_llm()
response = llm.invoke("Hello, test!")
print(response.content)
```

---

## Troubleshooting

### Error: "AZURE_OPENAI_API_KEY not found"

**Fix:** Set the environment variable:

```bash
set AZURE_OPENAI_API_KEY=your-key
```

### Error: "OPENAI_API_KEY not found"

**Fix:** Set the environment variable:

```bash
set OPENAI_API_KEY=sk-your-key
```

### Error: "Model not found"

**Fix for Azure:** Use your **deployment name** (not model name):

```json
{
  "azure": {
    "model": "my-gpt4-deployment" // Your deployment name in Azure Portal
  }
}
```

**Fix for OpenAI:** Use standard model names:

```json
{
  "openai": {
    "model": "gpt-4" // or "gpt-3.5-turbo", "gpt-4-turbo"
  }
}
```

---

## Current Configuration Check

```python
from services import LLMFactory

# Load and check config
config = LLMFactory.load_config()
print(f"Current provider: {config['provider']}")
print(f"Available providers: {LLMFactory.get_available_providers()}")
```

---

## Complete Example

```python
# example_test.py
from dotenv import load_dotenv
from main import AgentApp

# Load .env file
load_dotenv()

# Create app (uses llm_config.json)
app = AgentApp()

# Test it
result = app.run("What's 10 + 15?")
print(result)

# Test with different provider
app2 = AgentApp(llm_provider="openai")
result2 = app2.run("Tell me a joke")
print(result2)
```

Run it:

```bash
python example_test.py
```

---

**That's it!** You can now easily switch between providers. 🎉

For more details, see:

- [Complete Guide](llm_configuration_guide.md)
- [Architecture](llm_architecture.md)
- [Examples](../example_llm_switching.py)
