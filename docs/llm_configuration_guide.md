# LLM Configuration Guide

This guide explains how to switch between different LLM providers (Azure OpenAI and OpenAI) using the configuration system.

## Overview

The application now supports easy switching between:

- **Azure OpenAI** (AzureChatOpenAI)
- **OpenAI** (ChatOpenAI)

Configuration is managed through `llm_config.json` and environment variables.

## Configuration File: `llm_config.json`

### Structure

```json
{
  "provider": "azure", // Default provider: "azure" or "openai"
  "azure": {
    // Azure-specific configuration
  },
  "openai": {
    // OpenAI-specific configuration
  },
  "common_params": {
    // Parameters shared by both providers
  }
}
```

### Provider-Specific Parameters

#### Azure Configuration

```json
"azure": {
  "model": "gpt-4.1",              // Azure deployment name
  "temperature": 0.7,               // 0-2, creativity level
  "max_tokens": null,               // Max response tokens
  "timeout": 120,                   // Request timeout in seconds
  "max_retries": 3,                 // Retry attempts
  "streaming": false,               // Enable streaming
  "api_version": "2024-02-15-preview",
  "azure_endpoint": null,           // Override env var
  "api_key": null                   // Override env var
}
```

#### OpenAI Configuration

```json
"openai": {
  "model": "gpt-4",                 // OpenAI model name
  "temperature": 0.7,
  "max_tokens": null,
  "timeout": 120,
  "max_retries": 3,
  "streaming": false,
  "base_url": null,                 // Custom API endpoint
  "api_key": null,                  // Override env var
  "organization": null              // OpenAI organization ID
}
```

#### Common Parameters

```json
"common_params": {
  "verbose": false,
  "top_p": null,                    // Nucleus sampling (0-1)
  "frequency_penalty": null,        // -2 to 2
  "presence_penalty": null          // -2 to 2
}
```

## Environment Variables

### For Azure OpenAI

```bash
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # optional
```

### For OpenAI

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORGANIZATION=your_org_id  # optional
OPENAI_BASE_URL=https://api.openai.com/v1  # optional
```

## Usage

### Method 1: Using Configuration File (Recommended)

1. **Edit `llm_config.json`** to set your provider:

```json
{
  "provider": "openai",  // Change to "openai" or "azure"
  ...
}
```

2. **Set environment variables** for your chosen provider

3. **Run your application** - it will automatically use the configured provider:

```python
from main import AgentApp

app = AgentApp()  # Uses provider from llm_config.json
result = app.run("Your question here")
```

### Method 2: Override Provider at Runtime

Override the provider when creating the app:

```python
from main import AgentApp

# Use OpenAI instead of config default
app = AgentApp(llm_provider="openai")

# Use Azure
app = AgentApp(llm_provider="azure")
```

### Method 3: Override Specific Parameters

Override specific LLM parameters:

```python
from main import AgentApp

app = AgentApp(
    model="gpt-4-turbo",      # Override model
    temperature=0.9,           # Override temperature
    llm_provider="openai"      # Override provider
)
```

### Method 4: Direct LLMFactory Usage

For more control, use LLMFactory directly:

```python
from services import LLMFactory

# Use config file
llm = LLMFactory.create_llm()

# Override provider
llm = LLMFactory.create_llm(provider="openai")

# Override parameters
llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4-turbo",
    temperature=0.8,
    max_tokens=2000
)
```

## Quick Switch Examples

### Switch from Azure to OpenAI

**Option A: Edit config file**

```json
{
  "provider": "openai",  // Change this line
  ...
}
```

**Option B: Runtime override**

```python
app = AgentApp(llm_provider="openai")
```

### Use Different Models

```python
# Azure with different deployment
app = AgentApp(model="gpt-35-turbo", llm_provider="azure")

# OpenAI with GPT-4 Turbo
app = AgentApp(model="gpt-4-turbo", llm_provider="openai")
```

### Different Temperature for Experimentation

```python
# More creative
app = AgentApp(temperature=1.2)

# More deterministic
app = AgentApp(temperature=0.0)
```

## API Endpoint Usage

If using the FastAPI endpoint (`api.py`), the application will use the provider configured in `llm_config.json` or environment variables when initialized.

## Troubleshooting

### Provider Not Found Error

- **Issue**: `ValueError: Unsupported provider: xyz`
- **Solution**: Ensure `provider` is either "azure" or "openai"

### Missing API Key Warning

- **Issue**: Warning about missing API keys
- **Solution**: Set appropriate environment variables (see above)

### Model Not Found

- **Azure**: Ensure the model name matches your Azure deployment name
- **OpenAI**: Use standard OpenAI model names (gpt-4, gpt-3.5-turbo, etc.)

## Best Practices

1. **Keep secrets in environment variables**, not in `llm_config.json`
2. **Use config file for non-sensitive settings** (model, temperature, etc.)
3. **Test both providers** in development with different configs
4. **Document your model choices** for team consistency
5. **Set appropriate timeouts** based on expected query complexity

## Available Providers

Check available providers programmatically:

```python
from services import LLMFactory

providers = LLMFactory.get_available_providers()
print(providers)  # ['azure', 'openai']
```

## Example Configurations

### Development (OpenAI)

```json
{
  "provider": "openai",
  "openai": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "timeout": 60
  }
}
```

### Production (Azure)

```json
{
  "provider": "azure",
  "azure": {
    "model": "gpt-4",
    "temperature": 0.5,
    "timeout": 120,
    "max_retries": 3
  }
}
```

### Streaming Mode

```json
{
  "provider": "openai",
  "openai": {
    "model": "gpt-4",
    "streaming": true
  }
}
```
