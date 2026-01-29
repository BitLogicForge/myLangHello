# LLM Configuration Architecture

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                         │
│                         (main.py / api.py)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AgentApp                                │
│  - Reads llm_config.json                                        │
│  - Accepts runtime overrides (llm_provider, model, etc.)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLMFactory                                │
│  - load_config()         - Loads llm_config.json                │
│  - create_llm()          - Creates appropriate LLM instance     │
│  - _create_azure_llm()   - Creates AzureChatOpenAI             │
│  - _create_openai_llm()  - Creates ChatOpenAI                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   AzureChatOpenAI       │   │     ChatOpenAI          │
│   (provider: "azure")   │   │   (provider: "openai")  │
└─────────────────────────┘   └─────────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ┌────────────────┐
                    │ BaseChatModel  │
                    │  (Interface)   │
                    └────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│    AgentFactory         │   │    ToolsManager         │
│  (accepts BaseChatModel)│   │  (accepts BaseChatModel)│
└─────────────────────────┘   └─────────────────────────┘
```

## Configuration Sources Priority

```
1. Runtime Parameters (highest priority)
   app = AgentApp(llm_provider="openai", model="gpt-4")
                 ↓

2. Configuration File
   llm_config.json: { "provider": "azure", ... }
                 ↓

3. Environment Variables
   AZURE_OPENAI_API_KEY, OPENAI_API_KEY, etc.
                 ↓

4. LangChain Defaults (lowest priority)
```

## Example: Creating an LLM

### Scenario 1: Using Config File

```python
# llm_config.json: { "provider": "azure", "azure": { "model": "gpt-4" } }
app = AgentApp()
# Result: AzureChatOpenAI with model="gpt-4"
```

### Scenario 2: Runtime Override

```python
# llm_config.json: { "provider": "azure", ... }
app = AgentApp(llm_provider="openai")
# Result: ChatOpenAI (config overridden)
```

### Scenario 3: Parameter Override

```python
# llm_config.json: { "provider": "azure", "azure": { "model": "gpt-4", "temperature": 0.7 } }
app = AgentApp(temperature=0.9)
# Result: AzureChatOpenAI with model="gpt-4", temperature=0.9 (merged)
```

## Component Responsibilities

| Component           | Responsibility                                             |
| ------------------- | ---------------------------------------------------------- |
| **llm_config.json** | Store provider settings, model configs, parameters         |
| **LLMFactory**      | Read config, create LLM instances, handle environment vars |
| **AgentApp**        | Orchestrate components, accept user overrides              |
| **AgentFactory**    | Create LangGraph agent (provider-agnostic)                 |
| **ToolsManager**    | Register tools with LLM (provider-agnostic)                |

## File Structure

```
myLangHello/
├── llm_config.json              ← Main configuration file
├── main.py                       ← Uses LLMFactory
├── api.py                        ← FastAPI wrapper
├── example_llm_switching.py     ← Usage examples
├── services/
│   ├── llm_factory.py           ← NEW: Creates LLM instances
│   ├── agent_factory.py         ← Updated: Accepts BaseChatModel
│   ├── tools_manager.py         ← Updated: Accepts BaseChatModel
│   ├── database_manager.py
│   └── prompt_builder.py
└── docs/
    ├── llm_configuration_guide.md  ← Complete guide
    └── ...
```

## Environment Variables Reference

### Azure OpenAI (Required)

```bash
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # optional
```

### OpenAI (Required)

```bash
OPENAI_API_KEY=sk-...
OPENAI_ORGANIZATION=org-...      # optional
OPENAI_BASE_URL=https://...      # optional
```

### Database (Existing)

```bash
DATABASE_URI=mssql+pyodbc://...
```

## Type Hierarchy

```
BaseChatModel (langchain_core)
    │
    ├── AzureChatOpenAI (langchain_openai)
    │   └── Used when provider="azure"
    │
    └── ChatOpenAI (langchain_openai)
        └── Used when provider="openai"

All components accept BaseChatModel for flexibility!
```

## Quick Reference

| Task               | Code                                            |
| ------------------ | ----------------------------------------------- |
| Use config default | `app = AgentApp()`                              |
| Override provider  | `app = AgentApp(llm_provider="openai")`         |
| Override model     | `app = AgentApp(model="gpt-4-turbo")`           |
| Override temp      | `app = AgentApp(temperature=0.9)`               |
| Direct factory     | `llm = LLMFactory.create_llm(provider="azure")` |
| Check providers    | `LLMFactory.get_available_providers()`          |

---

For detailed documentation, see: [`docs/llm_configuration_guide.md`](docs/llm_configuration_guide.md)
