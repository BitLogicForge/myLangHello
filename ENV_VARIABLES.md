# Environment Variables

This document lists all expected environment variables for the application.

## Required Variables

### OpenAI Configuration

#### `OPENAI_API_KEY`

- **Required**: Yes (when using OpenAI provider)
- **Description**: API key for OpenAI services
- **Example**: `sk-proj-...`
- **Used by**: LLM Factory for OpenAI provider

### Database Configuration

#### `DB_HOST`

- **Required**: Yes (when using database tools)
- **Description**: Database server hostname or IP address
- **Example**: `localhost` or `192.168.1.100`
- **Used by**: Database Manager, Agent Configurator

#### `DB_NAME`

- **Required**: Yes (when using database tools)
- **Description**: Database name to connect to
- **Example**: `MyDatabase`
- **Used by**: Database Manager, Agent Configurator

#### `DB_USERNAME`

- **Required**: No (required for SQL Authentication, not needed for Windows Authentication)
- **Description**: Database username for SQL Authentication
- **Example**: `sa` or `dbuser`
- **Used by**: Database Manager, Agent Configurator

#### `DB_PASSWORD`

- **Required**: No (required for SQL Authentication, not needed for Windows Authentication)
- **Description**: Database password for SQL Authentication
- **Example**: `Password123`
- **Used by**: Database Manager, Agent Configurator

#### `DB_USE_WINDOWS_AUTH`

- **Required**: No
- **Description**: Use Windows Authentication instead of SQL Authentication
- **Values**: `true` or `false`
- **Default**: `false`
- **Used by**: Database Manager, Agent Configurator

#### `DB_DRIVER`

- **Required**: No
- **Description**: ODBC driver name for SQL Server
- **Default**: `ODBC Driver 17 for SQL Server`
- **Example**: `ODBC Driver 18 for SQL Server`
- **Used by**: Database Manager, Agent Configurator

## Optional Variables

### Azure OpenAI Configuration

#### `AZURE_OPENAI_API_KEY`

- **Required**: No (only when using Azure provider)
- **Description**: API key for Azure OpenAI services
- **Example**: `abc123xyz...`
- **Used by**: LLM Factory for Azure provider

#### `AZURE_OPENAI_ENDPOINT`

- **Required**: No (only when using Azure provider)
- **Description**: Azure OpenAI endpoint URL
- **Example**: `https://your-resource.openai.azure.com/`
- **Used by**: LLM Factory for Azure provider

#### `AZURE_OPENAI_API_VERSION`

- **Required**: No
- **Description**: Azure OpenAI API version
- **Default**: `2024-02-15-preview`
- **Example**: `2024-02-15-preview`
- **Used by**: LLM Factory for Azure provider

### OpenAI Advanced Configuration

#### `OPENAI_ORGANIZATION`

- **Required**: No
- **Description**: OpenAI organization ID
- **Example**: `org-...`
- **Used by**: LLM Factory for OpenAI provider

#### `OPENAI_BASE_URL`

- **Required**: No
- **Description**: Custom base URL for OpenAI API (for proxies or compatible services)
- **Example**: `https://api.openai.com/v1`
- **Used by**: LLM Factory for OpenAI provider

### Observability & Monitoring

#### `LANGCHAIN_TRACING_V2`

- **Required**: No
- **Description**: Enable LangChain tracing for debugging
- **Values**: `true` or `false`
- **Default**: Not enabled
- **Used by**: LangSmith integration

#### `LANGCHAIN_API_KEY`

- **Required**: No (only when using LangSmith)
- **Description**: API key for LangSmith tracing
- **Example**: `ls__...`
- **Used by**: LangSmith integration

#### `LANGCHAIN_PROJECT`

- **Required**: No
- **Description**: Project name for LangSmith tracing
- **Example**: `production-agent`
- **Default**: `default`
- **Used by**: LangSmith integration

### Application Configuration

#### `LOG_LEVEL`

- **Required**: No
- **Description**: Logging level for the application
- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default**: `INFO`
- **Used by**: Logging configuration

## .env File Template

```env
# Required - OpenAI API Key
OPENAI_API_KEY=sk-proj-your-api-key-here

# Required - Database Connection
DB_HOST=localhost
DB_NAME=YourDatabase

# Database Authentication (choose one method)
# Option 1: SQL Authentication
# DB_USERNAME=sa
# DB_PASSWORD=YourPassword123

# Option 2: Windows Authentication
DB_USE_WINDOWS_AUTH=true

# Optional - Database Driver
# DB_DRIVER=ODBC Driver 17 for SQL Server

# Optional - Azure OpenAI (if using Azure provider)
# AZURE_OPENAI_API_KEY=your-azure-api-key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional - OpenAI Advanced
# OPENAI_ORGANIZATION=org-your-org-id
# OPENAI_BASE_URL=https://api.openai.com/v1

# Optional - LangSmith Tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls__your-langsmith-key
# LANGCHAIN_PROJECT=my-project

# Optional - Logging
# LOG_LEVEL=INFO
```

## Notes

- Store sensitive values like API keys securely and never commit `.env` files to version control
- Add `.env` to your `.gitignore` file
- Use `.env.example` as a template without actual secrets
- For production, use environment variables set directly on your server/container
- Provider selection (OpenAI vs Azure) is configured in `config.json`
