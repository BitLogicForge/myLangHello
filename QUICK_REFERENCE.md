# 📋 Quick Reference Card - Stock Chatbot Backend

**Ultra-short concept lists for rapid navigation.**

---

## 🎯 5-SECOND NAVIGATION

| Want to... | Go to | Line | Command |
|-------------|------|------|--------|
| **Start server** | `api.py:346` | `uv run python api.py` |
| **Add tool** | `tools.py:623` | Copy existing `@tool` |
| **Change provider** | `config.json:2` | Set `"provider": "openai"` |
| **View docs** | `http://localhost:8000/docs` | Browser |
| **Run demo** | `examples/stock_analysis_demo.py:201` | `uv run python examples/stock_analysis_demo.py --demo quick` |

---

## 🤖 AGENT SYSTEM (3 components)

```
AgentApp → AgentConfigurator → AgentFactory → AgentRunner
    ↓              ↓                  ↓              ↓
  Orchestrates   Builds Agent     Creates Agent    Executes Safely
```

**Files:**
- `main.py:45` - AgentApp orchestrator
- `services/agent_configurator.py:29` - Step-by-step builder  
- `services/agent_factory.py:54` - LangGraph creator
- `services/agent_runner.py:19` - Safety limits

---

## 🛠️ TOOLS BY CATEGORY (20 total)

### **📊 Stock Data (3 tools)**
- `stock_price_query` - Current prices, OHLC data
- `market_data` - Indices, sectors, movers
- `stock_comparison` - Multi-stock comparison

### **📈 Analysis (3 tools)**
- `financial_metrics_calculator` - P/E, EPS, market cap
- `technical_analysis` - RSI, MACD, Bollinger Bands
- `risk_assessment` - Volatility, VaR, beta analysis

### **💼 Portfolio (3 tools)**
- `portfolio_create` - Create investment portfolio
- `portfolio_add_stock` - Add positions
- `portfolio_analyze` - Performance analysis

### **⚙️ Utilities (11 tools)**
- `calculator` - Math expressions
- `weather` - Demo weather data
- `read_file` / `write_file` - File I/O
- `current_date` - Date/time utilities
- `http_get` - HTTP requests
- `currency_converter` - Currency exchange
- `loan_calculator` - Loan payments
- `city_to_coordinates` - Location data
- `random_joke` / `joke_format` - Demo tools

**Location:** `tools.py:623-1477`

---

## 📊 STRUCTURED OUTPUT (6 endpoints)

### **Endpoints**
```
/structured/query                    → General structured queries
/structured/stock-analysis          → Stock analysis schema
/structured/portfolio-analysis      → Portfolio performance
/structured/market-overview         → Market data
/structured/stock-comparison        → Multi-stock comparison  
/structured/investment-recommendation → Investment advice
```

### **Schemas (models/structured_models.py)**
- `StockAnalysisResponse` - Complete stock analysis
- `PortfolioAnalysisResponse` - Portfolio performance
- `MarketOverviewResponse` - Market indices/sectors
- `StockComparisonResponse` - Scoring & ranking
- `InvestmentRecommendation` - Action items with confidence

### **Reliability**
| Provider | Method | Success |
|----------|--------|--------|
| Ollama (current) | Prompt-based | 40-60% |
| OpenAI GPT-3.5 | Prompt-based | 85-90% |
| OpenAI GPT-4o | **Native** | **95-98%** |

---

## 🚀 SCALABILITY (4 levels)

### **Level 1: File-Based** (Current default)
- **Capacity**: ~10 concurrent users
- **Response Time**: ~100ms
- **Cost**: Free
- **Use Case**: Development

### **Level 2: Memory Cache** 
- **Capacity**: ~50 concurrent users
- **Response Time**: ~1ms
- **Cost**: Free
- **File**: `services/portfolio_cache_simple.py`

### **Level 3: Redis Cache**
- **Capacity**: ~1000+ concurrent users  
- **Response Time**: ~2ms
- **Cost**: $5-50/month
- **File**: `services/portfolio_cache_redis.py`

### **Level 4: Multi-Tier** (Production)
- **Capacity**: ~1000+ users
- **Response Time**: ~1ms
- **Cost**: $5-50/month
- **Features**: Redis → Memory → File fallback
- **File**: `services/portfolio_manager_service.py`

---

## 🎯 CONFIGURATION (3 files)

### **config.json** (Provider selection)
```json
{
  "provider": "ollama",        // ← Change AI provider
  "ollama": {
    "model": "gemma4:e4b",     // ← Change model
    "temperature": 0.7         // ← Creativity
  },
  "agent": {
    "enable_db": false,       // ← Database tools
    "max_iterations": 10      // ← Reasoning depth
  }
}
```

### **.env** (API keys - not in git)
```bash
OPENAI_API_KEY="sk-your-key"    # ← Add for OpenAI
DB_HOST="192.168.0.70"         # ← Database server
OLLAMA_BASE_URL="..."        # ← Ollama server
```

### **messages/system_prompt.txt** (AI personality)
- Line 1-500: Agent instructions and behavior
- Line 501+: Stock-specific guidance and disclaimers

---

## 🧪 TESTING (3 methods)

### **1. Manual Testing**
```bash
# Start server
uv run python api.py

# Interactive docs  
http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

### **2. Demo Scripts**
```bash
# Quick tools demo
uv run python examples/stock_analysis_demo.py --demo quick

# Advanced workflows  
uv run python examples/advanced_stock_workflow_demo.py --workflow portfolio

# Interactive conversation
uv run python examples/interactive_conversation_demo.py --conversation natural

# Structured output
uv run python examples/structured_query_demo.py
```

### **3. Unit Tests**
```bash
# Run all tests
uv run pytest

# Specific test
uv run pytest tests/test_tools.py

# With verbose output
uv run pytest -v
```

---

## 🔧 TROUBLESHOOTING (Common issues)

| Problem | Check | Solution |
|---------|-------|----------|
| **Agent not working** | `config.json`, `.env` | Check provider, API keys |
| **Tools missing** | `services/tools_manager.py:50` | Tool registration |
| **JSON fails** | `routes/structured_query_routes.py:244` | Parser logic |
| **Caching broken** | `services/portfolio_*.py` | Cache configuration |
| **Server won't start** | `api.py:346` | Port conflicts, dependencies |

---

## 📈 PERFORMANCE COMPARISON

| Setup | Users | Reliability | Cost/month |
|-------|-------|-------------|------------|
| **Ollama (current)** | ~10 | 40-60% | $0 |
| **Memory cache** | ~50 | 60-75% | $0 |
| **Redis** | ~1000+ | 85-90% | $5-50 |
| **Multi-tier** | ~1000+ | 95-98% (OpenAI) | $30-50 |

---

## 🎯 KEY CONCEPTS (5-second overview)

### **🤖 Agent Lifecycle**
```
Config → LLM Factory → Agent Builder → Agent Runner → Response
  ↓         ↓            ↓              ↓            ↓
Settings  Provider    Tools Added    Safety Limits   JSON/Text
```

### **🛠️ Tool System**
```
20 Tools = 11 Stock + 9 Utility
   ↓
Tool Manager Registers All
   ↓
Agent Calls Tools as Needed
   ↓
Results Compiled into Response
```

### **📊 Dual API Modes**
```
Conversational Mode (/query):
User → Agent → Text Response → Parsing → Fallback

Structured Mode (/structured/*):  
User → Agent → Enhanced Prompt → JSON → Validation → Response
```

### **🚀 Scaling Strategy**
```
Start: File-based (10 users)
  ↓ Add: Memory cache (50 users)  
  ↓ Add: Redis (1000+ users)
  ↓ Optimize: Multi-tier fallback
```

### **🎯 Production Path**
```
Development (Ollama): 40-60% reliability
  ↓ Switch Provider: OpenAI GPT-3.5 (85-90%)
  ↓ Upgrade Model: OpenAI GPT-4o (95-98%)
  ↓ Native Output: with_structured_output()
```

---

## 📚 FILE LOCATIONS (Quick find)

**Core System:**
- Main app: `api.py:346`
- Agent orchestrator: `main.py:45`
- Configuration: `config.py`

**AI Brains:**
- Agent building: `services/agent_configurator.py:29`
- Agent creation: `services/agent_factory.py:54`
- Execution: `services/agent_runner.py:19`
- Provider selection: `services/llm_factory.py:25`

**Tools:**
- All 20 tools: `tools.py:623-1477`
- Registration: `services/tools_manager.py:50`

**API:**
- Conversational: `routes/agent_routes.py:45`
- Structured: `routes/structured_query_routes.py:68`
- OpenAI native: `routes/openai_structured_routes.py:1`

**Examples:**
- Quick demos: `examples/stock_analysis_demo.py:201`
- Workflows: `examples/advanced_stock_workflow_demo.py:29`
- Conversation: `examples/interactive_conversation_demo.py:128`
- Structured: `examples/structured_query_demo.py:1`

**Models:**
- Structured schemas: `models/structured_models.py:1`
- API models: `models/api_models.py:1`

---

**💡 Use this guide to navigate the 4000+ line codebase in seconds, not hours!**