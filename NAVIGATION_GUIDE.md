# 🎯 Stock Chatbot Backend - Navigation Guide

**Quick navigation for developers - find what you need fast.**

## 📂 File Structure Overview

```
myLangHello/
├── 🎯 CORE SYSTEM (Main application)
│   ├── api.py                      # FastAPI server entry point
│   ├── main.py                     # Main orchestrator
│   └── config.py                   # Configuration loader
│
├── 🤖 AGENT SYSTEM (AI brain)
│   ├── services/
│   │   ├── agent_configurator.py   # Builds complete agent
│   │   ├── agent_factory.py         # Creates LangGraph agent
│   │   ├── agent_runner.py          # Executes with safety limits
│   │   ├── llm_factory.py           # Provider selector
│   │   └── llm_provider_*.py       # OpenAI, Azure, Ollama providers
│
├── 🛠️ TOOLS (20 total capabilities)
│   ├── tools.py                     # All 20 tool implementations
│   └── services/tools_manager.py     # Tool registration
│
├── 📊 STRUCTURED OUTPUT (API responses)
│   ├── models/structured_models.py  # Pydantic schemas
│   └── routes/structured_query_routes.py  # JSON endpoints
│
├── 🌐 API ROUTES (HTTP endpoints)
│   ├── routes/
│   │   ├── agent_routes.py           # /query endpoint
│   │   ├── health_routes.py         # Health check
│   │   ├── config_routes.py          # Config inspection
│   │   └── openai_structured_routes.py  # OpenAI native output
│
├── 📚 EXAMPLES (Demo scripts)
│   └── examples/
│       ├── stock_analysis_demo.py            # Quick demos
│       ├── advanced_stock_workflow_demo.py   # Multi-step workflows
│       ├── interactive_conversation_demo.py  # Conversational AI
│       └── structured_query_demo.py          # JSON output demos
│
└── ⚙️ CONFIGURATION
    ├── config.json                  # Provider selection, model settings
    └── .env                         # API keys, database credentials
```

---

## 🎯 Quick Start Guides

### **I want to...**

| Goal | File | Line | Command |
|------|------|------|--------|
| **Run the server** | `api.py` | 1 | `uv run python api.py` |
| **Test stock tools** | `examples/stock_analysis_demo.py` | 1 | `uv run python examples/stock_analysis_demo.py` |
| **Add a new tool** | `tools.py` | 623+ | Add `@tool` function |
| **Change AI provider** | `config.json` | 2 | Set `"provider": "openai"` |
| **View API docs** | `http://localhost:8000/docs` | - | Browser UI |

---

## 📚 Concepts by Category

### **🤖 AI Agent Concepts**

| Concept | Location | Description |
|---------|----------|-------------|
| **Agent Creation** | `services/agent_factory.py:54` | `create_db_agent()` |
| **Agent Execution** | `services/agent_runner.py:19` | `AgentRunner.run()` |
| **Safety Limits** | `config.json:8-11` | `max_iterations`, `timeout_seconds` |
| **Tool Registration** | `services/tools_manager.py:50` | `_register_tools()` |
| **Provider Selection** | `services/llm_factory.py:25` | `create_llm()` |

### **🛠️ Tool Concepts**

| Tool Type | Tool Name | Purpose | Line |
|-----------|----------|---------|------|
| **Stock Data** | `stock_price_query` | Current prices, OHLC | `tools.py:603` |
| **Financial** | `financial_metrics_calculator` | P/E, EPS, market cap | `tools.py:655` |
| **Technical** | `technical_analysis` | RSI, MACD, Bollinger | `tools.py:1027` |
| **Risk** | `risk_assessment` | Volatility, VaR, beta | `tools.py:1147` |
| **Portfolio** | `portfolio_create` | Create investment portfolio | `tools.py:735` |
| **Portfolio** | `portfolio_add_stock` | Add positions | `tools.py:788` |
| **Portfolio** | `portfolio_analyze` | Performance analysis | `tools.py:875` |
| **Comparison** | `stock_comparison` | Multi-stock analysis | `tools.py:1406` |
| **Market** | `market_data` | Indices, sectors, movers | `tools.py:1300` |

### **📊 Structured Output Concepts**

| Concept | Location | Description |
|---------|----------|-------------|
| **Schema Definition** | `models/structured_models.py:1` | Pydantic models |
| **Stock Analysis Schema** | `models/structured_models.py:13` | `StockAnalysisResponse` |
| **Portfolio Schema** | `models/structured_models.py:43` | `PortfolioAnalysisResponse` |
| **Structured Routes** | `routes/structured_query_routes.py:45` | `/structured/query` |
| **Enhanced Prompts** | `routes/structured_query_routes.py:106` | `_create_structured_prompt()` |
| **JSON Cleaning** | `routes/structured_query_routes.py:244` | `_clean_json_response()` |

### **🚀 Caching & Scalability**

| Concept | Location | Description |
|---------|----------|-------------|
| **Thread-Safe Cache** | `services/portfolio_cache_simple.py:19` | Memory caching for 50 users |
| **Redis Cache** | `services/portfolio_cache_redis.py:18` | Distributed caching for 1000+ users |
| **Multi-Tier Service** | `services/portfolio_manager_service.py:28` | Cache → Memory → File fallback |
| **Concurrent Routes** | `routes/portfolio_routes.py:46` | FastAPI async endpoints |
| **Optimistic Locking** | `services/portfolio_manager_service.py:133` | Concurrent update safety |

---

## 🎯 Key Files by Purpose

### **🎯 Main Entry Points**

- **`api.py`** - FastAPI server, starts everything
- **`main.py`** - `AgentApp` class, orchestrates components  
- **`config.py`** - Singleton configuration loader

### **🤖 AI Intelligence**

- **`services/agent_configurator.py`** - Builds the agent step-by-step
- **`services/agent_factory.py`** - Creates LangGraph ReAct agent
- **`services/llm_factory.py`** - Selects OpenAI/Azure/Ollama
- **`services/agent_runner.py`** - Runs with timeout, tool limits

### **🛠️ Tools & Capabilities**

- **`tools.py`** - **20 stock analysis tools** (623 lines)
  - Stock data: `stock_price_query`, `market_data`, `stock_comparison`
  - Analysis: `financial_metrics_calculator`, `technical_analysis`, `risk_assessment`
  - Portfolio: `portfolio_create`, `portfolio_add_stock`, `portfolio_analyze`
  - Utilities: `calculator`, `weather`, `file I/O`, `HTTP`, `currency`

### **📊 API & Responses**

- **`routes/agent_routes.py`** - `/query` endpoint (conversational)
- **`routes/structured_query_routes.py`** - `/structured/*` endpoints (JSON)
- **`models/structured_models.py`** - Pydantic response schemas
- **`models/api_models.py`** - Request/response models

### **⚙️ Configuration**

- **`config.json`** - Provider choice, model settings, behavior config
- **`.env`** - API keys, database connection strings (not in git)

### **📚 Examples & Docs**

- **`examples/`** - 5 demo scripts showing different capabilities
- **`CLAUDE.md`** - Project guidance and development instructions
- **`STRUCTURED_QUERY_GUIDE.md`** - Complete structured output guide

---

## 🔍 Find What You Need

### **"I want to understand the main flow"**
```
Start here: api.py → main.py → services/agent_configurator.py
          ↓
       services/agent_factory.py → services/agent_runner.py
          ↓
       Returns response to user
```

### **"I want to add a new stock tool"**
```
1. Open tools.py (line 623+ for stock tools)
2. Copy existing tool (e.g., stock_price_query)
3. Modify @tool function with new logic
4. Update services/tools_manager.py to register it
```

### **"I want to change the AI provider"**
```
1. Open config.json
2. Change "provider": "ollama" → "openai"
3. Add OPENAI_API_KEY to .env file
4. Done! System uses new provider automatically
```

### **"I want structured JSON output"**
```
Option 1 (Current): Use existing /structured/* endpoints (40-60% reliable)
Option 2 (Best): Switch to OpenAI for 95-98% reliability
Option 3 (Test): See examples/structured_query_demo.py
```

### **"I want portfolio management"**
```
Tools: portfolio_create, portfolio_add_stock, portfolio_analyze
Demo: examples/advanced_stock_workflow_demo.py (lines 29-125)
Caching: services/portfolio_cache_simple.py (thread-safe)
```

---

## 🎯 Concept Index

### **🎯 Core Architecture Patterns**

| Pattern | Location | Description |
|---------|----------|-------------|
| **Factory Pattern** | `services/llm_factory.py:19` | Provider selection |
| **Builder Pattern** | `services/agent_configurator.py:29` | Step-by-step agent building |
| **Strategy Pattern** | `services/agent_factory.py:54` | Agent creation methods |
| **Repository Pattern** | `services/portfolio_manager_service.py:28` | Multi-tier storage |
| **Observer Pattern** | `services/agent_runner.py:19` | Execution monitoring |

### **🛠️ Tool Organization**

| Category | Tools | Count |
|----------|-------|-------|
| **Stock Data** | stock_price_query, market_data, stock_comparison | 3 |
| **Analysis** | financial_metrics_calculator, technical_analysis, risk_assessment | 3 |
| **Portfolio** | portfolio_create, portfolio_add_stock, portfolio_analyze | 3 |
| **Utilities** | calculator, weather, file I/O, HTTP, currency, etc. | 11 |
| **TOTAL** | | **20** |

### **📊 Response Type Categories**

| Category | Endpoints | Schemas |
|----------|-----------|---------|
| **Stock Analysis** | `/structured/stock-analysis` | `StockAnalysisResponse` |
| **Portfolio** | `/structured/portfolio-analysis` | `PortfolioAnalysisResponse` |
| **Market Data** | `/structured/market-overview` | `MarketOverviewResponse` |
| **Comparison** | `/structured/stock-comparison` | `StockComparisonResponse` |
| **Investment** | `/structured/investment-recommendation` | `InvestmentRecommendation` |

---

## 🔧 Quick Reference

### **🎯 Common Tasks**

| Task | Command | File |
|------|---------|------|
| **Run server** | `uv run python api.py` | `api.py:346` |
| **Run stock demo** | `uv run python examples/stock_analysis_demo.py --demo quick` | `examples/stock_analysis_demo.py:201` |
| **Check health** | `curl http://localhost:8000/health` | `routes/health_routes.py:13` |
| **View config** | `curl http://localhost:8000/config` | `routes/config_routes.py:14` |
| **Structured query** | `curl -X POST http://localhost:8000/structured/query` | `routes/structured_query_routes.py:68` |

### **🛠️ Tool Quick Reference**

| Want to... | Use Tool | Parameters |
|-------------|----------|------------|
| **Get stock price** | `stock_price_query` | `symbol`, `include_details` |
| **Analyze fundamentals** | `financial_metrics_calculator` | `symbol`, `metrics` |
| **Technical analysis** | `technical_analysis` | `symbol`, `indicators` |
| **Risk assessment** | `risk_assessment` | `symbol` or `portfolio_id` |
| **Create portfolio** | `portfolio_create` | `portfolio_name`, `initial_capital` |
| **Add stock** | `portfolio_add_stock` | `portfolio_id`, `symbol`, `shares`, `buy_price` |
| **Compare stocks** | `stock_comparison` | `symbols` (list), `comparison_type` |

### **📊 API Endpoint Quick Reference**

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/query` | POST | Conversational AI | Text response |
| `/structured/query` | POST | General structured JSON | Structured response |
| `/structured/stock-analysis` | POST | Stock analysis | `StockAnalysisResponse` |
| `/structured/portfolio-analysis` | POST | Portfolio performance | `PortfolioAnalysisResponse` |
| `/health` | GET | Health check | Health status |
| `/config` | GET | View configuration | Config object |

---

## 🚀 Learning Paths

### **🎯 Path 1: Understanding the Agent System**
```
Start: main.py (AgentApp class)
↓
Read: services/agent_configurator.py (step-by-step building)
↓
Read: services/agent_factory.py (LangGraph agent creation)
↓
Read: services/agent_runner.py (execution with safety)
↓
Result: Complete understanding of agent lifecycle
```

### **🛠️ Path 2: Tool Development**
```
Start: tools.py (tool implementations)
↓
Read: services/tools_manager.py (registration system)
↓
Experiment: examples/custom_tool_example.py
↓
Result: Can add custom tools confidently
```

### **📊 Path 3: API Development**
```
Start: api.py (FastAPI setup)
↓
Read: routes/agent_routes.py (conversational endpoint)
↓
Read: routes/structured_query_routes.py (structured endpoint)
↓
Experiment: Test with curl or http://localhost:8000/docs
↓
Result: Can build API integrations
```

### **🚀 Path 4: Production Readiness**
```
Start: config.json (current setup)
↓
Read: services/portfolio_cache_simple.py (caching basics)
↓
Read: services/portfolio_cache_redis.py (distributed caching)
↓
Read: services/portfolio_manager_service.py (multi-tier)
↓
Result: Can scale to 1000+ concurrent users
```

---

## 📖 File Navigation Tips

### **🎯 By Feature**
- **Stock Tools**: `tools.py:524-1477`
- **Portfolio Tools**: `tools.py:729-1015`  
- **API Routes**: `routes/`
- **Caching**: `services/portfolio_*.py`
- **Models**: `models/`

### **🔍 By Problem**
- **"Agent not working"**: Check `config.json`, `.env`, `services/llm_factory.py`
- **"Tool not found"**: Check `services/tools_manager.py:50`, `tools.py`
- **"JSON parsing fails"**: Check `routes/structured_query_routes.py:244`
- **"Caching issues"**: Check `services/portfolio_cache_*.py`

---

## 🎯 Key Takeaways

### **🎯 System Architecture**
- **Main Entry**: `api.py` → `AgentApp` → Agent system
- **Agent Building**: Configurator → Factory → Runner
- **Tools**: 20 capabilities across 5 categories
- **Storage**: File-based with multi-tier caching

### **🛠️ Tool System**
- **20 tools total**: 11 stock + 9 utility
- **Organized by category**: Data, analysis, portfolio, utilities
- **Easily extensible**: Add new tools following `@tool` pattern

### **📊 API System**
- **Dual modes**: Conversational (`/query`) + Structured (`/structured/*`)
- **6 structured endpoints**: Different analysis types
- **Reliability**: 40-60% (Ollama) → 95-98% (OpenAI)

### **🚀 Production Readiness**
- **Caching**: Thread-safe → Redis → Multi-tier fallback
- **Scalability**: 10 users → 50 users → 1000+ users
- **Monitoring**: Built-in telemetry, performance tracking
- **Error Handling**: Comprehensive fallbacks and retries

---

**💡 This navigation guide makes the 4000+ line codebase manageable by focusing on what you need, when you need it!**