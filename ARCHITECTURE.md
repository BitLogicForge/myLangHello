# System Architecture

## High-Level Architecture Diagram

```mermaid
flowchart TB
    %% External Actors
    User([Business User])
    Browser[Web Browser/Client]

    %% Main Entry Points
    User -->|Asks Questions| Browser
    Browser -->|HTTP Requests| API[FastAPI Application<br/>api.py]

    %% API Layer
    API -->|Health Checks| Health[Health Routes<br/>Status & Diagnostics]
    API -->|Query Requests| Agent[Agent Routes<br/>Process Questions]
    API -->|Configuration| Config[Config Routes<br/>View Settings]

    %% Core Application
    Agent -->|Initializes| App[Agent Application<br/>main.py<br/>Core Orchestrator]

    %% Business Logic Layer
    App -->|Configures| Configurator[Agent Configurator<br/>Setup & Coordination]

    %% Component Setup
    Configurator -->|1. Creates| LLM[AI Language Model<br/>OpenAI/Azure]
    Configurator -->|2. Connects| DB[Database Manager<br/>SQL Server Connection]
    Configurator -->|3. Prepares| Tools[Tools Manager<br/>Available Actions]
    Configurator -->|4. Builds| Prompt[Prompt Builder<br/>Instructions for AI]
    Configurator -->|5. Assembles| AgentFactory[Agent Factory<br/>Creates AI Agent]

    %% External Resources
    LLM -->|API Calls| OpenAI[(OpenAI/Azure<br/>AI Services)]
    DB -->|SQL Queries| Database[(SQL Server<br/>Business Data)]

    %% Tools Available
    Tools -->|Can Use| ToolsList[Available Tools:<br/>• Database Queries<br/>• Calculator<br/>• File Operations<br/>• Weather Info<br/>• System Info<br/>• HTTP Requests]

    %% Response Flow
    AgentFactory -->|Creates| FinalAgent[AI Agent Executor<br/>Answers Questions]
    FinalAgent -->|Formatted Response| Browser

    %% Configuration Files
    ConfigFiles[(Configuration Files)]
    ConfigFiles -.->|llm_config.json<br/>DB settings<br/>System prompt| Configurator

    %% Monitoring
    API -->|Tracks| Metrics[Telemetry & Monitoring<br/>Performance Metrics]

    %% Styling
    classDef userClass fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef apiClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef coreClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef serviceClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef dataClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef toolClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px

    class User,Browser userClass
    class API,Health,Agent,Config apiClass
    class App,Configurator,AgentFactory,FinalAgent coreClass
    class LLM,DB,Tools,Prompt,ToolsList serviceClass
    class OpenAI,Database,ConfigFiles dataClass
    class Metrics toolClass
```

## Business-Friendly Overview

### What This System Does

This is an **AI-powered Question-Answering System** that helps business users get information by asking questions in natural language.

### Key Components Explained

#### 1. **User Interface Layer**

- **What**: Web browser or any HTTP client
- **Purpose**: Where users type their questions
- **Example**: "What were our sales last month?" or "Calculate 15% of 1000"

#### 2. **API Application (api.py)**

- **What**: The main gateway to the system
- **Purpose**: Receives questions, routes them to the right handlers, and sends back answers
- **Like**: A receptionist directing visitors to the right department

#### 3. **Core Orchestrator (main.py - AgentApp)**

- **What**: The brain coordinator
- **Purpose**: Sets up all the pieces and makes sure they work together
- **Like**: A project manager ensuring all teams are ready

#### 4. **AI Language Model**

- **What**: OpenAI or Azure AI service
- **Purpose**: Understands questions and generates intelligent responses
- **Like**: A knowledgeable expert who understands what you're asking

#### 5. **Database Connection**

- **What**: Connection to your SQL Server database
- **Purpose**: Retrieves business data when needed
- **Like**: A librarian who can fetch specific books (data) from the library (database)

#### 6. **Tools Manager**

- **What**: Collection of capabilities the AI can use
- **Purpose**: Provides various actions the AI can perform
- **Available Tools**:
  - Query database for business data
  - Perform calculations
  - Check weather
  - Read/write files
  - Make web requests
  - Get system information

#### 7. **Configuration**

- **What**: Settings files that control behavior
- **Purpose**: Define which AI to use, database location, and system behavior
- **Files**:
  - `llm_config.json` - AI model settings
  - `.env` - Database credentials and API keys
  - `db_schema_config.json` - Database structure information

### How It Works (Step by Step)

1. **User asks a question** through their web browser
2. **API receives the question** and routes it to the Agent Routes
3. **Agent Configurator sets up everything needed**:
   - Connects to the AI service (OpenAI/Azure)
   - Connects to the database
   - Prepares available tools
   - Loads instructions for the AI
4. **AI Agent processes the question**:
   - Understands what's being asked
   - Decides which tools to use (e.g., query database, calculate)
   - Executes actions as needed
   - Formulates a clear answer
5. **Response sent back to user** with the answer

### Example Workflow

**User Question**: "What are the top 5 products by sales?"

1. Browser sends question to API
2. API forwards to Agent Routes
3. Agent uses Database Tool to query sales data
4. AI formats the results into readable answer
5. User receives: "Top 5 products are: Product A ($50k), Product B ($45k)..."

### Monitoring & Health

- **Health Checks**: System can report if it's working properly
- **Telemetry**: Tracks performance metrics (response times, errors)
- **Logging**: Records what happened for troubleshooting

### Security & Configuration

- API keys stored in `.env` file (never in code)
- Database credentials secured
- CORS enabled for web browser access
- Configurable timeouts and limits

## Technical Stack Summary

- **Framework**: FastAPI (Python web framework)
- **AI**: LangChain with OpenAI/Azure models
- **Database**: SQL Server (MSSQL)
- **Streaming**: LangServe for real-time responses
- **Configuration**: JSON and environment files
- **Monitoring**: Built-in telemetry and health checks

---

## Modern Agent Architecture - State of the Art (SOTA)

### How the Agent Works: ReAct Pattern with LangGraph

```mermaid
flowchart TB
    Start([User Question]) --> Receive[Agent Receives Question]

    Receive --> Think1{THINK:<br/>What do I need?}

    Think1 -->|Need data| Act1[ACT:<br/>Call Database Tool]
    Think1 -->|Need calculation| Act2[ACT:<br/>Call Calculator Tool]
    Think1 -->|Have answer| Answer[Generate Final Answer]

    Act1 --> Observe1[OBSERVE:<br/>Got query results]
    Act2 --> Observe2[OBSERVE:<br/>Got calculation]

    Observe1 --> Think2{THINK:<br/>Is this enough?}
    Observe2 --> Think2

    Think2 -->|Need more info| Act3[ACT:<br/>Use Another Tool]
    Think2 -->|Complete| Answer

    Act3 --> Observe3[OBSERVE:<br/>New information]
    Observe3 --> Think2

    Answer --> Return([Return Answer to User])

    %% Styling
    classDef thinkClass fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    classDef actClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    classDef observeClass fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef finalClass fill:#f3e5f5,stroke:#4a148c,stroke-width:3px

    class Think1,Think2 thinkClass
    class Act1,Act2,Act3 actClass
    class Observe1,Observe2,Observe3 observeClass
    class Answer,Return finalClass
```

### Why This is State-of-the-Art (SOTA)

```mermaid
flowchart LR
    subgraph Old["❌ Old Approach (2022-2023)"]
        direction TB
        O1[Rule-Based Chatbots]
        O2[Fixed Response Trees]
        O3[No Tool Access]
        O4[Scripted Flows]

        O1 -.->|Limited| O2
        O2 -.->|Inflexible| O3
        O3 -.->|Can't adapt| O4
    end

    subgraph Modern["✅ Modern Approach (2024+)"]
        direction TB
        M1[LangGraph Framework]
        M2[ReAct Agent Pattern]
        M3[Function Calling]
        M4[Dynamic Reasoning]
        M5[Multi-Step Planning]
        M6[Memory & Context]

        M1 -->|Enables| M2
        M2 -->|Uses| M3
        M3 -->|Powers| M4
        M4 -->|Supports| M5
        M5 -->|With| M6
    end

    Old -.->|Evolution| Modern

    classDef oldStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef newStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px

    class O1,O2,O3,O4 oldStyle
    class M1,M2,M3,M4,M5,M6 newStyle
```

### Key SOTA Features Explained

#### 1. **ReAct Pattern (Reason + Act)**

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Tools
    participant LLM as AI Brain

    User->>Agent: "What are top 3 products by sales?"

    rect rgb(255, 249, 196)
    Agent->>LLM: THINK: What do I need?
    LLM-->>Agent: Need to query database
    end

    rect rgb(232, 245, 233)
    Agent->>Tools: ACT: sql_query_tool
    Tools->>Tools: Execute SQL Query
    Tools-->>Agent: OBSERVE: Results [Product A, B, C]
    end

    rect rgb(255, 249, 196)
    Agent->>LLM: THINK: Do I have enough?
    LLM-->>Agent: Yes, can answer now
    end

    rect rgb(243, 229, 245)
    Agent->>User: ANSWER: Top 3 products are...
    end
```

**Why Better:**

- 🧠 **Thinks before acting** (not random tool calls)
- 🔄 **Self-correcting** (learns from observations)
- 📊 **Multi-step reasoning** (can chain actions)
- ✅ **Validates results** (checks if answer is complete)

#### 2. **Native Function Calling**

**Traditional (Old):**

```
User: "Calculate 15% of sales"
Bot: "I don't understand. Try rephrasing."
```

**Modern (SOTA):**

```
User: "Calculate 15% of sales"
Agent:
  1. THINK: Need sales data, then calculate
  2. ACT: Query database → $10,000
  3. OBSERVE: Got $10,000
  4. THINK: Now calculate 15%
  5. ACT: calculator_tool(10000 * 0.15)
  6. OBSERVE: Result = $1,500
  7. ANSWER: "15% of sales is $1,500"
```

#### 3. **LangGraph State Machine**

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> Reasoning: Question received

    Reasoning --> ToolSelection: Needs data/action
    Reasoning --> FinalAnswer: Has enough info

    ToolSelection --> ExecuteTool: Choose tool
    ExecuteTool --> ObserveResult: Tool executed

    ObserveResult --> Reasoning: Process result

    FinalAnswer --> [*]: Return to user

    note right of Reasoning
        AI decides next step
        based on context
    end note

    note right of ToolSelection
        Intelligently picks
        the right tool
    end note
```

**Advantages:**

- ✅ **Predictable execution flow**
- ✅ **Easy to debug and visualize**
- ✅ **Can save/resume conversations**
- ✅ **Built-in error recovery**

#### 4. **Comparison: Traditional vs SOTA**

| Feature              | Traditional Chatbot | SOTA Agent (Our System)    |
| -------------------- | ------------------- | -------------------------- |
| **Intelligence**     | Pattern matching    | LLM reasoning              |
| **Data Access**      | Pre-programmed only | Real-time database queries |
| **Calculation**      | Fixed formulas      | Dynamic calculations       |
| **Adaptability**     | Must reprogram      | Learns from tools          |
| **Multi-step**       | Single response     | Chain of actions           |
| **Memory**           | No context          | Remembers conversation     |
| **Error Handling**   | Breaks easily       | Self-correcting            |
| **Tool Integration** | Hard-coded          | Plug-and-play tools        |

### Real-World Example

**Question:** "Compare last month's sales to this month and calculate the growth percentage"

**Agent Process:**

```mermaid
graph TD
    Q[Question Received] --> T1{Think: What's needed?}

    T1 --> A1[ACT: Query last month sales]
    A1 --> O1[OBSERVE: $50,000]

    O1 --> T2{Think: What else?}
    T2 --> A2[ACT: Query this month sales]
    A2 --> O2[OBSERVE: $65,000]

    O2 --> T3{Think: How to calculate?}
    T3 --> A3[ACT: Calculator<br/>65000-50000 / 50000 * 100]
    A3 --> O3[OBSERVE: 30%]

    O3 --> T4{Think: Complete?}
    T4 --> Final[ANSWER:<br/>This month: $65,000<br/>Last month: $50,000<br/>Growth: 30%]

    classDef think fill:#fff9c4,stroke:#f57f17
    classDef act fill:#e8f5e9,stroke:#1b5e20
    classDef observe fill:#e1f5ff,stroke:#01579b
    classDef final fill:#f3e5f5,stroke:#4a148c

    class T1,T2,T3,T4 think
    class A1,A2,A3 act
    class O1,O2,O3 observe
    class Final final
```

### Why This Matters for Business

#### Traditional Limitations:

- ❌ "I can only answer predefined questions"
- ❌ "I can't access your database"
- ❌ "I can't do calculations"
- ❌ "I forget previous messages"

#### SOTA Benefits:

- ✅ **Understands intent** - Natural language, any phrasing
- ✅ **Accesses live data** - Real-time database queries
- ✅ **Performs calculations** - Mathematical operations
- ✅ **Chains actions** - Multi-step problem solving
- ✅ **Remembers context** - Conversation history
- ✅ **Self-validates** - Checks if answer makes sense

### Technical Innovation Stack

```mermaid
graph TB
    subgraph "SOTA Technologies"
        LG[LangGraph]
        LC[LangChain]
        OAI[OpenAI GPT-4]
        FC[Function Calling]
    end

    subgraph "Our Implementation"
        RA[ReAct Agent]
        TM[Tool Manager]
        PM[Prompt Engineering]
        SM[State Management]
    end

    subgraph "Business Value"
        AUTO[Automation]
        ACC[Accuracy]
        SCALE[Scalability]
        SAVE[Cost Savings]
    end

    LG --> RA
    LC --> TM
    OAI --> PM
    FC --> TM

    RA --> AUTO
    TM --> ACC
    PM --> ACC
    SM --> SCALE

    AUTO --> SAVE
    ACC --> SAVE
    SCALE --> SAVE

    classDef tech fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef impl fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef value fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class LG,LC,OAI,FC tech
    class RA,TM,PM,SM impl
    class AUTO,ACC,SCALE,SAVE value
```

### Summary: Why This is Cutting-Edge

1. **2024+ Best Practices**: Uses latest LangGraph framework (released 2024)
2. **OpenAI Function Calling**: Native tool integration (most advanced capability)
3. **ReAct Pattern**: Research-proven reasoning method
4. **Production-Ready**: Used by companies like Microsoft, Google, Stripe
5. **Scalable Architecture**: Can handle millions of requests
6. **Extensible**: Easy to add new tools and capabilities

This isn't just a chatbot - it's an **intelligent agent that reasons, acts, and learns**.
