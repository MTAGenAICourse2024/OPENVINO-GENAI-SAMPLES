# 🎯 Skills-Based Agentic AI Tutorial

## Composable AI Capabilities for Agentic Systems

This tutorial demonstrates the **Skills pattern** for building agentic AI systems. Skills are composable, stateful units of capability that can be chained together for complex tasks.

---

## 🎯 Learning Objectives

1. **Understand the Skills Pattern**
   - What makes Skills different from simple tools
   - How Skills maintain context and state
   - When to use Skills vs MCP Tools

2. **Learn Skill Composition**
   - How Skills can be chained together
   - Sharing context between Skills
   - Building complex behaviors from simple Skills

3. **Master Skill Design**
   - Creating reusable Skill classes
   - Confidence scoring for skill selection
   - Prerequisites and dependencies

---

## 🚀 Running the Demo

```bash
# Activate the virtual environment
.\venv_agentic_demo\Scripts\activate  # Windows
source venv_agentic_demo/bin/activate  # Linux/Mac

# Run the basic Skills app
streamlit run 08_skills_agentic_tutorial.py --server.port 8511

# Run the LangGraph + LLM version (recommended)
streamlit run 09_skills_agentic_tutorial_langgraph.py --server.port 8502
```

---

## ⚙️ Model Options (09_skills_agentic_tutorial_langgraph.py)

The LangGraph version supports multiple LLM backends:

| Model Option | Type | Description |
|-------------|------|-------------|
| **TinyLlama 1.1B (Local)** | OpenVINO | Local model using OpenVINO GenAI |
| **Azure OpenAI GPT-4o (Cloud)** | Azure OpenAI | Cloud-based GPT-4o via Azure |
| **Simulation Mode** | Rule-based | Demo mode without actual LLM |

### Azure OpenAI Configuration

To use Azure OpenAI, set these environment variables in your `.env` file:

```env
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_LLM_4="gpt-4o"
```

### Required Packages

```bash
pip install openai python-dotenv
```

---

## 🔄 Skills vs MCP Tools: A Comparison

| Feature | MCP Tools | Skills |
|---------|-----------|--------|
| **State** | Stateless | Stateful (maintains context) |
| **Composition** | Independent | Chainable |
| **Context Sharing** | Manual passing | Automatic via SkillContext |
| **Prerequisites** | Not built-in | Built-in support |
| **Confidence Scoring** | Not standard | Built-in |
| **Standard Protocol** | Yes (MCP spec) | No (pattern) |
| **Best For** | External integrations | Internal agent logic |

### When to Use Each:

**Use MCP Tools when:**
- Integrating with external services (Claude Desktop, VS Code, etc.)
- You need standardized tool schemas
- Tools are stateless and independent

**Use Skills when:**
- Building internal agent capabilities
- Skills need to share context
- You want composable, chainable operations
- Building complex multi-step workflows

---

## 🎯 Available Skills

| Skill | Category | Description | Example |
|-------|----------|-------------|---------|
| 🌤️ **Weather** | Information | Get weather data | "Weather in Tokyo" |
| 🧮 **Calculator** | Utility | Math operations | "Calculate 50 * 12" |
| 📊 **Text Analysis** | Analysis | Analyze text | "Analyze: I love this!" |
| 🧠 **Memory** | Memory | Store/recall info | "Remember my name is John" |
| 💡 **Recommendation** | Generation | Context-based suggestions | "Give me recommendations" |

---

## 🔗 Skill Chaining Example

One of the powerful features of Skills is **chaining** - using the output of one skill as context for another:

### Example Chain:

```
Step 1: "What's the weather in London?"
        ↓ (Weather Skill stores: current_weather, current_location)
        
Step 2: "Give me recommendations"
        ↓ (Recommendation Skill reads weather context)
        
Result: "☔ Bring an umbrella!" (based on weather data)
```

### Try It:
1. Ask about weather first
2. Then ask for recommendations
3. Notice how recommendations are weather-aware!

---

## 📦 The SkillContext System

Skills share information through `SkillContext`:

```python
@dataclass
class SkillContext:
    user_input: str = ""
    previous_results: List[Dict] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
```

### Context Flow:

```
┌──────────────┐
│ Weather Skill │
│              │
│ context.set_variable("current_weather", {...})
│ context.set_variable("current_location", "London")
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Recommendation│
│    Skill      │
│              │
│ weather = context.get_variable("current_weather")
│ # Uses weather data to make recommendations!
└──────────────┘
```

---

## 🏗️ Creating Custom Skills

### Basic Skill Template:

```python
class MyCustomSkill(Skill):
    def __init__(self):
        super().__init__(
            name="my_skill",
            description="What my skill does",
            category=SkillCategory.UTILITY
        )
        self.triggers = ["keyword1", "keyword2"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(t in context.user_input.lower() for t in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        # Your skill logic here
        result = do_something()
        
        # Store for chaining
        context.set_variable("my_result", result)
        
        return {"data": result}
```

### Skill with Prerequisites:

```python
class AnalysisReportSkill(Skill):
    def __init__(self):
        super().__init__(...)
        self.prerequisites = ["text_analysis"]  # Requires this skill first
```

---

## 🧪 Exercises

### Exercise 1: Skill Chaining
1. Ask about weather in any city
2. Analyze some text for sentiment
3. Ask for recommendations
4. Observe how recommendations use both contexts!

### Exercise 2: Memory Skills
1. Tell the agent to remember something
2. Ask what it remembers
3. Reset context and check memory persists (session-based)

### Exercise 3: Confidence Scoring
Try different phrasings and observe confidence scores:
- "weather in paris" (high confidence)
- "is it cold outside?" (medium confidence)
- "what's happening in london?" (low confidence)

---

## 🔧 Architecture Deep Dive

### Skill Orchestrator

The orchestrator manages skill lifecycle:

```python
class SkillOrchestrator:
    def process_input(self, user_input: str):
        # 1. Update context
        self.context.user_input = user_input
        
        # 2. Select best skill (confidence scoring)
        skill = self.select_skill(user_input)
        
        # 3. Check prerequisites
        # 4. Execute skill
        # 5. Post-process results
        # 6. Update context
        
        return result
```

### Skill Lifecycle:

```
┌─────────────┐
│ can_execute │ ─── Check if skill can handle input
└──────┬──────┘
       │ yes
       ▼
┌─────────────┐
│   execute   │ ─── Main skill logic
└──────┬──────┘
       │
       ▼
┌─────────────┐
│post_process │ ─── Optional cleanup/formatting
└─────────────┘
```

---

## 🎓 Course Integration

This tutorial complements the MCP Tutorial:

| Tutorial | Focus | Pattern |
|----------|-------|---------|
| 07_mcp_agentic_tutorial.py | External tool integration | MCP Protocol |
| 08_skills_agentic_tutorial.py | Internal agent logic (basic) | Skills Pattern |
| **09_skills_agentic_tutorial_langgraph.py** | LangGraph + LLM integration | Skills + LangGraph |

### LangGraph Version Features (09):

- **LangGraph Workflow**: Stateful graph-based orchestration
- **LLM Integration**: Actual LLM calls for intelligent decisions
- **Azure OpenAI Support**: Cloud-based GPT-4o integration
- **Local OpenVINO**: TinyLlama running locally
- **Conditional Routing**: LLM decides which skills to use

### Combined Architecture (Production):

```
┌─────────────────────────────────────────┐
│            AI Agent                      │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    │
│  │   Skills    │    │  MCP Tools  │    │
│  │ (Internal)  │    │ (External)  │    │
│  │             │    │             │    │
│  │ • Reasoning │    │ • APIs      │    │
│  │ • Memory    │    │ • Databases │    │
│  │ • Planning  │    │ • Services  │    │
│  └─────────────┘    └─────────────┘    │
└─────────────────────────────────────────┘
```

---

## 📖 Further Reading

- [LangChain Tools & Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT Architecture](https://github.com/Significant-Gravitas/AutoGPT)
- [Semantic Kernel Skills](https://learn.microsoft.com/semantic-kernel/)
- [ReAct Pattern](https://arxiv.org/abs/2210.03629)

---

*Created for educational purposes - January 2026*
