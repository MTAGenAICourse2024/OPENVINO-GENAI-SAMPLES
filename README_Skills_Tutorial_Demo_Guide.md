# 🎯 Skills Tutorial Demo Guide

## Instructor's Guide for Demonstrating the Skills-Based Agentic Pattern

This guide provides a structured walkthrough for presenting the Skills Agentic Tutorial to students in a Generative AI Applications course.

**Two versions available:**
- `08_skills_agentic_tutorial.py` - Basic skills with rule-based selection
- `09_skills_agentic_tutorial_langgraph.py` - **Recommended**: LangGraph + LLM integration

---

## 📋 Pre-Demo Setup

### 1. Environment Setup

```powershell
# Navigate to the project directory
cd C:\Users\gkamhi\OPENVINO-GENAI-SAMPLES

# Activate the virtual environment
.\venv_agentic_demo\Scripts\activate

# Run the LangGraph version (recommended)
streamlit run 09_skills_agentic_tutorial_langgraph.py --server.port 8502

# Or run the basic version
streamlit run 08_skills_agentic_tutorial.py --server.port 8511
```

### 2. Azure OpenAI Setup (for 09 version)

Ensure your `.env` file has:
```env
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_LLM_4="gpt-4o"
```

### 3. Pre-Demo Checklist

- [ ] MCP Tutorial demo completed (recommended context)
- [ ] Virtual environment activated
- [ ] Required packages installed (`pip install openai python-dotenv`)
- [ ] Streamlit app running
- [ ] Browser window visible to students
- [ ] Clear any previous memory/context for a fresh start

---

## 🎯 Learning Objectives

By the end of this demo, students will understand:

1. What Skills are and how they differ from MCP Tools
2. How the SkillContext enables stateful interactions
3. How skills share data through context variables
4. Skill chaining - using output of one skill as input to another
5. Confidence scoring for skill selection
6. When to use Skills vs MCP Tools

---

## 🔄 Transition from MCP Tutorial

If you just completed the MCP demo, use this transition:

> "We just saw how MCP Tools work as standalone functions with schemas. But what if we need tools that can **share context** and **build on each other's results**? That's where the **Skills pattern** comes in!"

---

## 📚 Demo Script (20-25 minutes)

### Demo 1: Skills vs MCP Concepts (3 min)

**Action**: Navigate to the sidebar and show the "What are Skills?" section.

**Key Comparison Table to Display**:

| Aspect | MCP Tools | Skills |
|--------|-----------|--------|
| **State** | Stateless | Stateful (maintains context) |
| **Composition** | Called individually | Can be chained |
| **Context** | Manual passing | Automatic via SkillContext |
| **Reusability** | Function-level | Component-level |
| **Complexity** | Single operation | Multi-step capable |

**Teaching Point**: "Skills are like 'smart tools' that remember what happened before and can work together!"

---

### Demo 2: Basic Skill Execution (3 min)

**Goal**: Show individual skills working (similar to MCP, but with state).

**Try These Queries**:

| Query | Skill Selected |
|-------|---------------|
| `"What's the weather in London?"` | Weather Skill |
| `"Calculate 25 * 4"` | Calculator Skill |
| `"Analyze this text for sentiment: I love learning AI!"` | Text Analysis Skill |

**Point Out**:
- Which skill was selected
- The confidence score
- **KEY**: Context variables that were stored!

**Teaching Point**: "Notice how each skill stores data in the context. The Weather Skill stored `current_weather` and `current_location`. This is key for chaining!"

---

### Demo 3: The SkillContext System (3 min)

**Action**: Show the context variables panel in the UI after running a skill.

**Explain the SkillContext Structure**:

```python
@dataclass
class SkillContext:
    user_input: str = ""                    # Current request
    previous_results: List[Dict] = []       # History of skill outputs
    variables: Dict[str, Any] = {}          # Shared state!
    conversation_history: List[Dict] = []   # Full dialogue
```

**Key Methods**:
- `context.set_variable("name", value)` - Store data for other skills
- `context.get_variable("name")` - Retrieve data from other skills

**Teaching Point**: "This shared context is what makes skill chaining possible!"

---

### Demo 4: ⭐ Skill Chaining - The Main Event (7 min)

**This is the highlight of the demo - show how skills build on each other!**

#### Step 1: Get Weather First

```
"What's the weather in Paris?"
```

**Point Out**: 
- Weather data is returned
- `current_weather` and `current_location` are stored in context

#### Step 2: Now Ask for Recommendations

```
"What do you recommend?"
```

or

```
"Give me some suggestions"
```

**Magic Moment**: The Recommendation Skill reads the weather context and gives **weather-aware recommendations**!

- If rainy: "☔ Bring an umbrella today!"
- If hot (>25°C): "🧴 Don't forget sunscreen!" + "💧 Stay hydrated!"
- If cold (<10°C): "🧥 Dress warmly!"

**Teaching Point**: "The Recommendation Skill never asked about weather - it READ the context that the Weather Skill stored. This is the power of skill chaining!"

#### Step 3: Add Text Analysis to the Chain

```
"Analyze this text for sentiment: This product is terrible and disappointing!"
```

#### Step 4: Ask for Recommendations Again

```
"Any recommendations?"
```

**New Result**: Now recommendations include BOTH weather AND text analysis context:
- Weather recommendations (if still in context)
- Text improvement suggestions based on negative sentiment

**Teaching Point**: "Each skill adds to the shared context. The more skills you use, the more context-aware your recommendations become!"

---

### Demo 5: Memory Skill - Persistent Context (4 min)

**Goal**: Show how the agent can remember information across the session.

#### Step 1: Store Some Memories

```
"Remember that my name is Alex"
```

```
"Remember that I prefer Fahrenheit temperatures"
```

```
"Remember that I'm planning a trip to Tokyo"
```

#### Step 2: Recall Memories

```
"What do you remember?"
```

**Point Out**: All stored memories are returned with timestamps!

#### Step 3: Use Memory in Context

```
"What's the weather in Tokyo?"
```

**Teaching Point**: "In a full implementation, the Weather Skill could read the memory context to use Fahrenheit as preferred, or automatically check Tokyo weather based on stored trip plans!"

---

### Demo 6: Confidence Scoring (3 min)

**Goal**: Show how skills compete for handling requests.

**Try Ambiguous Queries**:

| Query | Discussion |
|-------|-----------|
| `"What is 50?"` | Could be Calculator... but is it? |
| `"Tell me about Paris"` | Weather? Memory? Neither? |
| `"Analyze 100 + 50"` | Calculator or Text Analysis? |

**Show Confidence Scores**: Point out how each skill rates its confidence for the input.

**Teaching Point**: "The orchestrator picks the skill with the highest confidence. Good trigger words in skill definitions are crucial!"

---

### Demo 7: Complete Chain Demo (5 min)

**Goal**: Walk through a realistic multi-skill workflow.

```
Step 1: "What's the weather in New York?"
        → Weather Skill stores: current_weather, current_location
        
Step 2: "Remember I'm visiting New York next week"
        → Memory Skill stores the trip plan
        
Step 3: "Analyze this text: I'm so excited about my trip!"
        → Text Analysis stores: sentiment = positive
        
Step 4: "What do you recommend?"
        → Recommendation Skill reads ALL context:
           - Weather data → appropriate clothing suggestions
           - Positive sentiment → encouraging message
           - (Memory context available for future enhancements)
```

---

## 🆚 Skills vs MCP: When to Use Each

Present this decision framework:

### Use MCP Tools When:
- ✅ Integrating with external services (Claude Desktop, VS Code)
- ✅ You need standardized tool schemas
- ✅ Tools are stateless and independent
- ✅ Interoperability with other MCP clients is important

### Use Skills When:
- ✅ Building internal agent capabilities
- ✅ Skills need to share context and state
- ✅ You want composable, chainable operations
- ✅ Building complex multi-step workflows
- ✅ Need confidence scoring for skill selection

### Use Both Together:
- ✅ Complex production agents often combine both patterns
- ✅ MCP for external tool integrations
- ✅ Skills for internal reasoning and memory
- ✅ LangGraph can orchestrate both

---

## ❓ Common Questions & Answers

| Question | Answer |
|----------|--------|
| "How is this different from just passing parameters?" | With MCP, you manually pass results. With Skills, the context is automatic - any skill can read what any other skill stored. |
| "Can Skills call other Skills?" | Yes! You can define prerequisites or use the orchestrator to chain skills programmatically. |
| "Is context persistent across sessions?" | By default, context is session-based. The Memory Skill uses Streamlit session_state to persist within a session. For production, you'd use a database. |
| "What about error handling between skills?" | The orchestrator can check prerequisites and handle failures. If a required skill hasn't run, the dependent skill can report the error. |
| "How does confidence scoring work?" | Each skill calculates a score (0-1) based on how many trigger words match the input. The highest-scoring skill is selected. |

---

## 🧪 Interactive Exercise (Optional, 5 min)

### Challenge: Build a Context Chain

Have students try to build the richest context possible:

1. Start with weather for a city
2. Do a calculation
3. Analyze some text with different sentiments
4. Store some memories
5. Ask for recommendations and see how many context sources are used!

### Advanced Challenge

```
"What's the weather in Miami?"
"Calculate how much 7 days of vacation costs at $150 per day"
"Analyze: I can't wait for my beach vacation!"
"Remember I need to book a hotel"
"What do you recommend?"
```

**Expected Result**: Recommendations should reference:
- Miami weather (beach/sun related)
- The calculated vacation cost (budget tips if large)
- Positive sentiment (encouraging message)

---

## 📝 Key Takeaways to Emphasize

1. **Skills are stateful** - Unlike MCP tools, skills maintain context
2. **Context flows automatically** - No manual parameter passing between skills
3. **Chaining enables complex workflows** - Skills build on each other's outputs
4. **Confidence scoring** - Skills compete based on how well they match the input
5. **Combine with MCP** - Production agents often use both patterns

---

## 🔗 Transition to LangGraph Demo

If continuing to the visual analysis demo:

> "Now that you understand both MCP Tools and Skills, let's see how we can orchestrate complex visual analysis workflows using **LangGraph**. This combines the power of skills with graph-based workflow management."

```powershell
streamlit run 05_agentic_visual_analysis_langgraph.py
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Context not showing variables | Run a skill first; context starts empty |
| Recommendations not context-aware | Make sure to run Weather or Text Analysis before Recommendations |
| Memory not persisting | Memory is session-based; page refresh clears it |
| Wrong skill selected | Check trigger words; try more specific phrasing |
| Port conflict with MCP demo | Use `--server.port 8511` or different port |

---

## 📊 Quick Reference: Available Skills

| Skill | Category | Triggers | Context Variables Set |
|-------|----------|----------|----------------------|
| **Weather** | Information | weather, temperature, forecast, rain, sunny | `current_weather`, `current_location` |
| **Calculator** | Utility | calculate, compute, math, +, -, *, / | `last_calculation` |
| **Text Analysis** | Analysis | analyze, sentiment, keywords, summarize | `text_analysis_result` |
| **Memory** | Memory | remember, recall, forget, store | Uses `st.session_state.agent_memory` |
| **Recommendation** | Generation | recommend, suggest, advice | Reads from all other skills' context |

---

## 📚 Additional Resources

- Course README: [README_Skills_Tutorial.md](README_Skills_Tutorial.md)
- MCP Tutorial: [README_MCP_Tutorial.md](README_MCP_Tutorial.md)
- LangGraph Concepts: [README_AgenticVisualAnalysis_LangGraph.md](README_AgenticVisualAnalysis_LangGraph.md)

---

*Created for the Generative AI Applications Course - January 2026*
