# 🎓 MCP Tutorial Demo Guide

## Instructor's Guide for Demonstrating Model Context Protocol

This guide provides a structured walkthrough for presenting the MCP Agentic Tutorial (`07_mcp_agentic_tutorial.py`) to students in a Generative AI Applications course.

---

## 📋 Pre-Demo Setup

### 1. Environment Setup

```powershell
# Navigate to the project directory
cd C:\Users\gkamhi\OPENVINO-GENAI-SAMPLES

# Activate the virtual environment
.\venv_agentic_demo\Scripts\activate

# Run the Streamlit app
streamlit run 07_mcp_agentic_tutorial.py
```

### 2. Pre-Demo Checklist

- [ ] Virtual environment activated
- [ ] Streamlit app running on `http://localhost:8501`
- [ ] Browser window visible to students
- [ ] Clear any previous task list (if demonstrating task manager)

---

## 🎯 Learning Objectives

By the end of this demo, students will understand:

1. What MCP (Model Context Protocol) is and why it matters
2. How tools are defined with JSON schemas
3. How AI agents route natural language to appropriate tools
4. The agentic decision-making process
5. Stateful interactions across multiple requests

---

## 📚 Demo Script (15-20 minutes)

### Demo 1: Introduce MCP Concepts (2-3 min)

**Action**: Navigate to the sidebar and show the "What is MCP?" section.

**Key Points to Explain**:

| Concept | Description |
|---------|-------------|
| **Tools** 🔧 | Functions the AI can call to perform actions |
| **Resources** 📁 | Data sources the AI can access |
| **Prompts** 💬 | Pre-defined conversation templates |

**Show the Flow Diagram**:

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐
│  User Request   │────▶│  AI Agent   │────▶│  MCP Tools   │
│  (Natural Lang) │     │  (Decides)  │     │  (Execute)   │
└─────────────────┘     └─────────────┘     └──────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │   Response  │
                        └─────────────┘
```

**Teaching Point**: "The AI agent understands your natural language and automatically decides which tool to use!"

---

### Demo 2: Weather Tool - Basic Routing (3 min)

**Goal**: Show how different phrasings map to the same tool.

**Try These Queries**:

| Query | What it Demonstrates |
|-------|---------------------|
| `"What's the weather in Paris?"` | Basic tool routing |
| `"Is it raining in Seattle?"` | Different phrasing, same tool |
| `"Temperature in Tokyo"` | Minimal query still works |
| `"How hot is it in Dubai?"` | Informal language understanding |

**Show in UI**:
- Which tool was selected: `weather`
- Extracted parameters: `location = "Paris"`
- Confidence level: `high`

**Teaching Point**: "Notice how the agent extracts the location parameter from completely different sentence structures!"

---

### Demo 3: Calculator Tool - Parameter Extraction (2 min)

**Goal**: Demonstrate natural language to structured expression conversion.

**Try These Queries**:

| Query | Expected Expression |
|-------|-------------------|
| `"Calculate 125 * 4 + 50"` | `125 * 4 + 50` |
| `"What is 100 divided by 5?"` | `100 / 5` |
| `"Compute 15 + 27"` | `15 + 27` |

**Teaching Point**: "The agent converts natural language math into evaluable expressions. In production, this would handle much more complex math with an LLM."

---

### Demo 4: Tool Schema Display (2 min)

**Action**: Click to expand the "Tool Schemas" section in the UI.

**Show This Schema**:

```json
{
  "name": "weather",
  "description": "Get current weather information for a location. 
                  Use this when the user asks about weather, 
                  temperature, or climate conditions.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or location to get weather for"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

**Key Points**:

| Schema Element | Purpose |
|----------------|---------|
| `name` | Unique identifier for the tool |
| `description` | Helps LLM decide when to use this tool |
| `properties` | Defines each parameter |
| `type` | Data type validation |
| `enum` | Restricts allowed values |
| `required` | Mandatory parameters |

**Teaching Point**: "This schema is what the LLM reads to understand tool capabilities. Good descriptions are crucial!"

---

### Demo 5: Task Manager - Stateful Multi-Step (5 min)

**Goal**: This is the most impressive demo - show stateful interaction across multiple requests.

#### Step 1: Add Tasks

```
"Add a task: Complete homework"
```
*Wait for response, then:*

```
"Add a task: Study for exam (high priority)"
```
*Wait for response, then:*

```
"Remind me to call mom"
```

#### Step 2: List Tasks

```
"Show my tasks"
```

**Point Out**: All three tasks are displayed with their priorities!

#### Step 3: Complete a Task

```
"Complete the homework task"
```

#### Step 4: Verify the Update

```
"List my tasks"
```

**Point Out**: The homework task now shows as completed!

**Teaching Point**: "The agent maintains state across requests. This is a key capability for real-world agentic applications!"

---

### Demo 6: Additional Tools (2-3 min)

#### Text Analyzer

```
"Analyze this text for sentiment: I love this course! It's amazing!"
```

Expected output: Sentiment = `positive`

```
"Analyze for keywords: Machine learning and artificial intelligence are transforming how we build software applications."
```

Expected output: Keywords extracted

#### Unit Converter

```
"Convert 100 kilometers to miles"
```

```
"What is 32 degrees Fahrenheit in Celsius?"
```

---

### Demo 7: Edge Cases & Routing Decisions (3 min)

**Goal**: Show how the agent handles ambiguous or unusual queries.

| Query | Discussion Point |
|-------|-----------------|
| `"Hello!"` | No matching tool - shows fallback behavior |
| `"What's 50 degrees Fahrenheit in Celsius?"` | Routes to unit_converter, not calculator |
| `"Weather"` | Missing location - observe how it handles incomplete input |

---

## 🎯 Interactive Exercise (Optional, 5 min)

Have students try their own queries and observe:

1. Which tool was selected?
2. What parameters were extracted?
3. What was the confidence level?

### Challenge Queries for Students

```
"How many feet are in 10 meters?"
"Is the weather good for hiking in Denver?"
"Analyze: This product is terrible, worst purchase ever!"
"Add an urgent task: Finish the project by Friday"
```

---

## ❓ Common Questions & Answers

| Question | Answer |
|----------|--------|
| "Is this using a real LLM?" | No, this demo uses pattern matching to simulate LLM behavior. This makes it easier to understand the concepts without needing API keys or network calls. |
| "How does real MCP work?" | Real MCP servers expose tools via a standardized protocol. Clients like Claude Desktop or VS Code can discover and call these tools automatically. |
| "What about chaining tools?" | Great question! That's covered in the Skills tutorial (`08_skills_agentic_tutorial.py`) which we'll see next. |
| "Why use schemas?" | Schemas provide a contract that both the AI and the tool understand. They enable type validation, required parameter checking, and help the LLM make better tool choices. |
| "Can I add my own tools?" | Yes! Look at the `create_tool_registry()` function. You just need to define a Tool with parameters and a function. |

---

## 📝 Key Takeaways to Emphasize

1. **Tool descriptions matter** - They help the LLM choose the right tool
2. **Schema defines the contract** - Parameters, types, and requirements
3. **Agentic routing** - Natural language → Intent → Tool → Parameters → Response
4. **Stateful interactions** - Agents can maintain context across requests
5. **In production** - Pattern matching would be replaced by LLM intelligence

---

## 🔗 Transition to Skills Tutorial

After completing this demo, transition to the Skills tutorial:

> "Now that you understand how individual tools work with MCP, let's see how we can **chain** capabilities together and share context between them. That's where the **Skills pattern** comes in."

```powershell
streamlit run 08_skills_agentic_tutorial.py --server.port 8511
```

---

## 📚 Additional Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Building MCP Servers](https://modelcontextprotocol.io/docs/building-servers)
- Course README: [README_MCP_Tutorial.md](README_MCP_Tutorial.md)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Streamlit not starting | Ensure virtual environment is activated |
| Tasks not persisting | Tasks are session-based; refresh clears them |
| Tool not recognized | Check if query matches expected patterns |
| Port already in use | Use `--server.port 8502` to specify different port |

---

*Created for the Generative AI Applications Course - January 2026*
