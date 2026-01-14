"""
🎓 MCP (Model Context Protocol) Agentic Tutorial
=================================================
An educational demonstration of MCP concepts and agentic tool use
for a college course on Generative AI Applications.

This tutorial demonstrates:
- 🔧 Tool Definition: How to create tools that AI agents can use
- 🧠 Natural Language Understanding: Converting user intent to tool calls
- 🔄 Agentic Workflows: Autonomous decision-making about which tool to use
- 📡 MCP Concepts: Understanding the Model Context Protocol pattern

Learning Objectives:
1. Understand how AI agents decide which tools to use
2. Learn to define tools with clear schemas
3. See how natural language maps to structured tool calls
4. Explore the agentic decision-making process

Author: Educational Demo for GenAI Course
Date: January 2026
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

# ============================================================================
# 📚 EDUCATIONAL SECTION: What is MCP?
# ============================================================================

MCP_INTRO = """
## 🔌 What is the Model Context Protocol (MCP)?

**MCP** is a standardized way for AI models to interact with external tools and services.
Think of it as a "universal adapter" that lets AI assistants:

### Key Concepts:

1. **Tools** 🔧
   - Functions that the AI can call to perform actions
   - Each tool has a name, description, and input schema
   - Examples: search the web, read files, run calculations

2. **Resources** 📁
   - Data sources the AI can access
   - Examples: databases, file systems, APIs

3. **Prompts** 💬
   - Pre-defined conversation templates
   - Help guide the AI's behavior

### Why MCP Matters:

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐
│  User Request   │────▶│  AI Agent   │────▶│  MCP Tools   │
│  (Natural Lang) │     │  (Decides)  │     │  (Execute)   │
└─────────────────┘     └─────────────┘     └──────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │   Response  │
                        │  (Natural)  │
                        └─────────────┘
```

The AI agent understands your natural language request and automatically
decides which tool(s) to use to fulfill it!
"""

# ============================================================================
# 🔧 TOOL DEFINITIONS - The Heart of MCP
# ============================================================================

@dataclass
class ToolParameter:
    """Defines a parameter for a tool"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: List[str] = field(default_factory=list)

@dataclass 
class Tool:
    """
    Represents an MCP Tool that the AI agent can use.
    
    In MCP, tools are defined with:
    - A unique name
    - A description (helps the AI understand when to use it)
    - Input parameters with their types and descriptions
    - A function to execute
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: str = "general"
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format (MCP-compatible)"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

# ============================================================================
# 🛠️ SAMPLE TOOLS - Educational Examples
# ============================================================================

def weather_tool(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Simulated weather lookup tool"""
    # In real MCP, this would call an actual weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp_c = random.randint(10, 35)
    temp = temp_c if unit == "celsius" else int(temp_c * 9/5 + 32)
    unit_symbol = "°C" if unit == "celsius" else "°F"
    
    return {
        "location": location,
        "temperature": f"{temp}{unit_symbol}",
        "condition": random.choice(weather_conditions),
        "humidity": f"{random.randint(30, 80)}%",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def calculator_tool(expression: str) -> Dict[str, Any]:
    """Safe mathematical expression evaluator"""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression", "result": None}
        
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        return {"error": str(e), "result": None}

def text_analyzer_tool(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """Analyze text in various ways"""
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(re.split(r'[.!?]+', text))
    
    result = {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "analysis_type": analysis_type
    }
    
    if analysis_type == "summary":
        result["summary"] = f"Text contains {word_count} words in {sentence_count} sentences."
    elif analysis_type == "sentiment":
        # Simplified sentiment (real would use NLP)
        positive_words = ["good", "great", "excellent", "happy", "love", "amazing"]
        negative_words = ["bad", "terrible", "hate", "awful", "sad", "poor"]
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count > neg_count:
            result["sentiment"] = "positive"
        elif neg_count > pos_count:
            result["sentiment"] = "negative"
        else:
            result["sentiment"] = "neutral"
    elif analysis_type == "keywords":
        # Simple keyword extraction (top words by frequency)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        result["keywords"] = [w[0] for w in top_words]
    
    return result

def unit_converter_tool(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert between common units"""
    conversions = {
        # Length
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x / 3.28084,
        ("kilometers", "miles"): lambda x: x * 0.621371,
        ("miles", "kilometers"): lambda x: x / 0.621371,
        # Temperature
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        # Weight
        ("kilograms", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kilograms"): lambda x: x / 2.20462,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return {
            "input": f"{value} {from_unit}",
            "output": f"{result:.2f} {to_unit}",
            "value": round(result, 4)
        }
    else:
        return {
            "error": f"Conversion from {from_unit} to {to_unit} not supported",
            "supported_conversions": list(conversions.keys())
        }

def task_manager_tool(action: str, task: str = "", priority: str = "medium") -> Dict[str, Any]:
    """Manage a simple task list"""
    # Use session state to persist tasks
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    
    if action == "add":
        new_task = {
            "id": len(st.session_state.tasks) + 1,
            "task": task,
            "priority": priority,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "completed": False
        }
        st.session_state.tasks.append(new_task)
        return {"status": "success", "message": f"Added task: {task}", "task": new_task}
    
    elif action == "list":
        return {"status": "success", "tasks": st.session_state.tasks, "count": len(st.session_state.tasks)}
    
    elif action == "complete":
        for t in st.session_state.tasks:
            if task.lower() in t["task"].lower():
                t["completed"] = True
                return {"status": "success", "message": f"Completed task: {t['task']}"}
        return {"status": "error", "message": f"Task not found: {task}"}
    
    elif action == "clear":
        st.session_state.tasks = []
        return {"status": "success", "message": "All tasks cleared"}
    
    return {"status": "error", "message": f"Unknown action: {action}"}

# ============================================================================
# 📋 TOOL REGISTRY - MCP Tool Collection
# ============================================================================

def create_tool_registry() -> Dict[str, Tool]:
    """Create and register all available tools"""
    
    tools = {
        "weather": Tool(
            name="weather",
            description="Get current weather information for a location. Use this when the user asks about weather, temperature, or climate conditions.",
            parameters=[
                ToolParameter("location", "string", "City name or location to get weather for"),
                ToolParameter("unit", "string", "Temperature unit", required=False, enum=["celsius", "fahrenheit"])
            ],
            function=weather_tool,
            category="information"
        ),
        
        "calculator": Tool(
            name="calculator", 
            description="Perform mathematical calculations. Use this when the user wants to calculate, compute, or do math operations.",
            parameters=[
                ToolParameter("expression", "string", "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '100 / 4')")
            ],
            function=calculator_tool,
            category="utility"
        ),
        
        "text_analyzer": Tool(
            name="text_analyzer",
            description="Analyze text for various properties like word count, sentiment, or keywords. Use this when the user wants to analyze, summarize, or extract information from text.",
            parameters=[
                ToolParameter("text", "string", "The text to analyze"),
                ToolParameter("analysis_type", "string", "Type of analysis to perform", required=False, 
                            enum=["summary", "sentiment", "keywords"])
            ],
            function=text_analyzer_tool,
            category="analysis"
        ),
        
        "unit_converter": Tool(
            name="unit_converter",
            description="Convert values between different units of measurement. Use this when the user wants to convert temperatures, distances, weights, etc.",
            parameters=[
                ToolParameter("value", "number", "The numeric value to convert"),
                ToolParameter("from_unit", "string", "The unit to convert from"),
                ToolParameter("to_unit", "string", "The unit to convert to")
            ],
            function=unit_converter_tool,
            category="utility"
        ),
        
        "task_manager": Tool(
            name="task_manager",
            description="Manage a task/todo list. Use this when the user wants to add, list, complete, or manage tasks and todos.",
            parameters=[
                ToolParameter("action", "string", "Action to perform", enum=["add", "list", "complete", "clear"]),
                ToolParameter("task", "string", "Task description (for add/complete actions)", required=False),
                ToolParameter("priority", "string", "Task priority", required=False, enum=["low", "medium", "high"])
            ],
            function=task_manager_tool,
            category="productivity"
        )
    }
    
    return tools

# ============================================================================
# 🧠 AGENTIC ROUTER - Natural Language to Tool Selection
# ============================================================================

class AgenticRouter:
    """
    The Agentic Router is the "brain" that decides which tool to use.
    
    In a real MCP implementation, this would be powered by an LLM.
    For this educational demo, we use pattern matching to simulate
    how an LLM would understand and route requests.
    """
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.routing_patterns = self._build_routing_patterns()
    
    def _build_routing_patterns(self) -> List[Dict]:
        """Build patterns for routing natural language to tools"""
        return [
            {
                "tool": "weather",
                "patterns": [
                    r"weather\s+(?:in|for|at)\s+(\w+)",
                    r"temperature\s+(?:in|for|at)\s+(\w+)",
                    r"(?:how|what).*weather.*(\w+)",
                    r"(?:is it|will it)\s+(?:rain|sunny|cold|hot).*(\w+)",
                ],
                "extract": lambda m, text: {"location": m.group(1) if m else self._extract_location(text)}
            },
            {
                "tool": "calculator",
                "patterns": [
                    r"calculate\s+(.+)",
                    r"compute\s+(.+)",
                    r"what\s+is\s+(\d+[\s\d\+\-\*\/\.\(\)]+)",
                    r"(\d+\s*[\+\-\*\/]\s*\d+)",
                ],
                "extract": lambda m, text: {"expression": m.group(1).strip() if m else self._extract_math(text)}
            },
            {
                "tool": "text_analyzer",
                "patterns": [
                    r"analyze\s+(?:this\s+)?(?:text|sentence|paragraph)",
                    r"(?:word|character)\s+count",
                    r"sentiment\s+(?:of|for|analysis)",
                    r"summarize\s+(?:this)?",
                    r"(?:find|extract)\s+keywords",
                ],
                "extract": lambda m, text: {"text": text, "analysis_type": self._detect_analysis_type(text)}
            },
            {
                "tool": "unit_converter",
                "patterns": [
                    r"convert\s+(\d+\.?\d*)\s*(\w+)\s+to\s+(\w+)",
                    r"(\d+\.?\d*)\s*(\w+)\s+(?:in|to)\s+(\w+)",
                    r"how\s+many\s+(\w+)\s+(?:is|are|in)\s+(\d+\.?\d*)\s*(\w+)",
                ],
                "extract": lambda m, text: self._extract_conversion(m, text)
            },
            {
                "tool": "task_manager",
                "patterns": [
                    r"(?:add|create|new)\s+(?:a\s+)?task",
                    r"(?:show|list|display)\s+(?:my\s+)?tasks",
                    r"(?:complete|finish|done)\s+(?:the\s+)?task",
                    r"(?:clear|delete|remove)\s+(?:all\s+)?tasks",
                    r"todo",
                    r"remind\s+me\s+to",
                ],
                "extract": lambda m, text: self._extract_task_params(text)
            }
        ]
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text"""
        # Simple extraction - look for capitalized words
        words = text.split()
        for word in words:
            if word[0].isupper() and word.lower() not in ["what", "how", "the", "is", "it"]:
                return word
        return "New York"  # Default
    
    def _extract_math(self, text: str) -> str:
        """Extract mathematical expression from text"""
        # Find numbers and operators
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        return match.group().strip() if match else "0"
    
    def _detect_analysis_type(self, text: str) -> str:
        """Detect what type of text analysis is requested"""
        text_lower = text.lower()
        if "sentiment" in text_lower or "feeling" in text_lower:
            return "sentiment"
        elif "keyword" in text_lower or "important" in text_lower:
            return "keywords"
        return "summary"
    
    def _extract_conversion(self, match, text: str) -> Dict:
        """Extract conversion parameters"""
        if match:
            groups = match.groups()
            if len(groups) == 3:
                # Check if it's "how many X in Y Z" format
                if "how many" in text.lower():
                    return {"value": float(groups[1]), "from_unit": groups[2], "to_unit": groups[0]}
                return {"value": float(groups[0]), "from_unit": groups[1], "to_unit": groups[2]}
        return {"value": 1, "from_unit": "meters", "to_unit": "feet"}
    
    def _extract_task_params(self, text: str) -> Dict:
        """Extract task management parameters"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["add", "create", "new", "remind"]):
            # Extract task description
            task_match = re.search(r'(?:add|create|new)\s+(?:a\s+)?task[:\s]+(.+)', text_lower)
            if not task_match:
                task_match = re.search(r'remind\s+me\s+to\s+(.+)', text_lower)
            task_desc = task_match.group(1) if task_match else text
            
            priority = "medium"
            if "high" in text_lower or "urgent" in text_lower or "important" in text_lower:
                priority = "high"
            elif "low" in text_lower:
                priority = "low"
                
            return {"action": "add", "task": task_desc.strip(), "priority": priority}
        
        elif any(word in text_lower for word in ["show", "list", "display", "what"]):
            return {"action": "list"}
        
        elif any(word in text_lower for word in ["complete", "finish", "done"]):
            task_match = re.search(r'(?:complete|finish|done)\s+(?:the\s+)?(?:task\s+)?(.+)', text_lower)
            return {"action": "complete", "task": task_match.group(1) if task_match else ""}
        
        elif any(word in text_lower for word in ["clear", "delete", "remove"]):
            return {"action": "clear"}
        
        return {"action": "list"}
    
    def route(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Route natural language input to the appropriate tool.
        
        This is where the "agentic" magic happens - the router analyzes
        the user's intent and decides which tool to use.
        
        Returns:
            Dict with 'tool', 'parameters', and 'confidence' if a match is found
            None if no matching tool is found
        """
        user_lower = user_input.lower()
        
        for route in self.routing_patterns:
            for pattern in route["patterns"]:
                match = re.search(pattern, user_lower)
                if match:
                    params = route["extract"](match, user_input)
                    return {
                        "tool": route["tool"],
                        "parameters": params,
                        "confidence": "high",
                        "matched_pattern": pattern
                    }
        
        # Fallback: Try to detect intent from keywords
        keyword_mapping = {
            "weather": ["weather", "temperature", "rain", "sunny", "forecast", "climate"],
            "calculator": ["calculate", "compute", "math", "sum", "multiply", "divide", "add", "subtract"],
            "text_analyzer": ["analyze", "count", "sentiment", "summarize", "keywords"],
            "unit_converter": ["convert", "meters", "feet", "celsius", "fahrenheit", "miles", "kilometers"],
            "task_manager": ["task", "todo", "remind", "list", "complete"]
        }
        
        for tool_name, keywords in keyword_mapping.items():
            if any(kw in user_lower for kw in keywords):
                return {
                    "tool": tool_name,
                    "parameters": {},
                    "confidence": "low",
                    "matched_pattern": "keyword_fallback"
                }
        
        return None
    
    def execute(self, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the routed tool with the extracted parameters"""
        tool_name = routing_result["tool"]
        params = routing_result["parameters"]
        
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.function(**params)
            return {
                "success": True,
                "tool_used": tool_name,
                "parameters": params,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool_used": tool_name,
                "error": str(e)
            }

# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

def display_tool_schemas(tools: Dict[str, Tool]):
    """Display tool schemas in an educational format"""
    st.markdown("### 📋 Available Tools (MCP Schema Format)")
    
    for name, tool in tools.items():
        with st.expander(f"🔧 {name}: {tool.description[:50]}...", expanded=False):
            st.markdown(f"**Category:** {tool.category}")
            st.markdown(f"**Description:** {tool.description}")
            st.markdown("**Schema (JSON):**")
            st.json(tool.to_schema())

def display_routing_explanation(routing_result: Dict[str, Any], user_input: str):
    """Explain how the routing decision was made"""
    st.markdown("### 🧠 Agentic Decision Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input Analysis:**")
        st.info(f"User said: \"{user_input}\"")
        
    with col2:
        st.markdown("**Routing Decision:**")
        if routing_result:
            st.success(f"Selected Tool: **{routing_result['tool']}**")
            st.caption(f"Confidence: {routing_result['confidence']}")
        else:
            st.warning("No matching tool found")

def format_result(result: Dict[str, Any]) -> str:
    """Format tool result for display"""
    if result.get("success"):
        tool_result = result.get("result", {})
        if isinstance(tool_result, dict):
            if "error" in tool_result:
                return f"❌ Error: {tool_result['error']}"
            return json.dumps(tool_result, indent=2)
        return str(tool_result)
    else:
        return f"❌ Error: {result.get('error', 'Unknown error')}"

def main():
    st.set_page_config(
        page_title="🎓 MCP Agentic Tutorial",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply dark theme CSS
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .stApp h1 {
            color: #F5DEB3 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stApp h2, .stApp h3, .stApp h4 {
            color: #FAEBD7 !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #4a90d9 0%, #67b26f 100%);
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #67b26f 0%, #4a90d9 100%);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize tools and router
    tools = create_tool_registry()
    router = AgenticRouter(tools)
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎓 MCP Tutorial")
        st.markdown("---")
        
        st.markdown("### 📚 Learning Mode")
        show_schemas = st.checkbox("Show Tool Schemas", value=True)
        show_routing = st.checkbox("Show Routing Logic", value=True)
        show_raw_output = st.checkbox("Show Raw JSON Output", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Try These Examples:")
        
        example_prompts = [
            "What's the weather in Paris?",
            "Calculate 125 * 4 + 50",
            "Convert 100 kilometers to miles",
            "Analyze this text for sentiment: I love this amazing product!",
            "Add a task: Complete MCP tutorial",
            "Show my tasks",
        ]
        
        for prompt in example_prompts:
            if st.button(f"📝 {prompt[:30]}...", key=f"ex_{prompt}"):
                st.session_state.example_prompt = prompt
                st.rerun()  # Rerun to apply the example to the input field
        
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.tasks = []
            st.rerun()
    
    # Main content
    st.title("🔧 MCP Agentic Tutorial")
    st.markdown("### Learn how AI agents use tools through natural language")
    
    # Introduction
    with st.expander("📚 What is MCP? (Click to Learn)", expanded=False):
        st.markdown(MCP_INTRO)
    
    # Tool schemas section - displayed directly (not inside expander to avoid nesting)
    if show_schemas:
        st.markdown("---")
        display_tool_schemas(tools)
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("## 💬 Chat with the Agent")
    st.markdown("*Ask me anything! I'll decide which tool to use.*")
    
    # Check for example prompt
    initial_value = ""
    if "example_prompt" in st.session_state:
        initial_value = st.session_state.example_prompt
        del st.session_state.example_prompt
    
    # User input
    user_input = st.text_input(
        "Your message:",
        value=initial_value,
        placeholder="e.g., What's the weather in Tokyo? or Calculate 15 * 7",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("🚀 Send", type="primary")
    
    if send_button and user_input:
        # Route the request
        routing_result = router.route(user_input)
        
        # Store in session state so results persist
        st.session_state.last_routing_result = routing_result
        st.session_state.last_user_input = user_input
        
        if routing_result:
            execution_result = router.execute(routing_result)
            st.session_state.last_execution_result = execution_result
        else:
            st.session_state.last_execution_result = None
    
    # Display results (persisted in session state)
    if st.session_state.get("last_routing_result") or st.session_state.get("last_user_input"):
        routing_result = st.session_state.get("last_routing_result")
        user_input_display = st.session_state.get("last_user_input", "")
        execution_result = st.session_state.get("last_execution_result")
        
        # Show routing explanation
        if show_routing:
            display_routing_explanation(routing_result, user_input_display)
        
        st.markdown("---")
        
        if routing_result and execution_result:
            # Display results
            st.markdown("### 📤 Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Tool Used:** `{routing_result['tool']}`")
                st.markdown(f"**Parameters:**")
                st.json(routing_result['parameters'])
            
            with col2:
                st.markdown("**Output:**")
                if execution_result.get("success"):
                    result_data = execution_result.get("result", {})
                    
                    # Format nicely based on tool
                    if routing_result['tool'] == "weather":
                        st.success(f"🌡️ Weather in {result_data.get('location', 'Unknown')}")
                        st.metric("Temperature", result_data.get('temperature', 'N/A'))
                        st.write(f"Condition: {result_data.get('condition', 'N/A')}")
                        st.write(f"Humidity: {result_data.get('humidity', 'N/A')}")
                    
                    elif routing_result['tool'] == "calculator":
                        if result_data.get("error"):
                            st.error(result_data["error"])
                        else:
                            st.success(f"🔢 {result_data.get('expression')} = **{result_data.get('result')}**")
                    
                    elif routing_result['tool'] == "text_analyzer":
                        st.success("📊 Text Analysis Complete")
                        st.write(f"Words: {result_data.get('word_count')}")
                        st.write(f"Characters: {result_data.get('character_count')}")
                        if 'sentiment' in result_data:
                            sentiment = result_data['sentiment']
                            emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
                            st.write(f"Sentiment: {emoji} {sentiment}")
                        if 'keywords' in result_data:
                            st.write(f"Keywords: {', '.join(result_data['keywords'])}")
                    
                    elif routing_result['tool'] == "unit_converter":
                        if result_data.get("error"):
                            st.error(result_data["error"])
                        else:
                            st.success(f"📏 {result_data.get('input')} = **{result_data.get('output')}**")
                    
                    elif routing_result['tool'] == "task_manager":
                        if result_data.get("status") == "success":
                            st.success(f"✅ {result_data.get('message', 'Operation completed')}")
                            if 'tasks' in result_data:
                                if result_data['tasks']:
                                    for task in result_data['tasks']:
                                        status = "✅" if task.get('completed') else "⬜"
                                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get('priority', 'medium'), "🟡")
                                        st.write(f"{status} {priority_emoji} {task.get('task')}")
                                else:
                                    st.info("No tasks yet. Add one!")
                        else:
                            st.error(result_data.get('message', 'Operation failed'))
                    
                    if show_raw_output:
                        st.markdown("**Raw JSON:**")
                        st.json(result_data)
                else:
                    st.error(f"Error: {execution_result.get('error')}")
            
            # Add to history only on new submission
            if send_button and user_input:
                st.session_state.conversation_history.append({
                    "user": user_input,
                    "tool": routing_result['tool'],
                    "result": execution_result
                })
        elif routing_result is None and st.session_state.get("last_user_input"):
            st.warning("🤔 I couldn't determine which tool to use for that request.")
            st.info("Try being more specific, or use one of the example prompts in the sidebar!")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.container():
                st.markdown(f"**You:** {entry['user']}")
                st.markdown(f"**Tool:** `{entry['tool']}`")
                if entry['result'].get('success'):
                    st.markdown("**Status:** ✅ Success")
                else:
                    st.markdown("**Status:** ❌ Failed")
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎓 MCP Agentic Tutorial - Educational Demo for GenAI Course</p>
        <p>Learn more about MCP at <a href='https://modelcontextprotocol.io'>modelcontextprotocol.io</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
