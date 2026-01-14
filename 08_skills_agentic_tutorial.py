"""
🎓 Skills-Based Agentic AI Tutorial
====================================
An educational demonstration of the Skills pattern for agentic AI
for a college course on Generative AI Applications.

This tutorial demonstrates:
- 🎯 Skills: Composable, reusable units of AI capability
- 🔗 Skill Chaining: Combining skills for complex tasks
- 🧠 Skill Selection: How agents choose which skills to apply
- 📦 Skill Composition: Building complex behaviors from simple skills

Comparison with MCP:
- MCP: Tools are standalone functions with schemas
- Skills: Composable capabilities with context and memory

Learning Objectives:
1. Understand the Skills pattern for agentic AI
2. Learn to create composable, reusable skills
3. See how skills can be chained together
4. Compare Skills vs MCP approaches

Author: Educational Demo for GenAI Course
Date: January 2026
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import random

# ============================================================================
# 📚 EDUCATIONAL SECTION: What are Skills?
# ============================================================================

SKILLS_INTRO = """
## 🎯 What are Skills in Agentic AI?

**Skills** are composable units of capability that an AI agent can use to accomplish tasks.
Unlike simple tools, skills can:

### Key Differences from MCP Tools:

| Aspect | MCP Tools | Skills |
|--------|-----------|--------|
| **Structure** | Standalone functions | Objects with state |
| **Composition** | Called individually | Can be chained |
| **Context** | Stateless | Can maintain context |
| **Reusability** | Function-level | Component-level |
| **Complexity** | Single operation | Multi-step capable |

### The Skills Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      SKILL REGISTRY                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Skill A   │  │   Skill B   │  │   Skill C   │        │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │        │
│  │  │Execute│  │  │  │Execute│  │  │  │Execute│  │        │
│  │  │Validate│ │  │  │Validate│ │  │  │Validate│ │        │
│  │  │Context│  │  │  │Context│  │  │  │Context│  │        │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    SKILL ORCHESTRATOR                       │
│  • Selects appropriate skills based on intent              │
│  • Chains skills together for complex tasks                │
│  • Manages shared context between skills                   │
│  • Handles errors and fallbacks                            │
└─────────────────────────────────────────────────────────────┘
```

### When to Use Skills vs MCP:

- **Use MCP** when you need standardized tool interfaces for external integrations
- **Use Skills** when you need composable, stateful capabilities within your agent
- **Use Both** for complex agents that need both patterns!
"""

# ============================================================================
# 🎯 SKILL BASE CLASS - The Foundation
# ============================================================================

class SkillCategory(Enum):
    """Categories of skills for organization"""
    INFORMATION = "information"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    UTILITY = "utility"
    MEMORY = "memory"

@dataclass
class SkillContext:
    """
    Shared context that flows between skills.
    This is a key difference from MCP tools - skills can share state!
    """
    user_input: str = ""
    previous_results: List[Dict] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    
    def add_result(self, skill_name: str, result: Any):
        """Add a skill result to the context"""
        self.previous_results.append({
            "skill": skill_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent skill result"""
        return self.previous_results[-1] if self.previous_results else None
    
    def set_variable(self, name: str, value: Any):
        """Set a context variable for use by other skills"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.variables.get(name, default)


class Skill(ABC):
    """
    Abstract base class for all Skills.
    
    Skills are more sophisticated than simple tools:
    - They have a lifecycle (can_execute -> execute -> post_process)
    - They can access shared context
    - They can be chained together
    - They can have prerequisites
    """
    
    def __init__(self, name: str, description: str, category: SkillCategory):
        self.name = name
        self.description = description
        self.category = category
        self.prerequisites: List[str] = []  # Skills that must run first
        self.triggers: List[str] = []  # Keywords that trigger this skill
    
    @abstractmethod
    def can_execute(self, context: SkillContext) -> bool:
        """Check if this skill can be executed given the current context"""
        pass
    
    @abstractmethod
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        """Execute the skill and return results"""
        pass
    
    def post_process(self, result: Dict[str, Any], context: SkillContext) -> Dict[str, Any]:
        """Optional post-processing of results"""
        return result
    
    def get_confidence(self, user_input: str) -> float:
        """
        Calculate how confident we are that this skill should handle the input.
        Returns a value between 0 and 1.
        """
        input_lower = user_input.lower()
        matches = sum(1 for trigger in self.triggers if trigger in input_lower)
        if not self.triggers:
            return 0.0
        return min(matches / len(self.triggers) * 2, 1.0)  # Scale up, cap at 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "triggers": self.triggers,
            "prerequisites": self.prerequisites
        }

# ============================================================================
# 🛠️ CONCRETE SKILLS - Educational Examples
# ============================================================================

class WeatherSkill(Skill):
    """Skill for getting weather information"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather information for any location worldwide",
            category=SkillCategory.INFORMATION
        )
        self.triggers = ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "climate"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        # Extract location from input or kwargs
        location = kwargs.get("location") or self._extract_location(context.user_input)
        unit = kwargs.get("unit", "celsius")
        
        # Simulate weather API call
        conditions = ["sunny ☀️", "cloudy ☁️", "rainy 🌧️", "partly cloudy ⛅", "windy 💨"]
        temp_c = random.randint(10, 35)
        temp = temp_c if unit == "celsius" else int(temp_c * 9/5 + 32)
        unit_symbol = "°C" if unit == "celsius" else "°F"
        
        result = {
            "location": location,
            "temperature": temp,
            "unit": unit_symbol,
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 80),
            "wind_speed": random.randint(5, 30)
        }
        
        # Store in context for potential chaining
        context.set_variable("current_weather", result)
        context.set_variable("current_location", location)
        
        return result
    
    def _extract_location(self, text: str) -> str:
        """Extract location from natural language"""
        patterns = [
            r"(?:weather|temperature|forecast)\s+(?:in|for|at)\s+([A-Za-z\s]+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+weather",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "New York"  # Default


class CalculatorSkill(Skill):
    """Skill for mathematical calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and evaluations",
            category=SkillCategory.UTILITY
        )
        self.triggers = ["calculate", "compute", "math", "sum", "multiply", "divide", "add", "subtract", "what is", "+", "-", "*", "/"]
    
    def can_execute(self, context: SkillContext) -> bool:
        # Check for math operations or keywords
        text = context.user_input.lower()
        has_trigger = any(trigger in text for trigger in self.triggers)
        has_numbers = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', context.user_input))
        return has_trigger or has_numbers
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        expression = kwargs.get("expression") or self._extract_expression(context.user_input)
        
        try:
            # Safe evaluation
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression", "expression": expression}
            
            result = eval(expression)
            
            # Store for potential chaining
            context.set_variable("last_calculation", result)
            
            return {
                "expression": expression,
                "result": result,
                "formatted": f"{expression} = {result}"
            }
        except Exception as e:
            return {"error": str(e), "expression": expression}
    
    def _extract_expression(self, text: str) -> str:
        """Extract mathematical expression from text"""
        # Look for explicit expressions
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        if match:
            return match.group().strip()
        
        # Try to parse word-based math
        text_lower = text.lower()
        if "plus" in text_lower:
            text = text.replace("plus", "+")
        if "minus" in text_lower:
            text = text.replace("minus", "-")
        if "times" in text_lower or "multiplied by" in text_lower:
            text = text.replace("times", "*").replace("multiplied by", "*")
        if "divided by" in text_lower:
            text = text.replace("divided by", "/")
        
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        return match.group().strip() if match else "0"


class TextAnalysisSkill(Skill):
    """Skill for analyzing text content"""
    
    def __init__(self):
        super().__init__(
            name="text_analysis",
            description="Analyze text for sentiment, keywords, summary, and statistics",
            category=SkillCategory.ANALYSIS
        )
        self.triggers = ["analyze", "sentiment", "keywords", "summarize", "summary", "count words", "statistics"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        # Get text to analyze - either from kwargs, context, or extract from input
        text = kwargs.get("text") or context.get_variable("text_to_analyze") or self._extract_text(context.user_input)
        analysis_type = kwargs.get("analysis_type") or self._detect_analysis_type(context.user_input)
        
        # Basic statistics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        
        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "analysis_type": analysis_type
        }
        
        # Sentiment analysis
        if analysis_type in ["sentiment", "all"]:
            positive = ["good", "great", "excellent", "happy", "love", "amazing", "wonderful", "fantastic", "best"]
            negative = ["bad", "terrible", "hate", "awful", "sad", "poor", "worst", "horrible", "disappointing"]
            text_lower = text.lower()
            pos_score = sum(1 for w in positive if w in text_lower)
            neg_score = sum(1 for w in negative if w in text_lower)
            
            if pos_score > neg_score:
                result["sentiment"] = {"label": "positive 😊", "score": pos_score}
            elif neg_score > pos_score:
                result["sentiment"] = {"label": "negative 😞", "score": neg_score}
            else:
                result["sentiment"] = {"label": "neutral 😐", "score": 0}
        
        # Keyword extraction
        if analysis_type in ["keywords", "all"]:
            # Simple keyword extraction
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "it", "this", "that", "to", "of", "and", "for", "in", "on"}
            words_clean = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
            words_filtered = [w for w in words_clean if w not in stop_words]
            word_freq = {}
            for w in words_filtered:
                word_freq[w] = word_freq.get(w, 0) + 1
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            result["keywords"] = [{"word": w, "count": c} for w, c in top_keywords]
        
        # Store for chaining
        context.set_variable("text_analysis_result", result)
        
        return result
    
    def _extract_text(self, user_input: str) -> str:
        """Extract text to analyze from user input"""
        patterns = [
            r'analyze[:\s]+["\'](.+)["\']',
            r'analyze[:\s]+(.+)',
            r'text[:\s]+["\'](.+)["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
        return user_input
    
    def _detect_analysis_type(self, text: str) -> str:
        text_lower = text.lower()
        if "sentiment" in text_lower:
            return "sentiment"
        if "keyword" in text_lower:
            return "keywords"
        return "all"


class MemorySkill(Skill):
    """Skill for remembering and recalling information"""
    
    def __init__(self):
        super().__init__(
            name="memory",
            description="Remember information and recall it later. Can store notes, facts, and user preferences.",
            category=SkillCategory.MEMORY
        )
        self.triggers = ["remember", "recall", "forget", "what did", "store", "save", "note"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        # Initialize memory in session state if needed
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = {}
        
        action = self._detect_action(context.user_input)
        
        if action == "remember":
            # Extract what to remember
            match = re.search(r'remember\s+(?:that\s+)?(.+)', context.user_input, re.IGNORECASE)
            if match:
                memory_content = match.group(1)
                memory_key = f"memory_{len(st.session_state.agent_memory)}"
                st.session_state.agent_memory[memory_key] = {
                    "content": memory_content,
                    "timestamp": datetime.now().isoformat()
                }
                return {
                    "action": "stored",
                    "message": f"I'll remember: {memory_content}",
                    "key": memory_key
                }
        
        elif action == "recall":
            if st.session_state.agent_memory:
                memories = list(st.session_state.agent_memory.values())
                return {
                    "action": "recalled",
                    "memories": memories,
                    "count": len(memories)
                }
            return {"action": "recall", "message": "I don't have any memories stored yet."}
        
        elif action == "forget":
            count = len(st.session_state.agent_memory)
            st.session_state.agent_memory = {}
            return {"action": "forgot", "message": f"I've forgotten {count} memories."}
        
        return {"action": "unknown", "message": "I'm not sure what you want me to do with memory."}
    
    def _detect_action(self, text: str) -> str:
        text_lower = text.lower()
        if "remember" in text_lower or "store" in text_lower or "save" in text_lower:
            return "remember"
        if "recall" in text_lower or "what did" in text_lower or "what do you" in text_lower:
            return "recall"
        if "forget" in text_lower or "clear" in text_lower:
            return "forget"
        return "unknown"


class RecommendationSkill(Skill):
    """Skill that uses context from other skills to make recommendations"""
    
    def __init__(self):
        super().__init__(
            name="recommendation",
            description="Make recommendations based on context and previous skill results",
            category=SkillCategory.GENERATION
        )
        self.triggers = ["recommend", "suggest", "what should", "advice", "best"]
        self.prerequisites = []  # Can work with various other skills
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        recommendations = []
        
        # Check if we have weather context
        weather = context.get_variable("current_weather")
        if weather:
            temp = weather.get("temperature", 20)
            condition = weather.get("condition", "")
            
            if "rain" in condition.lower():
                recommendations.append("☔ Bring an umbrella today!")
            if temp > 25:
                recommendations.append("🧴 Don't forget sunscreen!")
                recommendations.append("💧 Stay hydrated!")
            if temp < 10:
                recommendations.append("🧥 Dress warmly!")
        
        # Check if we have text analysis context
        text_analysis = context.get_variable("text_analysis_result")
        if text_analysis and "sentiment" in text_analysis:
            sentiment = text_analysis["sentiment"]["label"]
            if "negative" in sentiment:
                recommendations.append("💡 The text seems negative. Consider a more positive tone?")
            if "positive" in sentiment:
                recommendations.append("✨ Great positive tone! Keep it up!")
        
        # Check calculation context
        last_calc = context.get_variable("last_calculation")
        if last_calc is not None:
            if last_calc > 1000:
                recommendations.append(f"📊 That's a large number ({last_calc})! Double-check your calculation.")
        
        # Default recommendations if no context
        if not recommendations:
            recommendations = [
                "💡 Try asking about the weather first, then ask for recommendations!",
                "📝 Analyze some text, then I can recommend improvements!",
                "🔢 Do a calculation, and I'll provide context about the result!"
            ]
        
        return {
            "recommendations": recommendations,
            "context_used": list(context.variables.keys()),
            "message": "Based on our conversation, here are my recommendations:"
        }

# ============================================================================
# 🎭 SKILL ORCHESTRATOR - The Brain
# ============================================================================

class SkillOrchestrator:
    """
    The Skill Orchestrator manages skill selection and execution.
    
    This is the "brain" of the skills-based agent:
    - Analyzes user intent
    - Selects the best skill(s) to use
    - Manages context flow between skills
    - Handles skill chaining
    """
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.context = SkillContext()
    
    def register_skill(self, skill: Skill):
        """Register a skill with the orchestrator"""
        self.skills[skill.name] = skill
    
    def get_all_skills(self) -> List[Skill]:
        """Get all registered skills"""
        return list(self.skills.values())
    
    def select_skill(self, user_input: str) -> Optional[Skill]:
        """
        Select the best skill for the given input.
        Uses confidence scoring to pick the most appropriate skill.
        """
        self.context.user_input = user_input
        
        best_skill = None
        best_confidence = 0.0
        
        for skill in self.skills.values():
            if skill.can_execute(self.context):
                confidence = skill.get_confidence(user_input)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_skill = skill
        
        return best_skill
    
    def execute_skill(self, skill: Skill, **kwargs) -> Dict[str, Any]:
        """Execute a skill and manage context"""
        # Check prerequisites
        for prereq in skill.prerequisites:
            if prereq not in [r["skill"] for r in self.context.previous_results]:
                return {
                    "error": f"Prerequisite skill '{prereq}' must be run first",
                    "skill": skill.name
                }
        
        # Execute the skill
        result = skill.execute(self.context, **kwargs)
        
        # Post-process
        result = skill.post_process(result, self.context)
        
        # Add to context
        self.context.add_result(skill.name, result)
        
        return {
            "skill": skill.name,
            "result": result,
            "context_variables": list(self.context.variables.keys())
        }
    
    def execute_chain(self, skill_names: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Execute a chain of skills in sequence"""
        results = []
        
        for name in skill_names:
            if name not in self.skills:
                results.append({"error": f"Skill '{name}' not found"})
                continue
            
            skill = self.skills[name]
            result = self.execute_skill(skill, **kwargs)
            results.append(result)
        
        return results
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point: Process natural language input.
        Selects and executes the appropriate skill.
        """
        # Update context
        self.context.user_input = user_input
        self.context.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Select skill
        skill = self.select_skill(user_input)
        
        if not skill:
            return {
                "success": False,
                "message": "I couldn't determine which skill to use for that request.",
                "available_skills": [s.name for s in self.skills.values()]
            }
        
        # Execute skill
        result = self.execute_skill(skill)
        
        return {
            "success": True,
            "skill_used": skill.name,
            "skill_description": skill.description,
            "confidence": skill.get_confidence(user_input),
            "result": result,
            "context_state": {
                "variables": list(self.context.variables.keys()),
                "history_length": len(self.context.previous_results)
            }
        }
    
    def reset_context(self):
        """Reset the context for a new conversation"""
        self.context = SkillContext()

# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

def create_orchestrator() -> SkillOrchestrator:
    """Create and configure the skill orchestrator"""
    orchestrator = SkillOrchestrator()
    
    # Register all skills
    orchestrator.register_skill(WeatherSkill())
    orchestrator.register_skill(CalculatorSkill())
    orchestrator.register_skill(TextAnalysisSkill())
    orchestrator.register_skill(MemorySkill())
    orchestrator.register_skill(RecommendationSkill())
    
    return orchestrator

def display_skill_card(skill: Skill):
    """Display a skill as a card"""
    category_colors = {
        SkillCategory.INFORMATION: "🔵",
        SkillCategory.ANALYSIS: "🟢",
        SkillCategory.GENERATION: "🟣",
        SkillCategory.UTILITY: "🟡",
        SkillCategory.MEMORY: "🔴"
    }
    
    color = category_colors.get(skill.category, "⚪")
    
    st.markdown(f"""
    **{color} {skill.name}** - {skill.category.value}
    
    {skill.description}
    
    *Triggers:* {', '.join(skill.triggers[:5])}{'...' if len(skill.triggers) > 5 else ''}
    """)

def display_context_state(context: SkillContext):
    """Display the current context state"""
    st.markdown("### 📦 Context State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variables:**")
        if context.variables:
            for key, value in context.variables.items():
                st.markdown(f"- `{key}`: {type(value).__name__}")
        else:
            st.caption("No variables stored")
    
    with col2:
        st.markdown("**Skill History:**")
        if context.previous_results:
            for result in context.previous_results[-3:]:
                st.markdown(f"- {result['skill']}")
        else:
            st.caption("No skills executed yet")

def format_skill_result(skill_name: str, result: Dict[str, Any]) -> None:
    """Format and display skill result"""
    if skill_name == "weather":
        data = result.get("result", {})
        st.success(f"🌍 Weather in {data.get('location', 'Unknown')}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature", f"{data.get('temperature')}°")
        with col2:
            st.metric("Humidity", f"{data.get('humidity')}%")
        with col3:
            st.metric("Wind", f"{data.get('wind_speed')} km/h")
        st.write(f"Condition: {data.get('condition')}")
    
    elif skill_name == "calculator":
        data = result.get("result", {})
        if data.get("error"):
            st.error(f"Error: {data['error']}")
        else:
            st.success(f"🧮 {data.get('formatted')}")
    
    elif skill_name == "text_analysis":
        data = result.get("result", {})
        st.success("📊 Text Analysis Complete")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", data.get("word_count"))
        with col2:
            st.metric("Characters", data.get("character_count"))
        with col3:
            st.metric("Sentences", data.get("sentence_count"))
        
        if "sentiment" in data:
            st.write(f"**Sentiment:** {data['sentiment']['label']}")
        
        if "keywords" in data:
            keywords = [k["word"] for k in data["keywords"]]
            st.write(f"**Keywords:** {', '.join(keywords)}")
    
    elif skill_name == "memory":
        data = result.get("result", {})
        st.info(f"🧠 {data.get('message', data.get('action'))}")
        if "memories" in data:
            for mem in data["memories"]:
                st.write(f"- {mem['content']}")
    
    elif skill_name == "recommendation":
        data = result.get("result", {})
        st.success("💡 Recommendations")
        for rec in data.get("recommendations", []):
            st.write(f"• {rec}")
    
    else:
        st.json(result)

def main():
    st.set_page_config(
        page_title="🎓 Skills-Based Agentic AI",
        page_icon="🎯",
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
            background: linear-gradient(90deg, #9b59b6 0%, #3498db 100%);
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3498db 0%, #9b59b6 100%);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize orchestrator in session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = create_orchestrator()
    
    orchestrator = st.session_state.orchestrator
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Skills Tutorial")
        st.markdown("---")
        
        st.markdown("### 📚 Learning Options")
        show_skills = st.checkbox("Show Available Skills", value=True)
        show_context = st.checkbox("Show Context State", value=True)
        show_comparison = st.checkbox("Show MCP Comparison", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Try These:")
        
        examples = [
            "What's the weather in London?",
            "Calculate 50 * 12 + 100",
            "Analyze: This product is amazing and wonderful!",
            "Remember that my favorite color is blue",
            "What do you remember?",
            "Give me recommendations",
        ]
        
        for ex in examples:
            if st.button(f"📝 {ex[:25]}...", key=f"ex_{ex}"):
                st.session_state.example_input = ex
        
        st.markdown("---")
        
        st.markdown("### ⚡ Skill Chaining")
        st.caption("Try this sequence:")
        st.code("1. Ask about weather\n2. Ask for recommendations", language=None)
        
        st.markdown("---")
        if st.button("🔄 Reset Context"):
            orchestrator.reset_context()
            st.rerun()
    
    # Main content
    st.title("🎯 Skills-Based Agentic AI")
    st.markdown("### Learn how Skills enable composable AI capabilities")
    
    # Introduction
    with st.expander("📚 What are Skills? (Click to Learn)", expanded=False):
        st.markdown(SKILLS_INTRO)
    
    # MCP Comparison
    if show_comparison:
        with st.expander("🔄 Skills vs MCP Comparison", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### MCP Tools")
                st.markdown("""
                - ✅ Standardized schemas
                - ✅ Great for external integrations
                - ✅ Stateless execution
                - ❌ No built-in context sharing
                - ❌ No skill chaining
                """)
            with col2:
                st.markdown("### Skills")
                st.markdown("""
                - ✅ Composable & chainable
                - ✅ Shared context between skills
                - ✅ Can have prerequisites
                - ✅ Confidence scoring
                - ❌ Not a standard protocol
                """)
    
    # Available Skills
    if show_skills:
        with st.expander("🎯 Available Skills", expanded=False):
            cols = st.columns(2)
            for i, skill in enumerate(orchestrator.get_all_skills()):
                with cols[i % 2]:
                    with st.container():
                        display_skill_card(skill)
                        st.markdown("---")
    
    st.markdown("---")
    
    # Context State
    if show_context:
        display_context_state(orchestrator.context)
        st.markdown("---")
    
    # Chat interface
    st.markdown("## 💬 Interact with Skills")
    
    # Check for example
    initial_value = ""
    if "example_input" in st.session_state:
        initial_value = st.session_state.example_input
        del st.session_state.example_input
    
    user_input = st.text_input(
        "Your message:",
        value=initial_value,
        placeholder="Try: What's the weather in Paris? or Calculate 25 * 4",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("🚀 Execute", type="primary")
    
    if send_button and user_input:
        # Process input
        response = orchestrator.process_input(user_input)
        
        st.markdown("---")
        
        if response["success"]:
            # Show skill selection
            st.markdown("### 🎯 Skill Selection")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skill Used", response["skill_used"])
            with col2:
                st.metric("Confidence", f"{response['confidence']:.0%}")
            with col3:
                st.metric("Context Vars", len(response["context_state"]["variables"]))
            
            st.caption(f"*{response['skill_description']}*")
            
            # Show result
            st.markdown("### 📤 Result")
            format_skill_result(response["skill_used"], response["result"])
            
            # Show context update
            if response["context_state"]["variables"]:
                st.markdown("### 📦 Context Updated")
                st.info(f"Variables available for chaining: {', '.join(response['context_state']['variables'])}")
        else:
            st.warning(response["message"])
            st.info(f"Available skills: {', '.join(response.get('available_skills', []))}")
    
    # Skill History
    if orchestrator.context.previous_results:
        st.markdown("---")
        st.markdown("### 📜 Skill Execution History")
        for i, result in enumerate(reversed(orchestrator.context.previous_results[-5:])):
            st.markdown(f"**{i+1}. {result['skill']}** - {result['timestamp'][:19]}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎓 Skills-Based Agentic AI Tutorial - GenAI Course</p>
        <p>Compare with MCP Tutorial to understand both patterns!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
