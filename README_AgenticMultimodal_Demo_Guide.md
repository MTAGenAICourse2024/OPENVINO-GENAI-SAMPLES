# 🎓 Agentic Multimodal Demo Guide

## Instructor's Guide for Demonstrating the Full Agentic Multimodal Pipeline

This guide provides a structured walkthrough for presenting the Agentic Multimodal Demo (`06_agentic_multimodal_demo.py`).

---

## 📋 Pre-Demo Setup

### 1. Environment Setup

```powershell
# Navigate to the project directory
cd C:\Users\gkamhi\OPENVINO-GENAI-SAMPLES

# Activate the virtual environment
.\venv_agentic_demo\Scripts\activate

# Run the Streamlit app
streamlit run 06_agentic_multimodal_demo.py
```

### 2. Pre-Demo Checklist

- [ ] Virtual environment activated
- [ ] Vision-Language Model downloaded (`Phi-3.5-vision-instruct-int4-ov/`)
- [ ] GPU available (recommended) or CPU fallback ready
- [ ] Sample images prepared for demo (recommended: variety of scenes)
- [ ] Sample video clip ready (optional, for video analysis demo)
- [ ] Streamlit app running on `http://localhost:8501`

### 3. Recommended Sample Images

Prepare these types of images for effective demonstration:
- 🏞️ Landscape/nature scene (good for scene analysis)
- 🏢 Urban/architectural photo (good for object detection)
- 👤 Portrait or group photo (good for describing people)
- 📄 Document with text (good for OCR demo)
- 🎨 Artwork or creative image (good for style analysis)

---

## 🎯 Learning Objectives

By the end of this demo, students will understand:

1. **Vision-Language Models (VLMs)** - How AI processes images and generates text
2. **Agentic AI Architecture** - Multi-step autonomous reasoning
3. **The Analyze → Understand → Create Pipeline** - Full generative AI cycle
4. **Workflow Orchestration** - State management and step sequencing
5. **Prompt Engineering** - Converting analysis into generation prompts

---

## 🔄 Context: Where This Fits

This demo brings together concepts from the entire course:

| Previous Demo | Concept | How It Connects |
|--------------|---------|-----------------|
| MCP Tutorial | Tool schemas | Tools become workflow steps |
| Skills Tutorial | Context & chaining | Skills become analysis stages |
| This Demo | Full pipeline | Combines vision, analysis, generation |

---

## 📚 Demo Script (30-40 minutes)

### Part 1: Introduction to Multimodal AI (5 min)

**Action**: Show the course introduction in the sidebar.

**Key Concepts to Explain**:

#### Vision-Language Models (VLMs)

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION-LANGUAGE MODEL                     │
├─────────────────────────────────────────────────────────────┤
│   ┌──────────────┐         ┌──────────────────┐             │
│   │   Image      │         │  Text Query      │             │
│   │   Input      │         │  "What is this?" │             │
│   └──────┬───────┘         └────────┬─────────┘             │
│          │                          │                        │
│          ▼                          ▼                        │
│   ┌──────────────┐         ┌──────────────────┐             │
│   │   Vision     │         │   Language       │             │
│   │   Encoder    │         │   Encoder        │             │
│   └──────┬───────┘         └────────┬─────────┘             │
│          └───────────┬──────────────┘                        │
│                      ▼                                       │
│              ┌──────────────────┐                           │
│              │   Cross-Modal    │                           │
│              │   Attention      │                           │
│              └────────┬─────────┘                           │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │   Generated      │                           │
│              │   Response       │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**Teaching Point**: "The model processes the image through a vision encoder, combines it with the text query, and generates natural language responses!"

---

### Part 2: Load the Model (3 min)

**Action**: Click "🚀 Load Model" in the sidebar.

**During Loading, Explain**:

| Component | Purpose |
|-----------|---------|
| **Model Path** | `./Phi-3.5-vision-instruct-int4-ov/` - Optimized for OpenVINO |
| **INT4 Quantization** | Reduces model size, enables faster inference |
| **Device Selection** | GPU recommended for speed, CPU works for demo |

**Wait for**: ✅ Model Ready indicator

**Teaching Point**: "This Phi-3.5-Vision model can understand images AND generate text about them. It's running locally using Intel's OpenVINO for hardware optimization."

---

### Part 3: Demo Mode 1 - Image Analysis (10 min)

**This is the core demo - show multi-step visual analysis.**

#### Step 1: Upload an Image

- Use "📁 Upload File" or "📷 Camera Capture"
- Recommend starting with a rich scene (landscape with objects)

#### Step 2: Select Workflow Type

| Workflow | Steps | Best For |
|----------|-------|----------|
| **Educational** | 3 | Course demo (recommended to start) |
| **Comprehensive** | 5 | Full capability showcase |
| **Quick** | 2 | Fast results |

**Start with "Educational"** for the clearest demonstration.

#### Step 3: Run Analysis

Click "🚀 Run Analysis" and watch the step-by-step results.

#### 📚 NEW: Quality Validation & Retry Logic

**Key Agentic Feature**: The system now validates its own outputs and retries if quality is low!

```
┌──────────────────────────────────────────────────────────────┐
│             QUALITY VALIDATION & RETRY LOOP                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Execute Step ──▶ Get Result ──▶ Quality Check               │
│                                         │                     │
│                              ┌──────────┴──────────┐          │
│                              ▼                     ▼          │
│                         Low Quality           Good Quality    │
│                              │                     │          │
│                              ▼                     ▼          │
│                     Retry (max 2x)          Continue/Done     │
│                              │                                │
│                              └──────▶ (back to Execute Step)  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Watch for these indicators during demo**:

| Indicator | Meaning |
|-----------|--------|
| 🔄 Retry 1/2 | Step produced low-quality result, retrying |
| ⚠️ Quality Check | Result quality below threshold |
| ⚠️ Max Retries | Step failed after 2 retries, proceeding anyway |

**Low-Quality Detection Criteria**:
- Very short responses (< 10 characters)
- Error phrases: "I cannot", "unable to", "sorry"
- Generic non-answers: "unclear", "not visible"

**Teaching Point**: "This is a KEY agentic pattern - the agent evaluates its own outputs! If it produces a low-quality result, it automatically retries before moving on. This is self-correction in action!"

**Example Output for a Beach Sunset Image**:

```
═══════════════════════════════════════════════════════════
Step 1: Visual Feature Detection
═══════════════════════════════════════════════════════════
Prompt: "As an AI vision system, explain what visual features 
        you detect in this image."
        
Result: "I detect a complex scene with multiple visual elements:
        - Primary subject: Setting sun on the horizon
        - Color gradient: Warm oranges and reds transitioning 
          to deep blues
        - Foreground elements: Sandy beach with gentle waves
        - Atmospheric effects: Soft clouds reflecting sunlight
        - Lighting: Golden hour backlighting creating silhouettes"
        
Processing Time: 2.34s

═══════════════════════════════════════════════════════════
Step 2: Object Relationship Analysis
═══════════════════════════════════════════════════════════
Prompt: "Demonstrate your understanding by explaining 
        relationships between objects."
        
Result: "The scene demonstrates careful compositional balance:
        - The sun serves as the focal point, positioned at 
          the rule-of-thirds intersection
        - The horizon line divides sky and sea at 2/3 height
        - Waves create leading lines toward the viewer
        - Cloud formations frame the sun symmetrically"
        
Processing Time: 1.98s

═══════════════════════════════════════════════════════════
Step 3: Creative Prompt Generation
═══════════════════════════════════════════════════════════
Prompt: "Show generative AI capabilities by creating a detailed 
        prompt to recreate this scene."
        
Result: "A breathtaking beach sunset with vibrant orange and 
        crimson sky, golden sun reflecting on calm ocean waves, 
        silhouetted clouds, sandy foreground with gentle surf, 
        professional photography, golden hour lighting, 
        highly detailed, 8k resolution, National Geographic style"
        
Processing Time: 2.12s
```

**Teaching Points After Each Step**:

| Step | Teaching Point |
|------|---------------|
| Step 1 | "Notice how the AI identifies not just objects, but visual characteristics like colors and lighting!" |
| Step 2 | "The AI demonstrates understanding of composition and spatial relationships - not just listing what it sees." |
| Step 3 | "This is the Analyze → Understand → **Create** step - turning analysis into a generation-ready prompt!" |

---

### Part 4: The Analyze → Understand → Create Pipeline (5 min)

**Action**: Show the pipeline diagram in the sidebar.

```
   INPUT                                                OUTPUT
  ┌─────┐                                              ┌─────┐
  │Image│                                              │Image│
  │Video│                                              │Text │
  └──┬──┘                                              └──▲──┘
     │                                                    │
     ▼                                                    │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   ANALYZE    │───▶│  UNDERSTAND  │───▶│   CREATE     │──┘
│              │    │              │    │              │
│ • Detection  │    │ • Reasoning  │    │ • Generate   │
│ • Features   │    │ • Context    │    │ • Synthesize │
│ • Patterns   │    │ • Relations  │    │ • Transform  │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Explain the Stages**:

1. **ANALYZE**: Raw perception - "What features do I see?"
2. **UNDERSTAND**: Reasoning - "What does this mean in context?"
3. **CREATE**: Generation - "How can I recreate or enhance this?"

**Teaching Point**: "This is the fundamental pattern for most generative AI applications. Whether it's image editing, content creation, or data analysis - you analyze, understand, then create."

---

### Part 5: Demo Mode 2 - Video Analysis (5 min)

**Optional but impressive demo of temporal understanding.**

#### Step 1: Upload a Video

- Supported: MP4, AVI, MOV, MKV
- Recommend: Short clip (< 2 minutes)

#### Step 2: Configure Frame Sampling

```
Video Timeline:
├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
0    30   60   90   120  150  180  210  240  270  300 frames
     ↓         ↓         ↓         ↓         ↓
   Frame    Frame    Frame    Frame    Frame
     1        2        3        4        5
```

| Setting | Purpose | Recommended |
|---------|---------|-------------|
| Frame Interval | Frames between samples | 30 (1 sec at 30fps) |
| Max Frames | Limit total analysis | 5 (for demo speed) |

#### Step 3: Run Analysis

**Example Output**:

```
Frame 0 (0.00s): "A person standing at the entrance of a building..."
Frame 30 (1.00s): "The same person now walking through the doorway..."
Frame 60 (2.00s): "Interior scene with the person approaching a desk..."
Frame 90 (3.00s): "Interaction with another person at the counter..."
Frame 120 (4.00s): "Transaction completed, person turning to leave..."
```

**Teaching Point**: "The AI doesn't just analyze individual frames - you can see it building a narrative understanding of what's happening over time!"

---

### Part 6: Demo Mode 3 - Comprehensive Workflow (7 min)

**Show the full 5-step comprehensive workflow.**

**Action**: Select "Comprehensive" workflow and run on a complex image.

**The 5 Steps**:

| Step | Task Type | Purpose |
|------|-----------|---------|
| 1 | DESCRIBE | What is the main subject? |
| 2 | DETECT_OBJECTS | List all identifiable objects |
| 3 | ANALYZE_SCENE | Context and storytelling |
| 4 | QUALITY_CHECK | Lighting, composition, clarity |
| 5 | CREATIVE_PROMPT | Generation-ready prompt |

**Example Analysis Flow**:

```
Step 1 (DESCRIBE): 
"A busy urban street scene during rush hour..."

Step 2 (DETECT_OBJECTS):
"Vehicles: cars, buses, bicycles
 People: pedestrians, cyclists
 Infrastructure: traffic lights, street signs, buildings
 Other: trees, street vendors, billboards"

Step 3 (ANALYZE_SCENE):
"This appears to be a commercial district during
 evening rush hour. The golden light suggests 
 late afternoon. The density of activity indicates
 a major metropolitan area..."

Step 4 (QUALITY_CHECK):
"Good: Interesting composition, dynamic movement
 Challenges: Some motion blur, busy background
 Lighting: Dramatic golden hour, strong shadows"

Step 5 (CREATIVE_PROMPT):
"Bustling city street at golden hour, busy urban
 intersection with diverse crowd, dynamic movement,
 warm evening light casting long shadows, cinematic
 street photography, detailed textures, 8k quality"
```

**Teaching Point**: "See how each step builds on the previous understanding? This is the power of agentic workflows - incremental, purposeful analysis!"

---

### Part 7: Understanding the Code Architecture (5 min)

**Show key code components (can display from IDE):**

#### The AnalysisStep Dataclass

```python
@dataclass
class AnalysisStep:
    step_number: int      # Order in workflow
    task_type: TaskType   # Type of analysis
    prompt: str           # Question to ask
    result: str = ""      # AI response
    processing_time: float = 0.0
```

#### The Workflow Execution

```python
def execute_analysis_workflow(self, image, workflow_type):
    # 1. Select predefined workflow
    workflow = workflows[workflow_type]
    
    # 2. Process image to tensor
    image_tensor = self.process_image_for_model(image)
    
    # 3. Execute each step with quality validation & retry
    max_retries = 2
    for step in workflow:
        retry_count = 0
        step_successful = False
        
        while not step_successful and retry_count <= max_retries:
            result = self.analyze_with_vlm(image_tensor, step.prompt)
            
            # 📚 AGENTIC PATTERN: Self-evaluation of output quality
            if self._is_low_quality_result(result):
                retry_count += 1
                continue  # Retry the step
            
            step_successful = True
            step.result = result
            step.retry_count = retry_count
        
    return results
```

**Teaching Point**: "This is essentially what we learned in the Skills tutorial - but now with SELF-CORRECTION! Each step validates its own output before moving on. This is a core agentic capability."

---

## 🎯 Interactive Exercises (If Time Permits)

### Exercise 1: Compare Image Types

Have students upload different types of images and observe:
- How does the AI handle photos vs. artwork?
- What details does it catch or miss?
- How do the generated prompts differ?

### Exercise 2: Prompt Quality Analysis

Take the generated creative prompt and discuss:
- What makes this a good image generation prompt?
- What's missing?
- How would you improve it?

### Exercise 3: Workflow Customization

```python
# Challenge: Design your own 4-step workflow
custom_workflow = [
    AnalysisStep(1, TaskType.DESCRIBE, "Your prompt here"),
    AnalysisStep(2, TaskType.DETECT_OBJECTS, "Your prompt here"),
    AnalysisStep(3, TaskType.ANALYZE_SCENE, "Your prompt here"),
    AnalysisStep(4, TaskType.CREATIVE_PROMPT, "Your prompt here"),
]
```

---

## ❓ Discussion Questions

Use these to engage students during or after the demo:

### Conceptual Questions

| Question | Expected Discussion Points |
|----------|---------------------------|
| "What makes this system 'agentic'?" | Multi-step reasoning, autonomous decisions, state management, **self-correction via retry** |
| "Why not just ask one big question?" | Incremental understanding, focused analysis, better results |
| "Could this replace human analysts?" | Limitations, hallucinations, need for human oversight |
| "Why does the agent retry low-quality results?" | **Self-evaluation is a core agentic capability; prevents poor results from propagating** |

### Technical Questions

| Question | Expected Discussion Points |
|----------|---------------------------|
| "Why use OpenVINO?" | Hardware optimization, local inference, no cloud dependency |
| "How is video different from images?" | Temporal understanding, frame sampling, narrative building |
| "What affects generation quality?" | Prompt specificity, model capabilities, input quality |

### Ethical Questions

| Question | Expected Discussion Points |
|----------|---------------------------|
| "What if the AI misidentifies something?" | Accuracy limits, verification needs, bias concerns |
| "Could this be used maliciously?" | Deepfakes, misinformation, need for safeguards |
| "Who owns AI-generated content?" | Copyright questions, creative attribution |

---

## 🆚 Connection to Other Demos

| Demo | Connection to Multimodal |
|------|-------------------------|
| **MCP Tutorial** | Tools → Workflow steps; Schemas → Step definitions |
| **Skills Tutorial** | Context sharing → Analysis building on previous steps |
| **LangGraph Demo** | State management → AgenticState class |
| **Image Generation** | Full pipeline → Use generated prompt for actual generation |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Check model path, ensure files downloaded |
| Slow inference | Use GPU device, reduce max_tokens |
| Out of memory | Use smaller images, close other applications |
| Gibberish output | Check prompt formatting, ensure model loaded correctly |
| Video not processing | Check codec support, try converting to MP4 |
| Frequent retries | Model may be struggling with image; try clearer input |
| Max retries reached | Low-quality input or complex scene; result may be partial |

### Understanding Retry Behavior

If you see frequent 🔄 retry indicators:
1. **Check image quality** - Blurry or dark images cause poor results
2. **Simplify the scene** - Very complex images may confuse the model
3. **Try a different workflow** - "Quick" workflow may work better for simple images

---

## 📝 Key Takeaways to Emphasize

1. **VLMs bridge vision and language** - Enabling natural interaction with visual content
2. **Agentic workflows enable complex analysis** - Breaking down tasks into purposeful steps
3. **The Analyze→Understand→Create cycle** - Fundamental pattern for generative AI
4. **State management is crucial** - Each step builds on previous understanding
5. **Prompt engineering is an art** - Quality of generated prompts affects downstream results
6. **Local inference is powerful** - OpenVINO enables on-device AI without cloud dependencies

---

## 🔗 Suggested Demo Order (Full Course)

For maximum learning progression:

```
1. MCP Tutorial (07_mcp_agentic_tutorial.py)
   └── Understand: Tools, schemas, routing
   
2. Skills Tutorial (08_skills_agentic_tutorial.py)
   └── Understand: Context, chaining, state
   
3. Agentic Multimodal Demo (06_agentic_multimodal_demo.py)
   └── Apply: Full pipeline with vision
   
4. LangGraph Visual Analysis (05_agentic_visual_analysis_langgraph.py)
   └── Advanced: Graph-based orchestration
```

---

## 📚 Additional Resources

- Course README: [README_AgenticMultimodalDemo.md](README_AgenticMultimodalDemo.md)
- OpenVINO Setup: [README_OpenVino.md](README_OpenVino.md)
- LangGraph Details: [README_AgenticVisualAnalysis_LangGraph.md](README_AgenticVisualAnalysis_LangGraph.md)
- Image Generation: [README_IMAGE_GENERATION.md](README_IMAGE_GENERATION.md)

---

*Created for the Generative AI Applications Course - January 2026*
