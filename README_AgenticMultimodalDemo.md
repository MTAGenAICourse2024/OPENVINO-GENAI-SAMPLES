# 🎓 Agentic Multimodal AI Demo - Course Walkthrough

## Educational Demonstration for Generative AI Applications

This comprehensive walkthrough demonstrates **agentic AI capabilities** combining image analysis, video analysis, and image generation for a college course on Generative AI Applications.

---

## 📚 Table of Contents

1. [Introduction & Learning Objectives](#-introduction--learning-objectives)
2. [Key Concepts Explained](#-key-concepts-explained)
3. [Prerequisites & Setup](#-prerequisites--setup)
4. [Demo Walkthrough](#-demo-walkthrough)
5. [Hands-On Exercises](#-hands-on-exercises)
6. [Understanding the Code](#-understanding-the-code)
7. [Discussion Questions](#-discussion-questions)
8. [Additional Resources](#-additional-resources)

---

## 🎯 Introduction & Learning Objectives

### Course Context

This demonstration is designed for students studying **Generative AI Applications** at the college level. It provides hands-on experience with:

- **Vision-Language Models (VLMs)** - AI that understands both images and text
- **Agentic AI Systems** - Autonomous, multi-step reasoning systems
- **Multimodal Processing** - Combining different types of data (vision, text)
- **Generative Workflows** - Creating new content from analysis

### Learning Objectives

By the end of this walkthrough, students will be able to:

1. ✅ Explain how Vision-Language Models process visual information
2. ✅ Describe the architecture of an agentic AI system
3. ✅ Understand state management in AI workflows using LangGraph
4. ✅ Implement the Analyze → Understand → Create pipeline
5. ✅ Apply prompt engineering techniques for image generation
6. ✅ Evaluate the strengths and limitations of current AI systems

---

## 🧠 Key Concepts Explained

### 1. Vision-Language Models (VLMs)

**What are they?**

Vision-Language Models are AI systems that can process both visual and textual information together. They use transformer architectures to:
- Encode images into meaningful representations
- Understand natural language queries
- Generate text based on visual content

**Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION-LANGUAGE MODEL                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐         ┌──────────────────┐             │
│   │   Image      │         │  Text Query      │             │
│   │   Input      │         │  "What is this?" │             │
│   └──────┬───────┘         └────────┬─────────┘             │
│          │                           │                       │
│          ▼                           ▼                       │
│   ┌──────────────┐         ┌──────────────────┐             │
│   │   Vision     │         │   Language       │             │
│   │   Encoder    │         │   Encoder        │             │
│   │  (ViT/CNN)   │         │  (Transformer)   │             │
│   └──────┬───────┘         └────────┬─────────┘             │
│          │                           │                       │
│          └───────────┬───────────────┘                       │
│                      ▼                                       │
│              ┌──────────────────┐                           │
│              │   Cross-Modal    │                           │
│              │   Attention      │                           │
│              └────────┬─────────┘                           │
│                       │                                      │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │   Language       │                           │
│              │   Decoder        │                           │
│              └────────┬─────────┘                           │
│                       │                                      │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │   Generated      │                           │
│              │   Response       │                           │
│              └──────────────────┘                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Examples in this Demo:**
- **Phi-3.5-Vision**: Microsoft's compact but powerful VLM
- Used for image description, object detection, scene analysis

---

### 2. Agentic AI Architecture

**What makes AI "Agentic"?**

Agentic AI systems exhibit:

| Capability | Description | Example in Demo |
|------------|-------------|-----------------|
| **Planning** | Breaking tasks into steps | Multi-step analysis workflow |
| **Execution** | Performing actions autonomously | Running each analysis step |
| **Reasoning** | Making decisions based on observations | Generating summary from findings |
| **State Management** | Maintaining context across steps | Tracking analysis history |

**Agentic Workflow Pattern:**

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENTIC WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   PERCEIVE       │
                    │   (Input Data)   │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   PLAN           │
                    │   (Define Steps) │
                    └──────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │         EXECUTE LOOP          │
              │   ┌───────────────────────┐   │
              │   │   Execute Step        │   │
              │   │   ↓                   │   │
              │   │   Observe Result      │   │
              │   │   ↓                   │   │
              │   │   Update State        │   │
              │   │   ↓                   │   │
              │   │   More Steps? ─Yes──→ │   │
              │   └───────────────────────┘   │
              └───────────────────────────────┘
                              │ No
                              ▼
                    ┌──────────────────┐
                    │   REASON         │
                    │   (Aggregate)    │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   ACT            │
                    │   (Generate)     │
                    └──────────────────┘
```

---

### 3. LangGraph for Orchestration

**Why LangGraph?**

LangGraph provides a robust framework for building stateful AI workflows:

| Feature | Traditional Approach | LangGraph Approach |
|---------|---------------------|-------------------|
| State Management | Manual tracking | Built-in typed state |
| Conditional Logic | Scattered if/else | Declarative edges |
| Error Handling | Try-catch blocks | Centralized at nodes |
| Testing | Full workflow needed | Individual node testing |
| Visualization | Mental model | Graph representation |

**LangGraph Components:**

```python
# Define State
class AgentState(TypedDict):
    image_tensor: Any
    workflow: List[AnalysisStep]
    current_step: int
    results: List[AnalysisStep]
    decision: Dict[str, Any]

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("analyze", analyze_node)
workflow.add_conditional_edges("analyze", should_continue, 
                               {"continue": "analyze", 
                                "complete": "decide"})
workflow.add_node("decide", decision_node)
```

---

### 4. The Analyze → Understand → Create Pipeline

**The Full Generative AI Cycle:**

```
┌─────────────────────────────────────────────────────────────┐
│            ANALYZE → UNDERSTAND → CREATE PIPELINE            │
└─────────────────────────────────────────────────────────────┘

   INPUT                                                OUTPUT
  ┌─────┐                                              ┌─────┐
  │Image│                                              │Image│
  │Video│                                              │Text │
  │Audio│                                              │Both │
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

**Practical Example:**

1. **ANALYZE**: VLM examines an image of a sunset
   - Detects: sun, clouds, ocean, horizon
   - Notes: warm colors, peaceful scene

2. **UNDERSTAND**: System reasons about the content
   - Context: Evening beach scene
   - Mood: Serene, contemplative
   - Style: Natural photography

3. **CREATE**: Generate new content
   - Prompt: "A serene evening beach scene with warm sunset colors, 
     peaceful atmosphere, golden hour lighting, highly detailed"
   - Output: New image or enhanced version

---

## 🛠️ Prerequisites & Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16 GB | 32 GB |
| **GPU** | Intel iGPU | Intel Arc / Discrete GPU |
| **Storage** | 20 GB free | 50 GB free |
| **Python** | 3.9+ | 3.10 |

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/MTAGenAICourse2024/OPENVINO-GENAI-SAMPLES.git
cd OPENVINO-GENAI-SAMPLES

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Additional dependencies for this demo
pip install langgraph streamlit opencv-python pillow

# 5. Download the Vision-Language Model (Phi-3.5-Vision)
# Follow instructions in README_OpenVino.md
```

### Model Download

**Vision-Language Model (Required):**
```bash
# Option 1: Using Hugging Face Hub
huggingface-cli download microsoft/Phi-3.5-vision-instruct --local-dir ../Phi-3.5-vision-instruct

# Option 2: Using OpenVINO optimum-intel (recommended)
optimum-cli export openvino --model microsoft/Phi-3.5-vision-instruct --output ../Phi-3.5-vision-instruct-ov
```

**Image Generation Model (Optional):**
```bash
# For image generation capabilities
# See README_IMAGE_GENERATION.md for detailed instructions
```

---

## 🎬 Demo Walkthrough

### Starting the Demo

```bash
# Navigate to the project directory
cd OPENVINO-GENAI-SAMPLES

# Run the Streamlit application
streamlit run 06_agentic_multimodal_demo.py
```

### Demo Mode 1: 🖼️ Image Analysis

**Step-by-Step:**

1. **Load the Model**
   - Click "🚀 Load Model" in the sidebar
   - Wait for confirmation (✅ Model Ready)

2. **Upload an Image**
   - Use "📁 Upload File" or "📷 Camera Capture"
   - Supported formats: JPG, PNG, BMP

3. **Select Workflow Type**
   | Workflow | Steps | Description |
   |----------|-------|-------------|
   | Educational | 3 | Best for learning |
   | Comprehensive | 5 | Full analysis |
   | Quick | 2 | Fast results |

4. **Run Analysis**
   - Click "🚀 Run Analysis"
   - Observe the step-by-step results

5. **Review Results**
   - Each step shows: Prompt, Result, Processing Time
   - Generated creative prompt at the bottom

**Example Output:**

```
Step 1: Visual Feature Detection
Prompt: "As an AI vision system, explain what visual features you detect..."
Result: "I detect a complex scene with multiple subjects. The primary 
        visual features include: color gradients from warm to cool tones,
        geometric shapes indicating man-made structures, and organic 
        patterns suggesting natural elements..."
Processing Time: 2.34s

Step 2: Object Relationship Analysis  
Prompt: "Demonstrate your understanding by explaining relationships..."
Result: "The objects in this scene show a hierarchical arrangement with
        the main subject positioned in the foreground, creating depth.
        Secondary elements support the composition..."
Processing Time: 1.98s

Step 3: Creative Prompt Generation
Prompt: "Show generative AI capabilities by creating a detailed prompt..."
Result: "A dynamic scene featuring [main subject], surrounded by 
        [supporting elements], captured in [style], with [lighting], 
        highly detailed, professional quality, 8k resolution"
Processing Time: 2.12s
```

---

### Demo Mode 2: 🎬 Video Analysis

**Step-by-Step:**

1. **Upload a Video**
   - Supported formats: MP4, AVI, MOV, MKV
   - Recommended: Short clips (< 2 minutes)

2. **Configure Analysis**
   - Frame Interval: How often to sample (e.g., every 30 frames)
   - Max Frames: Limit analysis scope
   - Analysis Prompt: What to look for

3. **Run Analysis**
   - System extracts frames at intervals
   - Each frame analyzed independently
   - Results show temporal progression

**Understanding Frame Sampling:**

```
Video Timeline:
├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
0    30   60   90   120  150  180  210  240  270  300 frames
     ↓         ↓         ↓         ↓         ↓
   Frame    Frame    Frame    Frame    Frame
     1        2        3        4        5
   [Analyzed] [Analyzed] [Analyzed] [Analyzed] [Analyzed]
```

---

### Demo Mode 3: 🎨 Image Generation

**Understanding Prompt Engineering:**

| Component | Purpose | Example |
|-----------|---------|---------|
| **Subject** | What to generate | "A friendly robot" |
| **Action** | What it's doing | "teaching students" |
| **Setting** | Where it is | "in a futuristic classroom" |
| **Style** | Art style | "digital art" |
| **Quality** | Output quality | "highly detailed, 8k" |

**Prompt Formula:**
```
[Subject] + [Action] + [Setting] + [Style] + [Lighting] + [Quality]
```

**Example Prompts:**

```
Good Prompt:
"A curious cat exploring a magical library filled with floating books,
 watercolor painting style, soft ambient lighting, warm color palette,
 highly detailed, professional illustration, 8k resolution"

Avoid:
"cat in library" (too vague)
"make me a picture of something nice" (not specific)
```

---

### Demo Mode 4: 🔄 Full Agentic Pipeline

**The Complete Cycle:**

```
┌────────────────────────────────────────────────────────────────┐
│                    FULL AGENTIC PIPELINE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: PREPROCESS                                            │
│  ├── Load image                                                │
│  ├── Resize for model                                          │
│  └── Convert to tensor                                         │
│                                                                 │
│  Step 2: ANALYZE (Multi-step)                                  │
│  ├── Step 2.1: Describe scene                                  │
│  ├── Step 2.2: Identify objects                                │
│  └── Step 2.3: Analyze context                                 │
│                                                                 │
│  Step 3: REASON                                                │
│  ├── Aggregate findings                                        │
│  ├── Extract key concepts                                      │
│  └── Determine creative direction                              │
│                                                                 │
│  Step 4: GENERATE PROMPT                                       │
│  ├── Combine analysis insights                                 │
│  ├── Apply prompt engineering                                  │
│  └── Output structured prompt                                  │
│                                                                 │
│  Step 5: CREATE (Optional)                                     │
│  ├── Use generated prompt                                      │
│  ├── Run text-to-image model                                   │
│  └── Output new image                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Hands-On Exercises

### Exercise 1: Basic Image Analysis

**Objective:** Understand VLM capabilities

**Task:**
1. Upload 3 different types of images:
   - A photograph of a person
   - A landscape scene
   - A document or text-heavy image
   
2. Run "comprehensive" analysis on each

3. Compare results:
   - Which types of content did the AI analyze best?
   - What details did it miss?
   - How accurate were the descriptions?

**Questions to Answer:**
- How does the VLM handle different image types?
- What are the limitations you observed?

---

### Exercise 2: Workflow Comparison

**Objective:** Compare standard vs. LangGraph workflows

**Task:**
1. Review the code in:
   - `05_agentic_visual_analysis.py` (standard approach)
   - `05_agentic_visual_analysis_langgraph.py` (LangGraph approach)

2. Identify differences in:
   - State management
   - Error handling
   - Code organization

3. Create a comparison table

**Template:**

| Aspect | Standard | LangGraph | Advantage |
|--------|----------|-----------|-----------|
| State Management | ? | ? | ? |
| Modularity | ? | ? | ? |
| Testability | ? | ? | ? |

---

### Exercise 3: Prompt Engineering Lab

**Objective:** Master prompt engineering for image generation

**Task:**
1. Take the generated prompt from image analysis
2. Modify it using different techniques:
   - Add style modifiers
   - Change mood/atmosphere
   - Add quality descriptors

3. Document which modifications produce better results

**Prompt Modification Template:**

```
Original: [Generated prompt from analysis]

Modification 1 (Style Change):
[Your modified prompt]

Modification 2 (Mood Change):
[Your modified prompt]

Modification 3 (Quality Enhancement):
[Your modified prompt]
```

---

### Exercise 4: Build Your Own Workflow

**Objective:** Create a custom agentic workflow

**Task:**
Design a 4-step workflow for a specific use case:
- E-commerce product analysis
- Medical image review (simulated)
- Social media content moderation
- Educational content creation

**Workflow Template:**

```python
my_workflow = [
    AnalysisStep(1, TaskType.DESCRIBE, 
                 "Your prompt here"),
    AnalysisStep(2, TaskType.DETECT_OBJECTS, 
                 "Your prompt here"),
    AnalysisStep(3, TaskType.ANALYZE_SCENE, 
                 "Your prompt here"),
    AnalysisStep(4, TaskType.CREATIVE_PROMPT, 
                 "Your prompt here"),
]
```

---

## 💻 Understanding the Code

### Key Code Components

#### 1. EducationalAgenticDemo Class

```python
class EducationalAgenticDemo:
    """Main demonstration class showcasing agentic capabilities"""
    
    def __init__(self, vlm_pipeline=None):
        self.vlm_pipeline = vlm_pipeline
        self.analysis_history = []
        self.workflow_log = []
    
    def process_image_for_model(self, image):
        """Convert PIL Image to OpenVINO Tensor"""
        # Preprocessing pipeline:
        # 1. Resize large images
        # 2. Convert to numpy array
        # 3. Reshape to [batch, height, width, channels]
        # 4. Create OpenVINO tensor
        pass
    
    def analyze_with_vlm(self, image_tensor, prompt):
        """Execute VLM analysis"""
        # Uses chat template format:
        # <|im_start|>system\n...\n<|im_end|>
        # <|im_start|>user\n...\n<|im_end|>
        # <|im_start|>assistant\n
        pass
    
    def execute_analysis_workflow(self, image, workflow_type):
        """Run multi-step analysis"""
        # 1. Select predefined workflow
        # 2. Process image
        # 3. Execute each step sequentially
        # 4. Return collected results
        pass
```

#### 2. Workflow Definition

```python
@dataclass
class AnalysisStep:
    step_number: int      # Order in workflow
    task_type: TaskType   # Type of analysis
    prompt: str           # Question to ask
    result: str = ""      # AI response
    processing_time: float = 0.0
```

#### 3. OpenVINO Integration

```python
# Load model with OpenVINO GenAI
pipeline = ov_genai.VLMPipeline(model_path, device="GPU")

# Generate response
result = pipeline.generate(
    prompt,
    image=image_tensor,
    max_new_tokens=200
)
```

---

## ❓ Discussion Questions

### Conceptual Questions

1. **On Agentic AI:**
   - What distinguishes an "agent" from a simple model inference?
   - How does state management enable more complex behaviors?
   - What are the risks of fully autonomous AI agents?

2. **On Multimodal AI:**
   - Why is combining vision and language powerful?
   - What are the challenges in aligning visual and textual representations?
   - How might multimodal AI change human-computer interaction?

3. **On Generative AI:**
   - What ethical considerations arise when AI can create realistic content?
   - How do you evaluate the quality of generated content?
   - What are the implications for creative industries?

### Technical Questions

4. **On Implementation:**
   - Why use LangGraph instead of simple function chains?
   - How does OpenVINO optimize model inference?
   - What factors affect generation quality?

5. **On Scalability:**
   - How would you deploy this system in production?
   - What monitoring would you implement?
   - How would you handle multiple concurrent users?

---

## 📖 Additional Resources

### Documentation

- [OpenVINO GenAI Documentation](https://docs.openvino.ai/2024/openvino-workflow/generative-ai.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Research Papers

- "Phi-3 Technical Report" - Microsoft Research
- "Attention Is All You Need" - Vaswani et al.
- "An Image is Worth 16x16 Words" - Dosovitskiy et al. (ViT)

### Related Demos in This Repository

| Demo | File | Description |
|------|------|-------------|
| Camera Analysis | `01_camera_to_text.py` | Real-time camera → text |
| Video CLI | `02_video_analyzer_cli.py` | Command-line video analysis |
| Text-to-Text | `03_text_to_text.py` | Pure text generation |
| Speech-to-Text | `04_microphone_to_text.py` | Audio transcription |
| Agentic (Standard) | `05_agentic_visual_analysis.py` | Basic agentic workflow |
| Agentic (LangGraph) | `05_agentic_visual_analysis_langgraph.py` | LangGraph-based workflow |
| Image Generation | `image_generation_app.py` | Text-to-image creation |

---

## 🏁 Conclusion

This walkthrough demonstrated the power of combining multiple AI capabilities into coherent agentic workflows. Key takeaways:

1. **VLMs enable rich visual understanding** - AI can describe, analyze, and reason about images
2. **Agentic systems provide structure** - Multi-step workflows enable complex tasks
3. **LangGraph simplifies orchestration** - State management and conditional routing made easy
4. **The full pipeline is powerful** - Analyze → Understand → Create opens creative possibilities

### What's Next?

- Extend the demo with additional analysis types
- Integrate speech-to-text for voice commands
- Add image generation for complete pipeline
- Explore fine-tuning models for specific domains

---

*🎓 This educational demo was created for the Generative AI Applications course. For questions or contributions, please submit issues to the repository.*

---

**Author:** GenAI Course Team  
**Last Updated:** January 2026  
**Version:** 1.0
