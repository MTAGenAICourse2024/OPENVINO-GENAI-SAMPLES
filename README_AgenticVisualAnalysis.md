# 🤖 Agentic Visual Analysis System

An autonomous AI agent powered by OpenVINO GenAI that performs multi-step visual analysis tasks using vision-language models. This application demonstrates advanced agentic AI capabilities with predefined workflows for quality inspection, safety checks, inventory counting, and more.

![Agentic Visual Analysis](images/agentic_visual_analysis.png)

## 🌟 Features

### Core Capabilities
- **🔍 Autonomous Agent Architecture**: Multi-step reasoning and decision-making
- **📋 Predefined Workflows**: Ready-to-use templates for common visual tasks
- **🛠️ Custom Workflow Builder**: Create your own analysis pipelines
- **📊 Results Dashboard**: Track and review all analyses
- **💾 Export Functionality**: Download analysis reports in JSON format
- **⚡ Hardware Acceleration**: Optimized for GPU/CPU with OpenVINO

### Task Types
The agent supports six core visual analysis task types:
1. **Object Detection** - Identify and describe objects in images
2. **Quality Inspection** - Assess product quality and detect defects
3. **Safety Checks** - Identify safety hazards and compliance issues
4. **Comparison Analysis** - Compare elements within scenes
5. **Counting & Inventory** - Count and categorize items
6. **Text Extraction** - Read and extract text from images

### Predefined Workflows
- **Quality Inspection**: 3-step analysis for product quality assessment
- **Safety Inspection**: 3-step workflow for workplace safety
- **Inventory Count**: 3-step counting and categorization
- **Document Analysis**: 3-step document processing and validation
- **Scene Understanding**: 3-step comprehensive scene analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenVINO toolkit
- Webcam or image files for analysis

### Installation

1. **Install Dependencies**
```bash
pip install openvino openvino-genai streamlit pillow numpy
```

2. **Download the Vision Model**

You'll need the Phi-3.5-vision-instruct model optimized for OpenVINO:
```bash
# The model should be placed in: ./Phi-3.5-vision-instruct-int4-ov/
```

3. **Run the Application**
```bash
streamlit run 05_agentic_visual_analysis.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Initialize the Agent

1. Open the application in your browser
2. In the sidebar, verify the model path: `./Phi-3.5-vision-instruct-int4-ov/`
3. Select your device (GPU recommended for faster inference)
4. Click **"🚀 Initialize Agent"**
5. Wait for the model to load (this may take a few moments)

### 2. Perform Visual Analysis

#### **Tab 1: Live Analysis**

1. **Capture or Upload Image**
   - Choose "Camera" to capture from webcam
   - Choose "Upload File" to analyze existing images

2. **Select Workflow**
   - Choose from 5 predefined workflows
   - View the workflow steps that will be executed

3. **Customize (Optional)**
   - Check "🛠️ Customize Workflow"
   - Add custom analysis prompts
   - Select task type for each step

4. **Execute**
   - Click **"🚀 Execute Agent Workflow"**
   - Watch the agent progress through each step
   - View detailed results for each analysis step

#### **Tab 2: Batch Processing**
*(Coming Soon)* - Process multiple images with the same workflow

#### **Tab 3: Results Dashboard**

- Review all previous analyses
- View comprehensive reports
- Export results as JSON

## 🔧 Configuration

### Model Path
Default: `./Phi-3.5-vision-instruct-int4-ov/`

You can change this in the sidebar to point to different vision models compatible with OpenVINO GenAI.

### Device Selection
- **GPU**: Faster inference (recommended if available)
- **CPU**: Fallback option for systems without GPU support

### Workflow Customization

Create custom workflows programmatically:

```python
from typing import List
from dataclasses import dataclass

custom_workflow = [
    AnalysisStep(1, TaskType.OBJECT_DETECTION, "Describe the main subject"),
    AnalysisStep(2, TaskType.QUALITY_INSPECTION, "Check for anomalies"),
    AnalysisStep(3, TaskType.TEXT_EXTRACTION, "Extract any visible text"),
]
```

## 📊 Output Format

### Analysis Step Result
```json
{
  "step_number": 1,
  "task_type": "object_detection",
  "prompt": "What is the main object in this image?",
  "result": "A red car parked on a street",
  "confidence": "1.23s",
  "timestamp": "14:30:45"
}
```

### Agent Decision
```json
{
  "timestamp": "2025-11-22 14:30:45",
  "total_steps": 3,
  "findings": [
    {
      "step": 1,
      "task": "object_detection",
      "result": "A red car parked on a street"
    }
  ],
  "summary": "Step 1 (object_detection): A red car... | Step 2..."
}
```

## 🎯 Use Cases

### Manufacturing Quality Control
Use the **Quality Inspection** workflow to:
- Detect product defects
- Assess surface quality
- Rate overall product quality

### Workplace Safety Monitoring
Use the **Safety Inspection** workflow to:
- Identify safety hazards
- Check PPE compliance
- Monitor workplace conditions

### Warehouse Management
Use the **Inventory Count** workflow to:
- Count items in storage
- Categorize inventory
- Track stock levels

### Document Processing
Use the **Document Analysis** workflow to:
- Identify document types
- Extract text and data
- Validate document completeness

## 🏗️ Architecture

### VisionAgent Class
The core agent class that:
- Processes images for model input
- Executes multi-step workflows
- Makes decisions based on results
- Maintains analysis history

### AnalysisStep Dataclass
Represents each step in a workflow:
- `step_number`: Sequential position
- `task_type`: Type of analysis (enum)
- `prompt`: Question to ask the model
- `result`: Model's response
- `confidence`: Processing time
- `timestamp`: When step was executed

### Workflow Engine
- Iterates through analysis steps
- Maintains context across steps
- Provides progress updates
- Aggregates results

## 🔍 Technical Details

### Image Processing
- Automatic resizing (max 1024px)
- Conversion to OpenVINO tensor format
- Maintains aspect ratio

### Model Interaction
- Uses ChatML prompt format
- System prompt optimized for concise, factual responses
- Configurable max_new_tokens (default: 200)

### Session Management
- Streamlit session state for persistence
- Analysis history tracking
- Result caching

## 🐛 Troubleshooting

### Model Loading Errors
```
Error: Failed to load model
```
**Solution**: Verify model path and ensure all required files are present

### GPU Not Available
```
Warning: GPU device not found
```
**Solution**: Switch to CPU device or install GPU drivers

### Out of Memory
```
Error: Insufficient memory
```
**Solution**: Reduce image size or switch to CPU device

## 📦 Dependencies

- `openvino` >= 2024.0.0
- `openvino-genai` >= 2024.0.0
- `streamlit` >= 1.28.0
- `pillow` >= 10.0.0
- `numpy` >= 1.24.0

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional workflow templates
- Batch processing implementation
- Real-time video analysis
- Multi-image comparison
- Enhanced visualization

## 📄 License

This project is part of the OpenVINO GenAI Samples collection.

## 🙏 Acknowledgments

- Built with [OpenVINO](https://docs.openvino.ai/)
- Powered by [Phi-3.5-Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- UI framework: [Streamlit](https://streamlit.io/)

## 📞 Support

For issues, questions, or feedback:
- Check existing issues in the repository
- Refer to OpenVINO documentation
- Review Streamlit documentation

---

**Note**: This is an experimental application demonstrating agentic AI capabilities with vision-language models. Results may vary based on image quality, model version, and hardware configuration.
