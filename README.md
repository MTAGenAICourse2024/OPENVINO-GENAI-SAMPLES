# OPENVINO-GENAI-SAMPLES

This repository contains sample applications for leveraging OpenVINO's Generative AI capabilities in various scenarios, including camera-based image analysis, video analysis, text-based Q&A, speech to text, agentic workflows, and image generation.

## Overview

The repository is divided into the following main functionalities:

1. **Camera to Text Analysis**: Capture images using a camera and analyze them with AI.
2. **Video Analysis CLI**: Analyze video streams and generate descriptions for video frames.
3. **Text-to-Text Q&A**: Generate AI-based responses to user-provided text prompts.
4. **Speech to Text**: Convert spoken language from audio files or microphone input into written text for further processing or analysis.
5. **Agentic Visual Analysis**: Autonomous multi-step visual analysis with LangGraph orchestration.
6. **Image Generation**: Create images from text prompts using diffusion models.
7. **🎓 Agentic Multimodal Demo**: Educational demonstration combining all capabilities for GenAI courses.
8. **🔧 MCP Agentic Tutorial**: Model Context Protocol tools integration for agentic AI.
9. **🎯 Skills Agentic Tutorial**: Composable skills pattern for building agentic systems.

## Sample Applications

| # | Sample | Description | Run Command |
|---|--------|-------------|-------------|
| 01 | Camera to Text | Capture and analyze images with AI | `python 01_camera_to_text.py` |
| 02 | Video Analyzer CLI | Analyze video streams | `python 02_video_analyzer_cli.py` |
| 03 | Text to Text | AI text generation | `python 03_text_to_text.py` |
| 04 | Microphone to Text | Speech recognition | `python 04_microphone_to_text.py` |
| 05 | Agentic Visual Analysis | LangGraph visual analysis | `streamlit run 05_agentic_visual_analysis_langgraph.py` |
| 06 | Agentic Multimodal Demo | Full multimodal demo | `streamlit run 06_agentic_multimodal_demo.py` |
| 07 | MCP Agentic Tutorial | MCP tools integration | `streamlit run 07_mcp_agentic_tutorial.py` |
| 08 | Skills Agentic Tutorial | Basic skills pattern | `streamlit run 08_skills_agentic_tutorial.py` |
| 09 | Skills + LangGraph + LLM | Skills with Azure OpenAI/OpenVINO | `streamlit run 09_skills_agentic_tutorial_langgraph.py` |

## Documentation

Refer to the following README files for detailed instructions on each functionality:

- **Camera to Text Analysis**: [README_CameraToText.md](README_CameraToText.md)
- **Video Analysis CLI**: [README_VideoAnalysisCLI.md](README_VideoAnalysisCLI.md)
- **Text-to-Text Q&A**: [README_TextToText.md](README_TextToText.md)
- **Speech to Text**: [README_SpeechToText.md](README_SpeechToText.md)
- **Agentic Visual Analysis**: [README_AgenticVisualAnalysis.md](README_AgenticVisualAnalysis.md)
- **Agentic Visual Analysis (LangGraph)**: [README_AgenticVisualAnalysis_LangGraph.md](README_AgenticVisualAnalysis_LangGraph.md)
- **Image Generation**: [README_IMAGE_GENERATION.md](README_IMAGE_GENERATION.md)
- **🎓 Agentic Multimodal Demo**: [README_AgenticMultimodalDemo.md](README_AgenticMultimodalDemo.md)
- **🔧 MCP Agentic Tutorial**: [README_MCP_Tutorial.md](README_MCP_Tutorial.md)
- **🎯 Skills Agentic Tutorial**: [README_Skills_Tutorial.md](README_Skills_Tutorial.md)

Each README file contains installation instructions, usage examples, and troubleshooting tips specific to the corresponding functionality.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MTAGenAICourse2024/OPENVINO-GENAI-SAMPLES.git
   cd OPENVINO-GENAI-SAMPLES
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required models as specified in the individual README files.

## Azure OpenAI Configuration (for Sample 09)

To use Azure OpenAI with the Skills + LangGraph tutorial, add these to your `.env` file:

```env
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_LLM_4="gpt-4o"
```



