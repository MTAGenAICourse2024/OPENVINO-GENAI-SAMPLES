# Speech-to-Text Application

This application demonstrates speech-to-text (STT) capabilities using OpenVINO's Generative AI tools. It captures audio input from a microphone, processes it, and generates transcriptions in real-time.

## Features

- **Real-Time Transcription**: Converts microphone input into text in real-time.
- **Streamlit Interface**: Provides an easy-to-use web-based interface for interaction.
- **Device Flexibility**: Supports inference on CPU, GPU, or NPU.
- **Customizable Settings**: Adjust model paths and device preferences via the interface.

## Prerequisites

- Python 3.8+
- OpenVINO Runtime
- Streamlit
- A pre-trained speech-to-text model (e.g., Whisper family models)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MTAGenAICourse2024/OPENVINO-GENAI-SAMPLES.git
   cd OPENVINO-GENAI-SAMPLES
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the speech-to-text model:
   - Place the model files in the appropriate directory.
   - Adjust the model path in the application settings if necessary.

## Usage

1. Launch the application:
   ```bash
   python launch_stt.py
   ```

2. Open the Streamlit interface in your browser (usually at `http://localhost:8501`).

3. Configure the model path and select your preferred device (CPU/GPU).

4. Start speaking into your microphone, and view the transcriptions in real-time.

## How It Works

1. **Model Loading**: The application initializes the speech-to-text model using OpenVINO.
2. **Audio Capture**: Captures audio input from the microphone.
3. **Transcription**: Processes the audio input and generates text using the loaded model.
4. **Streamlit Interface**: Displays the transcriptions in a user-friendly web interface.

## Troubleshooting

- **Microphone Not Detected**: Ensure your microphone is connected and accessible by your system.
- **Model Loading Errors**: Verify the model path and ensure the model files exist.
- **Performance Issues**: Use GPU or NPU for faster inference if available.



## Acknowledgments

- **OpenVINO**: For the inference engine.
- **Streamlit**: For the web application framework.
- **Whisper Models**: For the pre-trained speech-to-text models.

