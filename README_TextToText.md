# Text-to-Text Generation with OpenVINO GenAI

This script demonstrates text-to-text generation using OpenVINO's Generative AI capabilities. It loads a language model and generates responses based on user-provided prompts.

## Features

- **Text Generation**: Generate responses to user prompts using a pre-trained language model.
- **Device Flexibility**: Run inference on CPU, GPU, or NPU.
- **Customizable Prompts**: Input any text prompt for the model to process.

## Prerequisites

- Python 3.8+
- OpenVINO Runtime
- OpenVINO GenAI library
- A pre-trained language model (e.g., TinyLlama-1.1B-Chat-v1.0)

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

3. Download and prepare the model:
   - Place the model files in the appropriate directory (e.g., `./TinyLlama-1.1B-Chat-v1.0/`).
   - Alternatively, adjust the `model_dir` argument to point to your model's location.

## Usage

Run the script with the following command:
```bash
python 03_text_to_text.py <model_dir> <prompt>
```

### Arguments

| Argument      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `model_dir`   | Path to the directory containing the pre-trained model                     |
| `prompt`      | The text prompt for which the model will generate a response               |

### Example

1. Generate a response using the TinyLlama model:
   ```bash
   python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "What is the capital of France?"
   ```

2. Output:
   ```
   Loading model from ./TinyLlama-1.1B-Chat-v1.0/

   Input text: What is the capital of France?
   Assistant: The capital of France is Paris.
   ```

## How It Works

1. **Model Loading**: The script initializes the OpenVINO `LLMPipeline` with the specified model directory and device.
2. **Prompt Formatting**: The user prompt is formatted using a chat template to guide the model's response.
3. **Text Generation**: The model generates a response based on the input prompt and outputs it to the console.

## Customization

- **Device Selection**: Modify the `device` variable in the script to use `CPU`, `GPU`, or `NPU`.
- **Token Limit**: Adjust the `max_new_tokens` parameter in the `GenerationConfig` to control the length of the generated response.

## Troubleshooting

- **Model Loading Errors**: Ensure the `model_dir` path is correct and the model files are present.
- **Performance Issues**: Use GPU or NPU for faster inference if available.
- **Incomplete Responses**: Increase the `max_new_tokens` parameter to allow longer outputs.

## Acknowledgments

- **OpenVINO**: For the inference engine.
- **TinyLlama**: For the pre-trained language model.
- **OpenVINO GenAI**: For the lightweight APIs enabling generative AI scenarios.
