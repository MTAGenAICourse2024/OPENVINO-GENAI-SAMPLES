# OPENVINO-GENAI-SAMPLES
Samples for image and audio analysis 



## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MTAGenAICourse2024/OPENVINO-GENAI-SAMPLES.git
   cd OPENVINO-GENAI-SAMPLES
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the Phi-3.5 vision model:
   ```
   # Download and place in ./Phi-3.5-vision-instruct-int4-ov/ directory
   # Or adjust the model path in the application settings
   ```



## Camera Capture Image Analysis  : 01_camera_to_text.py 
### Usage  Image Analysis via VLM  via simple GUI 

1. Start the application:
   ```
   streamlit run 01_camera_to_text.py
   ```

2. In the sidebar, configure the model path and select your preferred device (GPU/CPU)

3. Click "Load Model" to initialize the AI

4. Take a photo using the camera input

5. Enter a prompt about the image (e.g., "What objects are in this image?")

6. Click "Generate" to analyze the image

7. View the AI's response and check your history of previous analyses
## Screenshot

![Screenshot of the Camera Capture Image Analysis application showing the sidebar with model settings, camera input, and text analysis results](./images/camera_to_text.png)

*The application interface displays camera input, model configuration options, and generated text analysis of captured images.*

## How It Works

1. **Model Loading**: The application initializes the Phi-3.5 vision model using OpenVINO
 
2. **Image Capture**: Images are captured through your device's camera

3. **Image Processing**: Large images are automatically resized to prevent memory issues

4. **Prompt Engineering**: A carefully designed system prompt guides the model to provide relevant, focused answers

5. **Response Generation**: The model analyzes the image and responds to your specific question

6. **Response Tracking**: All analyses are saved to the session history for future reference

### Performance Optimization

- Images are automatically resized to a maximum dimension of 1024 pixels
- The model is loaded only once and cached in the session state
- GPU acceleration is available for faster inference

### Hardware Requirements

- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU, Intel integrated graphics or discrete GPU
- **For optimal performance**: 16GB RAM, modern CPU with Intel GPU/NPU or discrete GPU

### Troubleshooting

- **Camera not working**: Ensure your browser has permission to access your camera
- **Model loading error**: Check that the model path is correct and the model files exist
- **Slow performance**: Try switching to GPU mode if available, or reduce the maximum token count
- **Out of memory errors**: Reduce the image size further by changing the `max_size` variable in the code




## Contributing

Created by gilaka



## Acknowledgments

- OpenVINO for the inference engine
- Microsoft for the Phi-3.5 vision model
- Streamlit for the web application framework


## Appendix 


### Supported Generative AI scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run following Generative Scenarios:
 - Text generation using Large Language Models. For example, chat with local LLaMa model
 - Image generation using Diffuser models, for example, generation using Stable Diffusion models
 - Speech recognition using Whisper family models
 - Text generation using Large Visual Models, for instance, Image analysis using LLaVa or miniCPM models family

Library efficiently supports LoRA adapters for Text and Image generation scenarios:
- Load multiple adapters per model
- Select active adapters for every generation
- Mix multiple adapters with coefficients via alpha blending

All scenarios are run on top of OpenVINO Runtime that supports inference on CPU, GPU and NPU. See [here](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html) for platform support matrix.

### Supported Generative AI optimization methods

OpenVINO™ GenAI library provides a transparent way to use state-of-the-art generation optimizations:
- Speculative decoding that employs two models of different sizes and uses the large model to periodically correct the results of the small model. See [here](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for more detailed overview
- KVCache token eviction algorithm that reduces the size of the KVCache by pruning less impacting tokens.

Additionally, OpenVINO™ GenAI library implements a continuous batching approach to use OpenVINO within LLM serving. Continuous batching library could be used in LLM serving frameworks and supports the following features:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query. See [here](https://google.com) for more detailed overview

Continuous batching functionality is used within OpenVINO Model Server (OVMS) to serve LLMs, see [here](https://docs.openvino.ai/2024/ovms_docs_llm_reference.html) for more details.

### Installing OpenVINO GenAI

```sh
    # Installing OpenVINO GenAI via pip
    pip install openvino-genai

    # Install optimum-intel to be able to download, convert and optimize LLMs from Hugging Face
    # Optimum is not required to run models, only to convert and compress
    pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git

    # (Optional) Install (TBD) to be able to download models from Model Scope
```

### Performing text generation 
<details>

For more examples check out our [LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)

### Converting and compressing text generation model from Hugging Face library

```sh
#(Basic) download and convert to OpenVINO TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"

#(Recommended) download, convert to OpenVINO and compress to int4 TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```

#### Run generation using LLMPipeline API in Python

```python
import openvino_genai as ov_genai
#Will run model on CPU, GPU or NPU are possible options
pipe = ov_genai.LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

####  Run generation using LLMPipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100)) << '\n';
}
```




---