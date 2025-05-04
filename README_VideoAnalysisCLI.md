# Video Stream Analysis CLI

This project provides a command-line interface (CLI) tool for analyzing video streams using OpenVINO's Generative AI capabilities. It supports various video sources, including local files, camera feeds, RTSP streams, and YouTube videos. The tool processes video frames, generates descriptions based on user-defined prompts, and optionally saves analyzed frames and logs.

## Features

- **Video Source Support**: Analyze video files, camera feeds, RTSP streams, or YouTube videos.
- **AI-Powered Analysis**: Uses OpenVINO's `VLMPipeline` for generating frame descriptions.
- **Custom Prompts**: Define specific prompts for frame analysis.
- **Frame Saving**: Optionally save analyzed frames as images.
- **Logging**: Save analysis results to a log file.
- **Device Flexibility**: Run inference on GPU or CPU.
- **YouTube Integration**: Automatically download and analyze YouTube videos.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MTAGenAICourse2024/OPENVINO-GENAI-SAMPLES.git
   cd OPENVINO-GENAI-SAMPLES
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the Phi-3.5 vision model:
   - Place the model in the `./Phi-3.5-vision-instruct-int4-ov/` directory.
   - Or adjust the model path using the `--model` argument.

## Usage

Run the script with the following command:
```sh
python 02_video_analyzer_cli.py --source <video_source> [options]
```

### Command-Line Arguments

| Argument         | Description                                                                 | Default                                   |
|------------------|-----------------------------------------------------------------------------|-------------------------------------------|
| `--source`       | Video source (file path, camera index, or URL) **[Required]**              | N/A                                       |
| `--model`        | Path to OpenVINO model                                                    | `./Phi-3.5-vision-instruct-int4-ov/`      |
| `--device`       | Device for inference (`GPU` or `CPU`)                                      | `GPU`                                     |
| `--interval`     | Analysis interval in seconds                                               | `1.0`                                     |
| `--prompt`       | Analysis prompt                                                           | `"What's happening in this frame?"`       |
| `--display`      | Display video frames                                                      | Disabled                                  |
| `--save-frames`  | Save analyzed frames                                                      | Disabled                                  |
| `--output-dir`   | Directory to save frames and logs                                          | `output`                                  |

### Examples

1. Analyze a video file at 1 frame per second:
   ```sh
   python 02_video_analyzer_cli.py --source video.mp4 --interval 1.0
   ```

2. Analyze a webcam feed with a custom prompt and display the video:
   ```sh
   python 02_video_analyzer_cli.py --source 0 --prompt "Are there any people in this frame?" --display
   ```

3. Analyze an RTSP stream, save frames, and use CPU for inference:
   ```sh
   python 02_video_analyzer_cli.py --source "rtsp://your-stream-url" --device CPU --save-frames
   ```

4. Analyze a YouTube video:
   ```sh
   python 02_video_analyzer_cli.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --display
   ```

## How It Works

1. **Model Loading**: The OpenVINO `VLMPipeline` is initialized with the specified model and device.
2. **Video Source Handling**: Supports local files, camera feeds, RTSP streams, and YouTube URLs.
3. **Frame Processing**: Frames are resized and converted to OpenVINO tensors for analysis.
4. **Prompt Engineering**: A system prompt guides the AI to generate concise and relevant descriptions.
5. **Output**: Results are displayed, logged, and optionally saved as images.

## Requirements

- Python 3.8+
- OpenVINO Runtime
- Dependencies listed in `requirements.txt`

## Troubleshooting

- **YouTube Support**: Ensure `pytube` is installed (`pip install pytube`).
- **Model Loading Errors**: Verify the model path and ensure the model files exist.
- **Video Source Issues**: Check the file path, camera connection, or stream URL.
- **Performance**: Use GPU for faster inference or reduce the analysis interval.



## Acknowledgments

- **OpenVINO**: For the inference engine.
- **Pytube**: For YouTube video downloading.
- **OpenCV**: For video processing.