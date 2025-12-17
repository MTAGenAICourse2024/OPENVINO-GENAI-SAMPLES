# OpenVINO Image Generation Suite

A comprehensive image generation toolkit using OpenVINO GenAI with Stable Diffusion models. This suite provides text-to-image generation, image-to-image transformation, and fine-tuned generation with training set support.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Available Scripts](#available-scripts)
  - [Text-to-Image Generation](#1-text-to-image-generation-image_gen_ovpy)
  - [Image-to-Image Transformation](#2-image-to-image-transformation-image2image_genpy)
  - [Fine-Tuned Generation](#3-fine-tuned-generation-image_gen_ov_finetunedpy)
- [Model Setup](#model-setup)
- [Usage Examples](#usage-examples)
- [Parameters Reference](#parameters-reference)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

This image generation suite leverages OpenVINO for optimized inference, enabling fast and efficient image generation on Intel hardware (CPU, GPU, NPU). The toolkit includes three main generation modes:

| Script | Purpose | Use Case |
|--------|---------|----------|
| `image_gen_ov.py` | Text-to-Image | Generate images from text prompts |
| `image2image_gen.py` | Image-to-Image | Transform existing images based on prompts |
| `image_gen_ov_finetuned.py` | Fine-Tuned Generation | Style transfer with training images |

---

## Prerequisites

- Python 3.8+
- OpenVINO GenAI toolkit
- Intel hardware (CPU, integrated GPU, or NPU)
- Pre-converted Stable Diffusion model in OpenVINO format

## Installation

```bash
# Install required packages
pip install openvino openvino-genai pillow numpy

# Optional: For GPU support
pip install openvino-gpu
```

---

## Available Scripts

### 1. Text-to-Image Generation (`image_gen_ov.py`)

Generate images from text prompts using Stable Diffusion.

#### Basic Usage

```bash
python image_gen_ov.py <model_dir> "<prompt>"
```

#### Examples

```bash
# Anime-style cityscape
python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"

# Desert landscape
python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "dessert landscape post thunder with focus on the especially starry sky including full moon at dusk golden hour cinematic lighting"

# Night scene
python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "Starry night with full moon cinematic lighting"
```

#### Output

Generates `image.bmp` in the current directory.

---

### 2. Image-to-Image Transformation (`image2image_gen.py`)

Transform an existing image based on a text prompt while preserving core elements.

#### Basic Usage

```bash
python image2image_gen.py [model_dir]
```

**Note:** Requires an `input.jpg` file in the current directory. If not found, a test image is automatically created.

#### Key Parameters (in code)

```python
strength = 0.5          # Lower = preserve more original (0.3-0.7 recommended)
guidance_scale = 8.5    # Higher = follow prompt more closely (7-10 recommended)
num_inference_steps = 50  # Quality vs speed tradeoff (30-50 recommended)
```

#### Example Prompt (in code)

```python
prompt = ("Keep the monkey in the reference image but place it on a smooth gradient blue background. "
          "High quality, detailed fur, professional studio lighting, 4k, sharp focus")

negative_prompt = "blurry, text, watermark, low quality, distorted, deformed"
```

#### Output

Generates `generated_image.png` in the current directory.

---

### 3. Fine-Tuned Generation (`image_gen_ov_finetuned.py`)

Generate images influenced by a training set of reference images, enabling style transfer and variation generation.

#### Features

- **Style Transfer**: Learn and apply styles from training images
- **Variation Generation**: Create multiple variations based on training set
- **Interpolation**: Blend between different training images
- **Collage Generation**: Create grid layouts of multiple outputs

#### Basic Usage

```bash
# With training set
python image_gen_ov_finetuned.py <model_dir> "<prompt>" --training-dir ./training_images

# Without training set (standard generation)
python image_gen_ov_finetuned.py <model_dir> "<prompt>"
```

#### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model_dir` | Path to OpenVINO model | Required |
| `prompt` | Text prompt for generation | Required |
| `--training-dir` | Directory with training images | None |
| `--device` | Inference device (CPU, GPU, NPU) | CPU |
| `--width` | Output image width | 512 |
| `--height` | Output image height | 512 |
| `--strength` | Style transfer strength | 0.7 |
| `--guidance-scale` | Prompt adherence | 7.5 |
| `--num-steps` | Inference steps | 30 |
| `--output` | Output filename | output.png |
| `--variations` | Number of variations | 1 |
| `--interpolate` | Enable interpolation mode | False |
| `--interpolate-steps` | Interpolation steps | 5 |
| `--negative-prompt` | Things to avoid | "" |

#### Examples

```bash
# Generate with style from training images
python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "beautiful sunset over ocean" --training-dir ./my_style_images --strength 0.6

# Generate multiple variations
python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "portrait of a warrior" --training-dir ./training_images --variations 4

# Style interpolation between training images
python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "fantasy landscape" --training-dir ./styles --interpolate --interpolate-steps 5

# High-quality generation with negative prompt
python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "detailed city scene" --training-dir ./refs --steps 50 --guidance-scale 8.5 --negative-prompt "blurry, low quality, distorted"
```

#### Output Files

| Mode | Output |
|------|--------|
| Single | `output.png` (or custom name) |
| Variations | `output_var_1.png`, `output_var_2.png`, ... + `output_collage.png` |
| Interpolate | `output_interp_1.png`, `output_interp_2.png`, ... + `output_collage.png` |

---

## Model Setup

### Converting Stable Diffusion to OpenVINO Format

Use the export script to convert Hugging Face models:

```bash
# Example: Export dreamlike-anime model
export_stability_diffusion_ov.bat
```

### Recommended Models

| Model | Path | Best For |
|-------|------|----------|
| Dreamlike Anime | `./dreamlike_anime_1_0_ov/FP16` | Anime/stylized images |
| Stable Diffusion v1.5 | `./stable-diffusion-ov` | General purpose |

---

## Parameters Reference

### Strength (`--strength`)

Controls how much the output differs from the input/style image.

| Value | Effect |
|-------|--------|
| 0.3 | Subtle changes, preserves most of original |
| 0.5 | Balanced transformation |
| 0.7 | Significant changes, loose preservation |
| 0.9 | Almost complete regeneration |

### Guidance Scale (`--guidance-scale`)

Controls how closely the output follows the prompt.

| Value | Effect |
|-------|--------|
| 5.0 | Creative, less literal |
| 7.5 | Balanced (recommended default) |
| 10.0 | Very literal interpretation |
| 15.0+ | May cause artifacts |

### Inference Steps (`--steps`)

Tradeoff between quality and speed.

| Value | Quality | Speed |
|-------|---------|-------|
| 20 | Lower | Fast |
| 30 | Good | Moderate |
| 50 | High | Slower |
| 100 | Maximum | Very slow |

---

## Tips and Best Practices

### Prompt Engineering

1. **Be Specific**: Include details about style, lighting, and composition
   ```
   "portrait of a woman, professional studio lighting, 4k, sharp focus, detailed skin texture"
   ```

2. **Use Negative Prompts**: Exclude unwanted elements
   ```
   negative_prompt = "blurry, text, watermark, low quality, distorted, deformed"
   ```

3. **Style Keywords**: Add artistic style descriptors
   ```
   "... in the style of oil painting, impressionist, vibrant colors"
   ```

### Training Set Best Practices

1. **Consistent Style**: Use images with similar artistic style
2. **High Quality**: Use high-resolution, well-lit images
3. **Variety**: Include 5-10 diverse examples of the target style
4. **Format**: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Device Selection

| Device | Best For | Notes |
|--------|----------|-------|
| CPU | Compatibility | Works on all systems |
| GPU | Speed | Requires integrated Intel GPU |
| NPU | Power efficiency | Intel NPU required |

---

## Troubleshooting

### Common Issues

#### "Model not found" Error

```bash
# Ensure model path exists and contains required files
ls ./dreamlike_anime_1_0_ov/FP16/
# Should contain: model.xml, model.bin, etc.
```

#### "Image2ImagePipeline not available"

This is normal if your model doesn't support img2img. The system will fall back to text2image.

#### Out of Memory

- Reduce image dimensions (`--width`, `--height`)
- Use `--device CPU` if GPU memory is insufficient
- Reduce `--steps`

#### Poor Quality Output

1. Increase `--steps` (try 50)
2. Adjust `--guidance-scale` (try 7.5-10)
3. Improve prompt specificity
4. Add negative prompts

### Debug Mode

For detailed debugging, check console output which shows:
- OpenVINO GenAI version
- Model loading status
- Pipeline initialization
- Generation progress

---

## File Structure

```
openvino.genai/
├── image_gen_ov.py              # Text-to-image generation
├── image2image_gen.py           # Image-to-image transformation
├── image_gen_ov_finetuned.py    # Fine-tuned generation with training set
├── dreamlike_anime_1_0_ov/      # Example OpenVINO model
│   └── FP16/
│       ├── model.xml
│       └── model.bin
├── input/                        # Input images for img2img
├── output/                       # Generated outputs
└── README_IMAGE_GENERATION.md   # This documentation
```

---

## API Usage (Programmatic)

### Using as a Module

```python
from image_gen_ov_finetuned import ImageTrainingSet, FineTunedGenerator

# Load training set
training_set = ImageTrainingSet("./my_training_images", target_size=(512, 512))

# Initialize generator
generator = FineTunedGenerator(
    model_dir="./dreamlike_anime_1_0_ov/FP16",
    device="CPU",
    training_set=training_set
)

# Generate with style transfer
result = generator.generate_with_style_transfer(
    prompt="beautiful landscape",
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=30
)

result.save("my_output.png")
```

### Generate Variations

```python
# Generate multiple variations
images = generator.generate_variations(
    prompt="fantasy character",
    num_variations=4,
    strength=0.7
)

for i, img in enumerate(images):
    img.save(f"variation_{i}.png")
```

### Style Interpolation

```python
# Interpolate between training images
interpolated = generator.generate_interpolated(
    prompt="abstract art",
    num_steps=5
)

for i, img in enumerate(interpolated):
    img.save(f"interpolated_{i}.png")
```

---

## License

This project is part of the OpenVINO GenAI toolkit. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure any new features maintain compatibility with the existing API.

---

*Generated for OpenVINO GenAI Image Generation Suite*
