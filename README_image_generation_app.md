# Image Generation Studio - Streamlit Application

Interactive web application for generating AI images with a guided prompt builder interface.

## Features

✅ **Guided Prompt Builder** - Step-by-step prompt construction with dropdown menus  
✅ **Free Prompt Mode** - Direct prompt input for advanced users  
✅ **Image Gallery** - View, download, and manage generated images  
✅ **Generation History** - Track all your prompts and generations  
✅ **Batch Generation** - Create multiple variations at once  
✅ **Child-Friendly Mode** - Generate age-appropriate educational content  
✅ **Advanced Parameters** - Control inference steps, guidance scale, etc.  
✅ **Multilingual Support** - Generate from prompts in multiple languages  
✅ **Bulk Download** - Download all images as ZIP  

## Installation

```bash
# Install dependencies
pip install -r requirements_streamlit_image_gen.txt

# Ensure you have the image generation service
# Place image_generation_service.py in the same directory
```

## Quick Start

```bash
# Run the application
streamlit run streamlit_image_generation_app.py

# Or with custom port
streamlit run streamlit_image_generation_app.py --server.port 8501
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Prompt Builder Tab

The guided prompt builder uses a structured approach:

**Prompt Template:**
```
[Subject] [Action/Pose] in [Setting/Location], [Art Style], 
[Camera Angle], [Lighting] lighting, [Mood/Atmosphere] atmosphere, 
[Color Palette] color palette, [Time/Weather], [Quality/Style], 
professional composition, sharp focus, highly detailed
```

**Steps:**
1. Select a **Subject** (e.g., "A friendly animal")
2. Choose an **Action** (e.g., "reading a book")
3. Pick a **Setting** (e.g., "a cozy library")
4. Select **Art Style** (e.g., "children's book illustration")
5. Choose **Camera Angle** (e.g., "eye level")
6. Pick **Lighting** (e.g., "soft natural")
7. Select **Mood** (e.g., "warm and cozy")
8. Choose **Color Palette** (e.g., "soft pastels")
9. Pick **Time/Weather** (e.g., "sunny day")
10. Select **Quality** (e.g., "highly detailed")

Each dropdown includes a "Custom" option for full flexibility.

**Example Generated Prompt:**
```
A friendly elephant reading a book in a cozy library, children's book 
illustration, eye level, soft natural lighting, warm and cozy atmosphere, 
soft pastels color palette, sunny day, highly detailed, professional 
composition, sharp focus, highly detailed
```

### 2. Configuration (Sidebar)

**Model Settings:**
- Model Directory: Path to your OpenVINO model
- Device: GPU (fastest), CPU (compatible), or NPU
- Output Directory: Where to save images

**Generation Parameters:**
- **Advanced Mode**: Enable fine-tuned control
  - Inference Steps: 10-100 (higher = better quality)
  - Guidance Scale: 1.0-20.0 (how closely to follow prompt)
  - Auto-enhance: Add quality keywords automatically
- **Basic Mode**: Quick 20-step generation

**Image Dimensions:**
- Width: 256-1024 pixels
- Height: 256-1024 pixels
- Common: 512x512, 768x768, 1024x1024

### 3. Free Prompt Mode

For advanced users who want direct control:

```
Enter your complete prompt:
"A majestic phoenix rising from flames, digital art, 
dramatic lighting, cinematic composition, 8k, highly detailed"
```

Optional negative prompt to avoid unwanted elements.

### 4. Gallery

- View all generated images
- Download individual images
- Download all as ZIP
- Delete unwanted images
- View generation details

### 5. History

- Track all generation sessions
- Review prompts used
- See thumbnails of generated images
- Copy successful prompts for reuse

## Configuration Examples

### High Quality Settings

```python
Device: GPU
Advanced Mode: On
Inference Steps: 70
Guidance Scale: 8.5
Image Size: 768x768
```

### Fast Preview Settings

```python
Device: GPU
Advanced Mode: Off
Inference Steps: 15
Image Size: 512x512
```

### Child-Friendly Settings

```python
Child-Friendly Mode: On
Inference Steps: 40
Subject: Educational content
Mood: Happy and cheerful
Color Palette: Bright and vibrant
```

## Prompt Engineering Tips

### Good Prompts

✅ **Specific and Detailed**
```
A curious red panda learning to paint with watercolors in a bright 
art studio, children's book illustration, warm afternoon lighting, 
joyful atmosphere, soft pastel colors
```

✅ **Use Quality Descriptors**
```
highly detailed, professional composition, sharp focus, 
award-winning, 8k resolution
```

✅ **Combine Multiple Elements**
```
Subject + Action + Setting + Style + Lighting + Mood
```

### Common Mistakes

❌ **Too Vague**
```
"an animal"  # Add specifics!
```

❌ **Conflicting Descriptions**
```
"photorealistic anime"  # Choose one style
```

❌ **Too Complex**
```
Don't pack 20 different elements in one image
```

## Keyboard Shortcuts

- `Ctrl + Enter`: Generate image (when in prompt field)
- `Ctrl + S`: Save current settings
- `Ctrl + R`: Refresh gallery

## Troubleshooting

### Issue: Slow Generation

**Solution:**
- Use GPU device
- Reduce inference steps to 20-30
- Use smaller image dimensions (512x512)

### Issue: Out of Memory

**Solution:**
- Switch to CPU device
- Reduce image dimensions
- Close other applications

### Issue: Low Quality Images

**Solution:**
- Increase inference steps to 50+
- Use Advanced Mode
- Enable Auto-enhance
- Increase guidance scale to 8-9

### Issue: Model Not Found

**Solution:**
```bash
# Check model path in sidebar
Model Directory: ./dreamlike_anime_1_0_ov/FP16

# Ensure model is downloaded and converted
# Use OpenVINO Model Converter if needed
```

## Advanced Features

### Batch Generation

Generate multiple variations:
1. Set "Variations" number (1-5)
2. Click "Generate Image"
3. Get slightly different versions of the same prompt

### Negative Prompts

Tell the AI what to avoid:
```
Negative Prompt:
blurry, low quality, distorted faces, text, watermark, 
bad anatomy, deformed, ugly, scary
```

### Custom Additions

Add specific details:
```
Additional Details:
with tiny reading glasses, surrounded by floating books, 
magical sparkles in the air, wearing a cozy sweater
```

## Integration with Other Services

### Export to Video

```python
# Use generated images in video creation
from streamlit_image_generation_app import st.session_state

image_paths = [item['path'] for item in st.session_state.generated_images]
# Pass to video generation service
```

### API Integration

```python
# Call from external script
import subprocess

subprocess.run([
    "streamlit", "run", 
    "streamlit_image_generation_app.py",
    "--server.headless", "true"
])
```

## Performance Tips

1. **Reuse Model**: Keep the app running to avoid reloading the model
2. **Batch Similar Images**: Generate variations in one session
3. **Optimize Dimensions**: Use 512x512 for previews, 1024x1024 for finals
4. **Use GPU**: 10-20x faster than CPU
5. **Cache Results**: Gallery saves all images for reuse

## File Structure

```
streamlit_image_generation_app.py  # Main application
image_generation_service.py        # Backend service
text_sanitization_utilities.py     # Text processing
translation_services.py             # Multilingual support
requirements_streamlit_image_gen.txt  # Dependencies
generated_images/                   # Output directory
```

## Contributing

To add new prompt components:

1. Edit the constants at the top:
```python
SUBJECTS = [
    "Your new subject",
    # ... existing items
]
```

2. The UI will automatically update!

## License

Part of the RIO Creative Studio toolkit.

---

**RIO Creative Studio** | AI-Powered Creative Content Generation