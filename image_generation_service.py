"""
Image Generation Service
========================
Standalone module for generating images from text prompts using OpenVINO GenAI models.

Author: Gila Kamhi
Date: November 2025
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import openvino_genai
import logging

# Set up logging
logger = logging.getLogger("image_generation_service")
logger.setLevel(logging.INFO)

# Import utilities (non-circular)
from text_sanitization_utilities import sanitize_filename

# ⚠️ DO NOT import from translation_services here to avoid circular import
# Translation functions will be imported lazily when needed


def check_model_exists(model_dir: str) -> bool:
    """
    Check if model directory exists and has required files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        True if model exists and is valid, False otherwise
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        logger.debug(f"Model directory does not exist: {model_dir}")
        return False
    
    # Check for model_index.json (required for diffusion models)
    model_index = model_path / "model_index.json"
    if not model_index.exists():
        logger.debug(f"model_index.json not found in: {model_dir}")
        return False
    
    logger.debug(f"Valid model found at: {model_dir}")
    return True


def get_available_model_path() -> Optional[str]:
    """
    Auto-detect available model in common locations.
    Checks parent directory (..) and current directory (.).
    
    Returns:
        Path to first available model, or None if no model found
    """
    # Possible model locations (check parent directory first)
    possible_paths = [
        # Parent directory models
        "../stable-diffusion-ov",
        "../dreamlike_anime_1_0_ov/FP16",
        "../dreamlike_anime_1_0_ov",
        "../models/stable-diffusion-ov",
        "../models/dreamlike_anime_1_0_ov/FP16",
        
        # Current directory models
        "./stable-diffusion-ov",
        "./dreamlike_anime_1_0_ov/FP16",
        "./dreamlike_anime_1_0_ov",
        "./models/stable-diffusion-ov",
        "./models/dreamlike_anime_1_0_ov/FP16",
        
        # Absolute paths (Windows)
        "C:/models/stable-diffusion-ov",
        "C:/models/dreamlike_anime_1_0_ov/FP16",
    ]
    
    logger.info("🔍 Searching for image generation models...")
    
    for path in possible_paths:
        logger.debug(f"Checking: {path}")
        if check_model_exists(path):
            abs_path = os.path.abspath(path)
            logger.info(f"✅ Found model at: {abs_path}")
            return abs_path
    
    logger.warning("❌ No model found in common locations")
    return None


def list_available_models() -> list:
    """
    List all available models in common locations.
    
    Returns:
        List of tuples (model_name, model_path)
    """
    possible_paths = [
        ("Stable Diffusion (parent dir)", "../stable-diffusion-ov"),
        ("Dreamlike Anime FP16 (parent dir)", "../dreamlike_anime_1_0_ov/FP16"),
        ("Dreamlike Anime (parent dir)", "../dreamlike_anime_1_0_ov"),
        ("Stable Diffusion (current dir)", "./stable-diffusion-ov"),
        ("Dreamlike Anime FP16 (current dir)", "./dreamlike_anime_1_0_ov/FP16"),
    ]
    
    available = []
    for name, path in possible_paths:
        if check_model_exists(path):
            abs_path = os.path.abspath(path)
            available.append((name, abs_path))
    
    return available


class ImageGenerationService:
    """
    Service class for generating images from text prompts using OpenVINO models.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "GPU",
        output_dir: str = "generated_images"
    ):
        """
        Initialize the image generation service.
        
        Args:
            model_dir: Path to the OpenVINO image generation model (auto-detected if None)
            device: Device to run inference on (GPU/CPU/NPU)
            output_dir: Directory to save generated images
        """
        # Auto-detect model if not specified
        if model_dir is None:
            logger.info("No model directory specified, auto-detecting...")
            model_dir = get_available_model_path()
            
            if model_dir is None:
                available_models = list_available_models()
                
                error_msg = (
                    "❌ No image generation model found!\n\n"
                    "Expected model locations (checked):\n"
                    "  - ../stable-diffusion-ov\n"
                    "  - ../dreamlike_anime_1_0_ov/FP16\n"
                    "  - ./stable-diffusion-ov\n"
                    "  - ./dreamlike_anime_1_0_ov/FP16\n\n"
                )
                
                if available_models:
                    error_msg += "Available models:\n"
                    for name, path in available_models:
                        error_msg += f"  - {name}: {path}\n"
                else:
                    error_msg += (
                        "Please download a model:\n"
                        "  1. Run: python setup_models.py\n"
                        "  2. Or manually download from Hugging Face\n"
                        "  3. Place in parent directory (..)\n\n"
                        "See README_Image_Generation.md for instructions."
                    )
                
                raise FileNotFoundError(error_msg)
        
        # Validate specified model path
        if not check_model_exists(model_dir):
            raise FileNotFoundError(
                f"❌ Model not found at: {model_dir}\n\n"
                f"Expected to find 'model_index.json' in that directory.\n"
                f"Absolute path checked: {os.path.abspath(model_dir)}\n\n"
                f"Available models:\n" +
                "\n".join([f"  - {name}: {path}" for name, path in list_available_models()]) +
                "\n\nPlease download/convert a model first."
            )
        
        self.model_dir = os.path.abspath(model_dir)
        self.device = device
        self.output_dir = output_dir
        self._pipe = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"✅ Initialized ImageGenerationService")
        logger.info(f"   Model: {self.model_dir}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Output: {self.output_dir}")
    
    def _get_pipeline(self) -> openvino_genai.Text2ImagePipeline:
        """
        Get or create the image generation pipeline (lazy loading).
        
        Returns:
            OpenVINO Text2ImagePipeline instance
        """
        if self._pipe is None:
            logger.info(f"📦 Loading image generation model...")
            logger.info(f"   Path: {self.model_dir}")
            logger.info(f"   Device: {self.device}")
            
            try:
                self._pipe = openvino_genai.Text2ImagePipeline(self.model_dir, self.device)
                logger.info("✅ Image generation pipeline loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load pipeline: {e}")
                raise RuntimeError(
                    f"Failed to load model from: {self.model_dir}\n"
                    f"Device: {self.device}\n"
                    f"Error: {e}\n\n"
                    f"Troubleshooting:\n"
                    f"  1. Verify model files exist in: {self.model_dir}\n"
                    f"  2. Check model_index.json is present\n"
                    f"  3. Try device='CPU' if GPU fails\n"
                    f"  4. Ensure OpenVINO GenAI is installed: pip install openvino-genai"
                )
        
        return self._pipe
    
    def _lazy_import_translation(self):
        """Lazy import translation functions to avoid circular dependency."""
        try:
            from translation_services import translate_to_english, create_translation
            return translate_to_english, create_translation
        except ImportError as e:
            logger.warning(f"Translation services not available: {e}")
            return None, None
    
    def generate_image_basic(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        output_filename: Optional[str] = None
    ) -> Tuple[Image.Image, str]:
        """
        Generate an image using basic parameters.
        
        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            output_filename: Optional custom filename (auto-generated if None)
        
        Returns:
            Tuple of (PIL Image, saved file path)
        """
        try:
            logger.info(f"🎨 Generating basic image with prompt: {prompt[:50]}...")
            
            # Get pipeline
            pipe = self._get_pipeline()
            
            # Generate image
            image_tensor = pipe.generate(
                prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps
            )
            
            # Convert to PIL Image
            image = Image.fromarray(image_tensor.data[0])
            
            # Save image
            if output_filename is None:
                sanitized_prompt = sanitize_filename(prompt)
                timestamp = int(time.time())
                output_filename = f"generated_image_{sanitized_prompt}_{timestamp}.png"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Handle long filenames (Windows 260 char limit)
            if len(output_path) > 200:
                sanitized_prompt = sanitized_prompt[:100]
                output_filename = f"generated_image_{sanitized_prompt}_{timestamp}.png"
                output_path = os.path.join(self.output_dir, output_filename)
            
            image.save(output_path)
            logger.info(f"✅ Image saved to: {output_path}")
            
            return image, output_path
            
        except Exception as e:
            logger.error(f"❌ Basic image generation failed: {e}")
            raise
    
    def generate_image_advanced(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_filename: Optional[str] = None,
        enhance_prompt: bool = True
    ) -> Tuple[Image.Image, str]:
        """
        Generate an image using advanced parameters for better quality.
        
        Args:
            prompt: Text description of the image to generate
            negative_prompt: Things to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps (higher = better quality)
            guidance_scale: How closely to follow the prompt (7.5 is typical)
            output_filename: Optional custom filename
            enhance_prompt: Whether to add quality enhancement phrases
        
        Returns:
            Tuple of (PIL Image, saved file path)
        """
        try:
            # Enhance prompt with quality terms if requested
            if enhance_prompt:
                positive_enhancement = (
                    "High quality, professional studio lighting, 4k, sharp focus, "
                    "clear facial figures, detailed, vibrant colors"
                )
                enhanced_prompt = f"{prompt}, {positive_enhancement}"
            else:
                enhanced_prompt = prompt
            
            # Set default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = (
                    "blurry, spooky eyes, distorted eyes, distorted eye brows, "
                    "distorted face, low quality, bad anatomy, bad hands, "
                    "missing fingers, extra digit, fewer digits, cropped, "
                    "worst quality, text, low quality, normal quality, "
                    "jpeg artifacts, signature, watermark, username, "
                    "blurry background"
                )
            
            logger.info(f"🎨 Generating advanced image with prompt: {enhanced_prompt[:50]}...")
            logger.info(f"   Using guidance_scale={guidance_scale}, steps={num_inference_steps}")
            
            # Get pipeline
            pipe = self._get_pipeline()
            
            # Generate image with advanced parameters
            image_tensor = pipe.generate(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Convert to PIL Image
            image = Image.fromarray(image_tensor.data[0])
            
            # Save image
            if output_filename is None:
                sanitized_prompt = sanitize_filename(prompt)
                timestamp = int(time.time())
                output_filename = f"generated_image_{sanitized_prompt}_{timestamp}.png"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Handle long filenames
            if len(output_path) > 200:
                sanitized_prompt = sanitized_prompt[:100]
                output_filename = f"generated_image_{sanitized_prompt}_{timestamp}.png"
                output_path = os.path.join(self.output_dir, output_filename)
            
            image.save(output_path)
            logger.info(f"✅ Advanced image saved to: {output_path}")
            
            return image, output_path
            
        except Exception as e:
            logger.error(f"❌ Advanced image generation failed: {e}")
            raise
    
    def generate_image_with_translation(
        self,
        prompt: str,
        prompt_language: str = "English",
        target_language: str = "English",
        advanced: bool = True,
        **kwargs
    ) -> Tuple[Image.Image, str, Dict[str, str]]:
        """
        Generate an image with multilingual support.
        
        Translates the prompt to English for generation, then provides
        translations in the target language.
        
        Args:
            prompt: Text description in the prompt_language
            prompt_language: Language of the input prompt
            target_language: Language for output translations
            advanced: Use advanced generation parameters
            **kwargs: Additional parameters for generate_image_advanced/basic
        
        Returns:
            Tuple of (PIL Image, saved file path, translation dict)
        """
        try:
            # Lazy import translation functions
            translate_to_english, create_translation = self._lazy_import_translation()
            
            if translate_to_english is None:
                logger.warning("Translation not available, using prompt as-is")
                english_prompt = prompt
            else:
                # Translate prompt to English if needed
                if prompt_language != "English":
                    english_prompt = translate_to_english(prompt, prompt_language)
                    logger.info(f"Translated prompt from {prompt_language} to English")
                else:
                    english_prompt = prompt
            
            # Generate image
            if advanced:
                image, output_path = self.generate_image_advanced(
                    english_prompt,
                    **kwargs
                )
            else:
                image, output_path = self.generate_image_basic(
                    english_prompt,
                    **kwargs
                )
            
            # Prepare translations
            translations = {
                "original_prompt": prompt,
                "original_language": prompt_language,
                "english_prompt": english_prompt,
                "target_language": target_language
            }
            
            # Add target language translation if different from original
            if create_translation and target_language != "English" and target_language != prompt_language:
                translations["target_prompt"] = create_translation(
                    english_prompt,
                    "English",
                    target_language
                )
            
            return image, output_path, translations
            
        except Exception as e:
            logger.error(f"Image generation with translation failed: {e}")
            raise
    
    def generate_child_friendly_image(
        self,
        prompt: str,
        age_group: str = "5-7",
        **kwargs
    ) -> Tuple[Image.Image, str]:
        """
        Generate a child-friendly educational image.
        
        Args:
            prompt: Description of what to draw
            age_group: Target age group (e.g., "3-5", "5-7", "7-10")
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (PIL Image, saved file path)
        """
        # Create child-appropriate prompt enhancement
        child_friendly_enhancement = (
            f"child-friendly illustration suitable for age {age_group}, "
            "bright colors, simple shapes, educational, cartoon style, "
            "happy, positive, clear facial expressions, appropriate for children"
        )
        
        # Child-safe negative prompt
        child_safe_negative = (
            "scary, frightening, violent, inappropriate, dark themes, "
            "complex details, realistic gore, weapons, adult content, "
            "disturbing imagery, text, letters, words"
        )
        
        enhanced_prompt = f"{prompt}, {child_friendly_enhancement}"
        
        return self.generate_image_advanced(
            prompt=enhanced_prompt,
            negative_prompt=child_safe_negative,
            enhance_prompt=False,  # We've already enhanced it
            **kwargs
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_image_from_prompt(
    prompt: str,
    output_dir: str = "generated_images",
    model_dir: Optional[str] = None,
    device: str = "GPU",
    advanced: bool = True,
    **kwargs
) -> Tuple[Image.Image, str]:
    """
    Quick function to generate an image from a prompt.
    
    Args:
        prompt: Text description
        output_dir: Where to save the image
        model_dir: Path to model (auto-detected if None)
        device: GPU/CPU/NPU
        advanced: Use advanced parameters
        **kwargs: Additional generation parameters
    
    Returns:
        Tuple of (PIL Image, saved file path)
    """
    service = ImageGenerationService(model_dir, device, output_dir)
    
    if advanced:
        return service.generate_image_advanced(prompt, **kwargs)
    else:
        return service.generate_image_basic(prompt, **kwargs)


def create_multilingual_image(
    prompt: str,
    prompt_language: str,
    output_dir: str = "generated_images",
    **kwargs
) -> Tuple[Image.Image, str, Dict[str, str]]:
    """
    Generate an image with multilingual support.
    
    Args:
        prompt: Text description
        prompt_language: Language of the prompt
        output_dir: Where to save
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (PIL Image, file path, translations dict)
    """
    service = ImageGenerationService(output_dir=output_dir)
    return service.generate_image_with_translation(
        prompt,
        prompt_language,
        **kwargs
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Image Generation Service - Examples")
    print("=" * 60)
    
    # List available models
    print("\n🔍 Searching for available models...")
    available = list_available_models()
    
    if available:
        print(f"\n✅ Found {len(available)} model(s):")
        for name, path in available:
            print(f"   • {name}")
            print(f"     Path: {path}")
    else:
        print("\n❌ No models found!")
        print("Please download a model to parent directory (..):")
        print("   • stable-diffusion-ov")
        print("   • dreamlike_anime_1_0_ov/FP16")
        exit(1)
    
    # Example 1: Basic image generation (auto-detect model)
    print("\n" + "=" * 60)
    print("Example 1: Basic Image Generation (Auto-detect Model)")
    print("=" * 60)
    try:
        service = ImageGenerationService()  # Auto-detect model
        image, path = service.generate_image_basic(
            "A friendly elephant learning to read"
        )
        print(f"✅ Saved to: {path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 2: Advanced image generation
    print("\n" + "=" * 60)
    print("Example 2: Advanced Image Generation")
    print("=" * 60)
    try:
        image, path = service.generate_image_advanced(
            "A magical library with floating books",
            num_inference_steps=50,
            guidance_scale=7.5
        )
        print(f"✅ Saved to: {path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 3: Child-friendly image
    print("\n" + "=" * 60)
    print("Example 3: Child-Friendly Image")
    print("=" * 60)
    try:
        image, path = service.generate_child_friendly_image(
            "A happy zebra playing with alphabet blocks",
            age_group="5-7"
        )
        print(f"✅ Saved to: {path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)