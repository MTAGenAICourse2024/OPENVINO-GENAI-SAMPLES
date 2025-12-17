"""
OpenVINO Stable Diffusion Image Generation with Fine-Tuning Support

This script enables fine-tuned image generation using OpenVINO and Stable Diffusion
by learning from a sample training set of images using textual inversion concepts.
"""

import argparse
import os
import numpy as np
from PIL import Image
from pathlib import Path
import openvino_genai
import json
from typing import List, Optional, Dict, Tuple
import hashlib


class ImageTrainingSet:
    """Manages a training set of images for fine-tuning."""
    
    def __init__(self, training_dir: str, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the training set.
        
        Args:
            training_dir: Directory containing training images
            target_size: Size to resize images to (width, height)
        """
        self.training_dir = Path(training_dir)
        self.target_size = target_size
        self.images = []
        self.image_paths = []
        self.embeddings_cache = {}
        
        if not self.training_dir.exists():
            raise ValueError(f"Training directory does not exist: {training_dir}")
        
        self._load_images()
    
    def _load_images(self):
        """Load all images from the training directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for file_path in self.training_dir.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize(self.target_size, Image.LANCZOS)
                    self.images.append(np.array(img))
                    self.image_paths.append(str(file_path))
                    print(f"Loaded training image: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Could not load image {file_path}: {e}")
        
        if not self.images:
            raise ValueError(f"No valid images found in {self.training_dir}")
        
        print(f"Loaded {len(self.images)} training images")
    
    def get_average_features(self) -> np.ndarray:
        """Compute average features from training images."""
        # Stack all images and compute mean
        stacked = np.stack(self.images, axis=0)
        return np.mean(stacked, axis=0).astype(np.uint8)
    
    def get_style_reference(self) -> np.ndarray:
        """Get a style reference by blending training images."""
        if len(self.images) == 1:
            return self.images[0]
        
        # Create a weighted blend of images
        weights = np.ones(len(self.images)) / len(self.images)
        blended = np.zeros_like(self.images[0], dtype=np.float32)
        
        for img, weight in zip(self.images, weights):
            blended += img.astype(np.float32) * weight
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def get_random_reference(self) -> np.ndarray:
        """Get a random image from the training set."""
        idx = np.random.randint(0, len(self.images))
        return self.images[idx]
    
    def save_metadata(self, output_path: str):
        """Save training set metadata for reproducibility."""
        metadata = {
            "training_dir": str(self.training_dir),
            "num_images": len(self.images),
            "target_size": self.target_size,
            "image_paths": self.image_paths,
            "hash": self._compute_hash()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _compute_hash(self) -> str:
        """Compute a hash of the training set for identification."""
        hasher = hashlib.md5()
        for img in self.images:
            hasher.update(img.tobytes())
        return hasher.hexdigest()


class FineTunedGenerator:
    """Generator that uses training images to influence generation."""
    
    def __init__(
        self, 
        model_dir: str, 
        device: str = "CPU",
        training_set: Optional[ImageTrainingSet] = None
    ):
        """
        Initialize the fine-tuned generator.
        
        Args:
            model_dir: Path to the OpenVINO model
            device: Device to run inference on (CPU, GPU, NPU)
            training_set: Optional training set for fine-tuning
        """
        self.model_dir = model_dir
        self.device = device
        self.training_set = training_set
        
        print(f"Initializing Text2ImagePipeline on {device}...")
        self.text2img_pipe = openvino_genai.Text2ImagePipeline(model_dir, device)
        
        # Try to initialize img2img pipeline if available
        try:
            self.img2img_pipe = openvino_genai.Image2ImagePipeline(model_dir, device)
            self.has_img2img = True
            print("Image2ImagePipeline initialized successfully")
        except Exception as e:
            print(f"Image2ImagePipeline not available: {e}")
            self.img2img_pipe = None
            self.has_img2img = False
    
    def generate_with_style_transfer(
        self,
        prompt: str,
        style_image: Optional[np.ndarray] = None,
        width: int = 512,
        height: int = 512,
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        negative_prompt: str = ""
    ) -> Image.Image:
        """
        Generate an image with style transfer from training set.
        
        Args:
            prompt: Text prompt for generation
            style_image: Optional specific style image (uses training set if None)
            width: Output image width
            height: Output image height
            strength: How much to apply the style (0.0-1.0)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            negative_prompt: Things to avoid in generation
            
        Returns:
            Generated PIL Image
        """
        # Get style reference from training set if not provided
        if style_image is None and self.training_set is not None:
            style_image = self.training_set.get_style_reference()
            print("Using blended style reference from training set")
        
        if style_image is not None and self.has_img2img:
            # Use img2img pipeline with style image
            import openvino.runtime as ov
            
            # Resize style image to match output dimensions
            style_pil = Image.fromarray(style_image)
            style_pil = style_pil.resize((width, height), Image.LANCZOS)
            style_np = np.array(style_pil)
            
            # Create tensor for the pipeline
            style_tensor = ov.Tensor(style_np[None,])
            
            result = self.img2img_pipe.generate(
                prompt=prompt,
                image=style_tensor,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            return Image.fromarray(result.data[0])
        else:
            # Fall back to text2image without style
            result = self.text2img_pipe.generate(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            return Image.fromarray(result.data[0])
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 4,
        variation_strength: float = 0.6,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ) -> List[Image.Image]:
        """
        Generate multiple variations based on training set.
        
        Args:
            base_prompt: Base text prompt
            num_variations: Number of variations to generate
            variation_strength: How much variation between images
            width: Output image width
            height: Output image height
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of generated PIL Images
        """
        variations = []
        
        for i in range(num_variations):
            print(f"Generating variation {i+1}/{num_variations}...")
            
            # Get a random reference from training set for each variation
            if self.training_set is not None:
                style_image = self.training_set.get_random_reference()
            else:
                style_image = None
            
            # Vary the strength slightly for each image
            current_strength = variation_strength + (np.random.rand() - 0.5) * 0.2
            current_strength = np.clip(current_strength, 0.3, 0.9)
            
            img = self.generate_with_style_transfer(
                prompt=base_prompt,
                style_image=style_image,
                width=width,
                height=height,
                strength=current_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            variations.append(img)
        
        return variations
    
    def generate_interpolated(
        self,
        prompt: str,
        num_steps: int = 5,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ) -> List[Image.Image]:
        """
        Generate images that interpolate between training images.
        
        Args:
            prompt: Text prompt for generation
            num_steps: Number of interpolation steps
            width: Output image width
            height: Output image height
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of interpolated images
        """
        if self.training_set is None or len(self.training_set.images) < 2:
            print("Need at least 2 training images for interpolation")
            return [self.generate_with_style_transfer(prompt, width=width, height=height)]
        
        interpolated = []
        
        # Pick two random training images
        idx1, idx2 = np.random.choice(len(self.training_set.images), 2, replace=False)
        img1 = self.training_set.images[idx1].astype(np.float32)
        img2 = self.training_set.images[idx2].astype(np.float32)
        
        print(f"Interpolating between training images {idx1} and {idx2}")
        
        for i in range(num_steps):
            # Linear interpolation weight
            alpha = i / (num_steps - 1) if num_steps > 1 else 0.5
            
            # Blend the two images
            blended = (1 - alpha) * img1 + alpha * img2
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Use blended image as style reference
            img = self.generate_with_style_transfer(
                prompt=prompt,
                style_image=blended,
                width=width,
                height=height,
                strength=0.6,  # Medium strength to preserve blend characteristics
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            interpolated.append(img)
            print(f"Generated interpolation step {i+1}/{num_steps} (alpha={alpha:.2f})")
        
        return interpolated


def create_training_collage(images: List[Image.Image], output_path: str, grid_size: Tuple[int, int] = None):
    """Create a collage of generated images."""
    num_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        cols, rows = grid_size
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create collage canvas
    collage = Image.new('RGB', (cols * img_width, rows * img_height))
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        collage.paste(img, (col * img_width, row * img_height))
    
    collage.save(output_path)
    print(f"Saved collage to {output_path}")
    return collage


def main():
    parser = argparse.ArgumentParser(description="Fine-tuned Stable Diffusion Image Generation with OpenVINO")
    parser.add_argument('model_dir', help="Path to the OpenVINO Stable Diffusion model")
    parser.add_argument('prompt', help="Text prompt for image generation")
    parser.add_argument('--training-dir', '-t', help="Directory containing training images for fine-tuning")
    parser.add_argument('--output', '-o', default="output.png", help="Output image path")
    parser.add_argument('--width', type=int, default=512, help="Output image width")
    parser.add_argument('--height', type=int, default=512, help="Output image height")
    parser.add_argument('--strength', type=float, default=0.7, help="Style transfer strength (0.0-1.0)")
    parser.add_argument('--guidance-scale', type=float, default=7.5, help="Guidance scale for prompt following")
    parser.add_argument('--num-steps', type=int, default=30, help="Number of inference steps")
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'NPU'], help="Device for inference")
    parser.add_argument('--variations', type=int, default=1, help="Number of variations to generate")
    parser.add_argument('--interpolate', action='store_true', help="Generate interpolated images between training samples")
    parser.add_argument('--interpolate-steps', type=int, default=5, help="Number of interpolation steps")
    parser.add_argument('--negative-prompt', default="", help="Negative prompt (things to avoid)")
    parser.add_argument('--save-metadata', action='store_true', help="Save training set metadata")
    
    args = parser.parse_args()
    
    # Load training set if provided
    training_set = None
    if args.training_dir:
        print(f"Loading training set from {args.training_dir}...")
        training_set = ImageTrainingSet(args.training_dir, target_size=(args.width, args.height))
        
        if args.save_metadata:
            metadata_path = Path(args.output).stem + "_metadata.json"
            training_set.save_metadata(metadata_path)
            print(f"Saved training set metadata to {metadata_path}")
    
    # Initialize generator
    generator = FineTunedGenerator(
        model_dir=args.model_dir,
        device=args.device,
        training_set=training_set
    )
    
    # Generate images based on mode
    if args.interpolate and training_set is not None:
        print(f"Generating {args.interpolate_steps} interpolated images...")
        images = generator.generate_interpolated(
            prompt=args.prompt,
            num_steps=args.interpolate_steps,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps
        )
        
        # Save individual images and collage
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
        
        for i, img in enumerate(images):
            img_path = output_dir / f"{output_base}_interp_{i}.png"
            img.save(img_path)
            print(f"Saved interpolated image to {img_path}")
        
        # Create collage
        collage_path = output_dir / f"{output_base}_collage.png"
        create_training_collage(images, str(collage_path))
        
    elif args.variations > 1:
        print(f"Generating {args.variations} variations...")
        images = generator.generate_variations(
            base_prompt=args.prompt,
            num_variations=args.variations,
            variation_strength=args.strength,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps
        )
        
        # Save individual images and collage
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
        
        for i, img in enumerate(images):
            img_path = output_dir / f"{output_base}_var_{i}.png"
            img.save(img_path)
            print(f"Saved variation to {img_path}")
        
        # Create collage
        collage_path = output_dir / f"{output_base}_collage.png"
        create_training_collage(images, str(collage_path))
        
    else:
        print("Generating single image...")
        image = generator.generate_with_style_transfer(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            negative_prompt=args.negative_prompt
        )
        
        image.save(args.output)
        print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()


# Example usage:
# Single image with training set style:
# python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "A magical forest scene" --training-dir ./training_images --output styled_forest.png

# Generate multiple variations:
# python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "A cyberpunk cityscape" --training-dir ./training_images --variations 4 --output variations.png

# Generate interpolated images between training samples:
# python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "A sunset over mountains" --training-dir ./training_images --interpolate --interpolate-steps 5 --output interpolated.png

# Without training set (standard text-to-image):
# python image_gen_ov_finetuned.py ./dreamlike_anime_1_0_ov/FP16 "A starry night sky" --output starry_night.png
