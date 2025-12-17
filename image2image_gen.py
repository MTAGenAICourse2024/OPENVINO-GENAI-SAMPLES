import argparse
import numpy as np
from PIL import Image
import sys
import pkg_resources
import os
import openvino.runtime as ov

# First, let's check what version we have
try:
    ov_genai_version = pkg_resources.get_distribution("openvino-genai").version
    print(f"Using openvino-genai version: {ov_genai_version}")
except:
    print("Could not determine openvino-genai version")

# Import openvino modules
import openvino_genai as og

def main():
    # Path to your reference image - change this to a file that exists
    input_image_path = "input.jpg"  # Make sure this file exists!
    
    # Create a test image if it doesn't exist
    if not os.path.exists(input_image_path):
        print(f"Warning: {input_image_path} not found. Creating a test image.")
        img = Image.new('RGB', (512, 512), color='white')
        for x in range(512):
            for y in range(512):
                img.putpixel((x, y), (x % 256, y % 256, 100))
        img.save(input_image_path)
        print(f"Created test image: {input_image_path}")
    
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "stable-diffusion-ov"  # Keeping this as requested
    
    print(f"Using model: {model_path}")
    
    # Use the Image2ImagePipeline directly from openvino_genai
    try:
        print("Creating Image2ImagePipeline...")
        pipe = og.Image2ImagePipeline(model_path, device="CPU")
    except Exception as e:
        print(f"Image2ImagePipeline creation failed: {e}")
        print("Please check your model path and compatibility.")
        return
    
    # Load reference image
    try:
        # For standard SD models
        init_image = Image.open(input_image_path).convert("RGB").resize((512, 512), Image.LANCZOS)

        # For SDXL models (if using SDXL)
        # init_image = Image.open(input_image_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)

        print(f"Loaded reference image: {input_image_path} ({init_image.size})")
        # Convert to numpy array
        init_image_np = np.array(init_image)
        
        # Convert numpy array to OpenVINO tensor
        init_image_tensor = ov.Tensor(init_image_np[None,])
        
        # Define prompt
        prompt = ("Keep the monkey in the reference image but place it on a smooth gradient blue background. "
                  "High quality, detailed fur, professional studio lighting, 4k, sharp focus")
        # Add negative prompts
        negative_prompt = "blurry, text, watermark, low quality, distorted, deformed, letter-like figures in the background"

        # Fine-tune these parameters for better results
        strength = 0.5  # Lower values preserve more of the original image (0.3-0.7 is a good range)
        guidance_scale = 8.5  # Higher values follow the prompt more closely (7-10 works well)
        num_inference_steps = 50  # Sweet spot between quality and speed (30-50 is usually sufficient)

        # Generate the transformed image
        result = pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,  # Add this
            image=init_image_tensor,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            #strength=0.7,
            #guidance_scale=7.5,
            #num_inference_steps=25
        )
        
        # Convert and save the result
        output_image = Image.fromarray(result.data[0])
        output_image.save("generated_image.png")
        print("Image saved as 'generated_image.png'")
        
    except Exception as e:
        print(f"Image generation failed: {e}")
        import traceback
        traceback.print_exc()

    # First pass - basic background replacement
    #result1 = pipe.generate(
    #    prompt="Keep the monkey exactly as is, remove text background, replace with solid blue",
    #    image=init_image_tensor,
    #    strength=0.4,  # Lower strength to preserve the monkey
    #    guidance_scale=8.0,
    #    num_inference_steps=30
    #)

    # Convert intermediate result to image
    #intermediate = Image.fromarray(result1.data[0])
    #intermediate_tensor = ov.Tensor(np.array(intermediate)[None,])
    #intermediate.save("intermediate_image.png")
    #print("Intermediate image saved as 'intermediate_image.png'")

    # Second pass - enhance details
    #result2 = pipe.generate(
    #    prompt="High quality photograph of monkey on solid blue background, detailed fur, studio lighting",
    #    image=intermediate_tensor, 
    #    strength=0.3,  # Even lower to preserve the first pass
    #    guidance_scale=7.0,
    #    num_inference_steps=30
    #)

    # Convert and save the result
    #output_image = Image.fromarray(result2.data[0])
    #output_image.save("generated_image_pass2.png")
    #print("Image saved as 'generated_image_pass2.png'")

if __name__ == "__main__":
    main()

#python image2image_gen.py ./stable-diffusion-ov
