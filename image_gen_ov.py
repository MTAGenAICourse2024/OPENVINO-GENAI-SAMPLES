import argparse
from PIL import Image
import openvino_genai

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()
    
    device = 'CPU'  # GPU, NPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)
    
    # Generate image from text prompt only (no reference image for Text2ImagePipeline)
    image_tensor = pipe.generate(
        prompt=args.prompt,
        width=512,
        height=512,
        num_inference_steps=20
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")

# Call to main function
if '__main__' == __name__:
    main()

#python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "Cartoonize the person in the refernce image with a dreamy anime style, vibrant colors, and a whimsical background"
#python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"
#python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "dessert landscape post thunder with focus on the especially starry sky including full moon at dusk golden hour cinematic lighting"
#python image_gen_ov.py ./dreamlike_anime_1_0_ov/FP16 "Starry night with full moon post  cinematic lighting"