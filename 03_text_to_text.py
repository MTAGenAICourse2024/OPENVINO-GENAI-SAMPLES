import argparse
#Will run model on CPU, GPU or NPU are possible options
import openvino_genai as ov_genai
import sys


def streamer(subword):
    print(subword, end="", flush=True)
    sys.stdout.flush()
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()
    #model = openvino_genai.load_model('phi3.5-vision', args.model_dir, device)
    device = 'CPU'  # GPU, NPU can be used as well
   
    model_dir="./TinyLlama-1.1B-Chat-v1.0/"
    model_dir = args.model_dir
    prompt = args.prompt

    print(f"Loading model from {model_dir}\n")


    pipe = ov_genai.LLMPipeline(str(model_dir), device)


    print(f"Input text: {prompt}")

    template = "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
    prompt = template.format(prompt)
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 128
    #generation_config.apply_chat_template = False
    #result = pipe.generate(input_prompt, generation_config, streamer)
   
    print(pipe.generate(prompt, generation_config))
    #print(pipe.generate(args.prompt, max_new_tokens=100))
   
# Call to main function
if '__main__' == __name__:
    main()

#python text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Sun is yellow because"
#python text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "My favorite breakfast is"