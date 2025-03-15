from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from PIL import Image
import torch

class Molmo:
    def __init__(self):
        repo_name = "cyan2k/molmo-7B-D-bnb-4bit"
        arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}


        # load the processor
        self.processor = AutoProcessor.from_pretrained(repo_name, **arguments)

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(repo_name, **arguments)

    def call(self, image_path, system_prompt):
        # load image and prompt
        inputs = self.processor.process(
            images=[Image.open(image_path)],
            text=system_prompt,
        )


        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)