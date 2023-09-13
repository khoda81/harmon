import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.streamers import BaseStreamer

from harmon.tokenizer import LosslessTokenizer
from harmon.train import SAVE_PATH


class Streamer(BaseStreamer):
    def __init__(self, tokenizer: LosslessTokenizer):
        self.tokenizer = tokenizer
        self.tokens = []
        self.printed = 0

    def put(self, token: torch.Tensor):
        if token.numel() == 1:
            self.tokens.append(token.item())
        elif len(token.shape) == 2 and token.shape[0] == 1:
            self.tokens.extend(token[0].tolist())
        else:
            raise ValueError(f"token has invalid shape={token.shape}")

        new_text = self.tokenizer.decode(self.tokens[:-8])
        print(new_text[self.printed :], end="")
        sys.stdout.flush()
        self.printed = len(new_text)

    def end(self):
        new_text = self.tokenizer.decode(self.tokens)
        print(new_text[self.printed :])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model_name = "harmon-tiny"
    model_path = SAVE_PATH / model_name
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = LosslessTokenizer()

    model: transformers.generation

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(
        # num_samples=3,
        max_length=1024,
        # return_dict_in_generate=True,
        # output_scores=True,
        do_sample=True,
    )

    def generate(prompt=""):
        streamer = Streamer(tokenizer)
        inputs = tokenizer.encode(prompt)
        inputs = torch.tensor([inputs], device=device, dtype=torch.int64)

        attention_mask = torch.ones_like(inputs)

        _ = model.generate(
            inputs=inputs,
            generation_config=generation_config,
            streamer=streamer,
            attention_mask=attention_mask,
        )

    generate()

    for line in sys.stdin:
        generate(line.rstrip("\n"))


if __name__ == "__main__":
    main()
