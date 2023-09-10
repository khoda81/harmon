import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.streamers import BaseStreamer

from harmon.tokenizer import LosslessTokenizer
from harmon.train import SAVE_PATH


class Streamer(BaseStreamer):
    def __init__(self, tokenizer: LosslessTokenizer):
        self.tokenizer = tokenizer
        self.tokens = []

    def put(self, token: torch.Tensor):
        self.tokens.append(token.item())
        text = self.tokenizer.decode(self.tokens)
        print("\x1b[2J\x1b[H", text, sep="")

    def end(self):
        print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(SAVE_PATH).to(device)
    tokenizer = LosslessTokenizer()

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(
        # num_samples=3,
        max_length=1024,
        # return_dict_in_generate=True,
        # output_scores=True,
        do_sample=True,
    )

    streamer = Streamer(tokenizer)
    _ = model.generate(generation_config=generation_config, streamer=streamer)


if __name__ == "__main__":
    main()
