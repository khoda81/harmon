import itertools
from pathlib import Path

import chess
import numpy as np
import torch
import tqdm.auto as tqdm
import transformers

import wandb
from harmon.context_filler import ContextFiller
from harmon.dataset import PgnDataset
from harmon.tokenizer import LosslessTokenizer

SAVE_PATH = Path(__file__).parent.parent.parent / "model" / "harmon-1.0"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config = transformers.AutoConfig.from_pretrained(SAVE_PATH)
    # model = transformers.AutoModelForCausalLM.from_config(config)

    model = transformers.AutoModelForCausalLM.from_pretrained(SAVE_PATH)
    config = model.config
    run_id = "103ilpg4"

    model: transformers.GPT2LMHeadModel
    model.to(device)

    tokenizer = LosslessTokenizer()
    context_filler = ContextFiller(
        tokenizer.pad_token_id,
        context_size=512,
    )

    lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-05.pgn.zst"
    batch_size = 1
    buffer_size = 16

    last_save = 0
    save_every = 1000  # games

    run = wandb.init(
        project="harmon",
        id=run_id,
        resume="allow",
        config={
            "train_options": {
                "lr": lr,
                "batch_size": batch_size,
                "train_context_size": context_filler.context_size,
            },
            "model_config": config.to_dict(),
        },
    )

    wandb.watch(model)

    with (
        PgnDataset(url) as dataset,
        tqdm.tqdm(dataset, unit=" games") as dataset_pbar,
    ):
        # a wrapper to make the pbar think its not done
        def dataset_wrapper():
            yield from dataset_pbar

        wrapper = dataset_wrapper()

        def get_samples(n):
            for _, game in zip(range(n), wrapper):
                string_exporter = chess.pgn.StringExporter(
                    columns=None,
                    comments=False,
                    headers=False,
                    variations=False,
                )

                game_str = game.accept(string_exporter)
                yield tokenizer.encode(game_str)

        context_filler.buffer.extend(get_samples(2 * buffer_size))
        while context_filler.buffer:
            batch = context_filler.get_batch(batch_size)
            batch = np.array(batch, dtype=np.int64)
            batch = torch.tensor(batch, dtype=torch.int64, device=device)

            # update the model
            out = model.forward(input_ids=batch, labels=batch)

            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

            dataset_pbar.set_description(f"loss={out.loss.item():.2f}")
            wandb.log({"loss": out.loss})
            if len(context_filler.buffer) < buffer_size:
                context_filler.buffer.extend(get_samples(buffer_size))

            if last_save + save_every < dataset_pbar.n:
                print(f"Saving model to {SAVE_PATH}...")
                model.save_pretrained(SAVE_PATH)
                last_save += save_every

    model.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    main()
