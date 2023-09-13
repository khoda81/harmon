from pathlib import Path

import chess.pgn
import numpy as np
import torch
import tqdm.auto as tqdm
import transformers

import wandb
from harmon.context_filler import ContextFiller
from harmon.dataset import Downloader, PgnDataset
from harmon.tokenizer import LosslessTokenizer

SAVE_PATH = Path(__file__).parent.parent.parent / "tmp" / "models"


def ask_user(prompt: str) -> bool:
    answer = input(f"{prompt} [Y/n]: ").lower()
    while True:
        if answer in ["", "y", "yes", "yep", "yeah"]:
            return True

        elif answer in ["n", "no", "nope", "nah"]:
            return False

        answer = input(f"Invalid answer: {answer!r}. {prompt} [Y/n]: ").lower()


def main():
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-05.pgn.zst"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "harmon-1.1"
    model_path = SAVE_PATH / model_name

    lr = 1e-5
    batch_size = 2
    buffer_size = 32
    save_every = 1000  # games

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to(device)

    except OSError:
        print(f"Failed to load the model from {model_path}.")

        if ask_user("Create new?"):
            print("Creating new model...")
            config = transformers.AutoConfig.from_pretrained(model_path)
            config.run_id = wandb.util.generate_id()
            model = transformers.AutoModelForCausalLM.from_config(config).to(device)

        else:
            print("Aborting...")
            return

    model: transformers.GPT2LMHeadModel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run = wandb.init(
        project="harmon",
        id=model.config.run_id,
        resume="allow",
        config=model.config.to_dict(),
    )

    wandb.watch(model)

    tokenizer = LosslessTokenizer(
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    context_filler = ContextFiller(
        tokenizer.pad_token_id,
        context_size=model.config.n_positions,
    )

    downloader = Downloader(url)
    dataset = PgnDataset(downloader)

    last_save = 0

    wandb.log(
        {
            "lr": lr,
            "batch_size": batch_size,
            "context_size": context_filler.context_size,
        }
    )

    with (
        tqdm.tqdm(
            iterable=None,
            total=downloader.download_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as download_pbar,
        tqdm.tqdm(dataset, unit=" games") as dataset_pbar,
    ):
        # a wrapper to make the pbar think its not done
        def dataset_wrapper():
            for game in dataset_pbar:
                download_pbar.update(downloader.downloaded - download_pbar.n)
                yield game

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
                model.save_pretrained(model_path)
                last_save += save_every

    model.save_pretrained(model_path)


if __name__ == "__main__":
    main()
