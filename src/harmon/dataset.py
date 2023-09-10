import io

import chess.pgn
import requests

import tqdm.auto as tqdm
import zstandard as zstd


class PgnDataset:
    def __init__(self, url: str):
        self.url = url

    def __enter__(self) -> "PgnDataset":
        try:
            self.response = requests.get(self.url, stream=True).__enter__()
            self.response.raise_for_status()

        except requests.RequestException as e:
            print("Error while downloading:", e)

        total_content_length = int(self.response.headers.get("content-length", 0))
        cctx = zstd.ZstdDecompressor()
        self.decompressed_stream = cctx.stream_reader(self.response.raw).__enter__()
        self.pgn_stream = io.TextIOWrapper(self.decompressed_stream).__enter__()
        self.download_pbar = tqdm.tqdm(
            iterable=None,
            total=total_content_length,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ).__enter__()

        return self

    def __iter__(self):
        return self

    def __next__(self) -> chess.pgn.Game:
        while True:
            game = chess.pgn.read_game(self.pgn_stream)
            self.download_pbar.update(self.response.raw.tell() - self.download_pbar.n)

            # TODO we should raise StopIteration at the end of stream

            if game is None:
                print("Failed to parse game")
                continue

            return game

    def __exit__(self, exc_type, exc_value, traceback):
        self.download_pbar.__exit__(exc_type, exc_value, traceback)
        self.pgn_stream.__exit__(exc_type, exc_value, traceback)
        self.decompressed_stream.__exit__(exc_type, exc_value, traceback)
        self.response.__exit__(exc_type, exc_value, traceback)
