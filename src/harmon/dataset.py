from __future__ import annotations

import io
import pathlib

import chess.pgn
import requests
import requests.adapters
import urllib3
import zstandard as zstd

DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


class Downloader(io.RawIOBase):
    def __init__(self, url: str, cache_dir="./dataset/"):
        self.url = url
        self.cache_dir = pathlib.Path(cache_dir)

        # get the download size
        response = requests.head(url)
        response.raise_for_status()
        self.download_size = int(response.headers.get("content-length", 0))
        self.downloaded = 0

        self.session = requests.Session()
        retries = urllib3.Retry(total=5)
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)

    def readable(self) -> bool:
        return True

    def read(self, __size: int = -1) -> bytes | None:
        while True:
            try:
                start_byte = self.downloaded
                end_byte = start_byte + __size if __size >= 0 else ""

                headers = {"Range": f"bytes={start_byte}-{end_byte}"}
                response = self.session.get(self.url, headers=headers)
                response.raise_for_status()

                data = response.content
                self.downloaded += len(data)

                return data
            except e:
                pass


class PgnDataset:
    def __init__(
        self,
        download_stream: io.RawIOBase,
        chunk_size=DOWNLOAD_CHUNK_SIZE,
    ):
        cctx = zstd.ZstdDecompressor()
        decompressd = cctx.stream_reader(download_stream, read_size=chunk_size)

        self.pgn_stream = io.TextIOWrapper(decompressd)

    def __iter__(self):
        return self

    def __next__(self) -> chess.pgn.Game:
        game = chess.pgn.read_game(self.pgn_stream)

        if game is None:
            raise StopIteration

        return game
