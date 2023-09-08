from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from bidict import bidict


@dataclass
class TokenizationConfig:
    add_bos_token: bool = True
    add_eos_token: bool = False
    pad_to_output_length: bool = False
    output_length: int | None = None

    @classmethod
    def with_no_special_tokens(
        cls,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        **kwargs,
    ) -> TokenizationConfig:
        return TokenizationConfig(
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )


class LosslessTokenizer:
    def __init__(
        self,
        encoding="utf-8",
        errors="backslashreplace",
        bos_token_id=256,
        eos_token_id=256,
        pad_token_id=256,
        unk_token_id=0,
    ):
        self.encoding = encoding
        self.errors = errors

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        self.special_tokens = bidict(
            {
                self.eos_token_id: "<|EOS|>",
                self.bos_token_id: "<|BOS|>",
                self.pad_token_id: "<|PAD|>",
                self.unk_token_id: "<|UNK|>",
            }
        )

    def special_token(self, token_id):
        id_to_token = self.special_tokens
        if token_id not in id_to_token:
            id_to_token[token_id] = f"<|UNK-{token_id}|>"

        return id_to_token[token_id]

    def special_token_id(self, token):
        token_to_id = self.special_tokens.inv
        if token not in token_to_id:
            raise ValueError(f"{token!r} is not a special token")

        return token_to_id[token]

    def parse_tokens(
        self,
        token_ids: Iterable[int],
        collapse_paddings=True,
    ):
        padding_count = 0
        segment_bytes = []

        for token_id in token_ids:
            if token_id == self.pad_token_id:
                if collapse_paddings:
                    padding_count = 1
                else:
                    padding_count += 1

                continue

            if padding_count:
                if segment_bytes:
                    yield self.decode_bytes(segment_bytes)
                    segment_bytes = []

                pad_token = self.special_token(self.pad_token_id)
                yield from [pad_token] * padding_count

                padding_count = 0

            if token_id < 256 and token_id not in self.special_tokens:
                segment_bytes.append(token_id)
            else:
                if segment_bytes:
                    yield self.decode_bytes(segment_bytes)
                    segment_bytes = []

                # special tokens
                yield self.special_token(token_id)

        if segment_bytes:
            yield self.decode_bytes(segment_bytes)

    def encode_to_bytes(self, sample: str):
        return bytes(sample, encoding=self.encoding)

    def encode(self, sample: str, config=TokenizationConfig()):
        token_ids = []

        if config.add_bos_token:
            token_ids.append(self.bos_token_id)

        buffer = self.encode_to_bytes(sample)
        token_ids.extend(buffer)

        if config.add_eos_token:
            token_ids.append(self.eos_token_id)

        return token_ids[: config.output_length]

    def re_encode(self, sample: str):
        special_tokens = self.special_tokens.values()
        special_tokens_re = (re.escape(token_name) for token_name in special_tokens)
        delimiter_pattern = r"|".join(special_tokens_re)
        pattern = rf"({delimiter_pattern})"

        parts = re.split(pattern, sample)

        tokens = []
        for i, part in enumerate(parts):
            is_special_token = i % 2 == 0
            if is_special_token:
                tokens.append(self.special_token_id(part))
            else:
                config = TokenizationConfig.with_no_special_tokens()
                tokens.extend(self.encode(part, config))

        return tokens

    def decode_bytes(self, sample):
        return bytes(sample).decode(self.encoding, errors=self.errors)

    def decode(self, tokens: list[int], collapse_padding=True) -> str:
        parts = self.parse_tokens(tokens, collapse_paddings=collapse_padding)
        return "".join(parts)


def main():
    tokenizer = LosslessTokenizer()
    # for line in sys.stdin:
    #     print(f"Input:   {line!r}")

    #     encoded = tokenizer.encode(line)
    #     print(f"Encoded: {encoded}")

    #     # encoded = tokenizer.encode_raw(line)
    #     # print(f"Tokens:  {encoded}")

    #     decoded = tokenizer.decode(encoded)
    #     print(f"Decoded: {decoded!r}")

    batch = tokenizer.encode_batch(
        [
            "this is a test sample",
            '#     print(f"Decoded: {decoded!r}")',
        ],
        context_size=1024,
        max_batch_size=2,
    )

    print("batch.shape =", batch.shape)
    print("batch:", batch, sep="\n")
    for sample in tokenizer.decode_batch(batch):
        print(repr(sample))


if __name__ == "__main__":
    main()
