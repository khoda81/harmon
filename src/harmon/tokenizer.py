from __future__ import annotations

import collections
import re
from typing import Iterable

import attrs
from bidict import bidict


@attrs.define
class EncodeConfig:
    """config for encoding into token ids

    >>> tokenizer = LosslessTokenizer()
    >>> config = EncodeConfig.with_no_special_tokens()
    >>> tokenizer.decode(tokenizer.encode("Hello world", config))
    'Hello world'
    >>> config = EncodeConfig(
    ...     add_eos_token=True,
    ...     output_length=16,
    ...     pad_to_output_length=True,
    ... )
    >>> token_ids = tokenizer.encode("Hello world", config)
    >>> len(token_ids)
    16
    >>> tokenizer.decode(token_ids, DecodeConfig(yield_ending_pad=True))
    '<|PAD|>Hello world<|PAD|><|PAD|><|PAD|><|PAD|>'
    >>> token_ids = tokenizer.encode("This is a long input", config)
    >>> len(token_ids)
    16
    >>> tokenizer.decode(token_ids, DecodeConfig(yield_ending_pad=True))
    '<|PAD|>This is a long '
    """

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
    ) -> EncodeConfig:
        return EncodeConfig(
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )


@attrs.define
class DecodeConfig:
    collapse_paddings: bool = False
    ending_pad: bool = False
    special_tokens: bool = False


@attrs.define
class Segment:
    content: str
    origin: list[int]
    is_special_token: bool = False


class SegmentedParser:
    def __init__(
        self,
        tokenizer: LosslessTokenizer,
        config: DecodeConfig = DecodeConfig(),
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config

        # parser context
        self.padding_counter = 0
        self.text_bytes: list[int] = []
        self.output_buffer: collections.deque[Segment] = collections.deque()

    def fill(self, segments: Iterable[Segment]):
        self.output_buffer.extend(segments)

    def drain(self):
        while self.output_buffer:
            yield self.output_buffer.popleft()

    def buffered(self, tokens: Iterable[Segment]):
        self.fill(tokens)
        yield from self.drain()

    def special_token(self, token_id: int):
        special_token = self.tokenizer.decode_special_token(token_id)
        return Segment(special_token, origin=[token_id], is_special_token=True)

    def _decode_text(self):
        token_ids = self.text_bytes.copy()
        self.text_bytes.clear()
        return Segment(self.tokenizer.decode_bytes(token_ids), token_ids)

    def drain_paddings(self):
        while self.padding_counter > 0:
            self.padding_counter -= 1
            yield self.special_token(self.tokenizer.pad_token_id)

    def drain_text(self):
        if self.text_bytes:
            yield from self.drain_paddings()
            yield self._decode_text()

    def _handle_special_token(self, token_id):
        """handle a single special token

        Args:
            token_id (int): the special token id

        Yields:
            Segment

        Caution:
            This method should be drained all the way to the end or it might leave
            the parser in an invalid state.
        """

        # yield the current text segment
        yield from self.drain_text()

        if token_id == self.tokenizer.pad_token_id:
            if self.config.collapse_paddings:
                self.padding_counter = 1
            else:
                self.padding_counter = self.padding_counter + 1

        else:
            yield from self.drain_paddings()
            yield self.special_token(token_id)

    def handle_special_token(self, token_id: int):
        # this method uses buffered yield because the generator might be stopped
        # in the middle of handling a special token which leaves the parser at a
        # broken state

        # we make sure to finish handling this token before starting to yield
        yield from self.buffered(self._handle_special_token(token_id))

    def decode(self, token_ids: Iterable[int]):
        """Decode token_ids into segments

        Args:
            token_ids (Iterable[int]): token ids to decode

        Yields:
            Segment

        >>> tokenizer = LosslessTokenizer(
        ...     bos_token_id=257,
        ...     pad_token_id=256,
        ...     unk_token_id=0,
        ... )
        >>> parser = SegmentedParser(tokenizer=tokenizer)
        >>> token_ids = tokenizer.encode("This is a test input")
        >>> list(segment.content for segment in parser.decode(token_ids))
        ['<|BOS|>', 'This is a test input']

        >>> token_ids = tokenizer.re_encode("<|BOS|>Hello<|UNK|>World<|PAD|><|PAD|>")
        >>> list(segment.content for segment in parser.decode(token_ids))
        ['<|BOS|>', 'Hello', '<|UNK|>', 'World']

        >>> parser.config.yield_ending_pad = True
        >>> token_ids = tokenizer.re_encode("<|BOS|>This is the second input<|PAD|><|PAD|><|PAD|><|PAD|>")
        >>> list(segment.content for segment in parser.decode(token_ids))
        ['<|PAD|>', '<|PAD|>', '<|BOS|>', 'This is the second input', '<|PAD|>', '<|PAD|>', '<|PAD|>', '<|PAD|>']

        >>> parser.config.collapse_paddings = True
        >>> token_ids = tokenizer.re_encode("<|BOS|>This is the second input<|PAD|><|PAD|><|PAD|><|PAD|>")
        >>> list(segment.content for segment in parser.decode(token_ids))
        ['<|BOS|>', 'This is the second input', '<|PAD|>']
        """

        # empty everything remaining in the buffer before taking the first token
        yield from self.drain()

        for token_id in token_ids:
            if token_id < 256 and token_id not in self.tokenizer.special_tokens:
                # this token is a normal byte
                self.text_bytes.append(token_id)

            elif self.config.special_tokens:
                yield from self.handle_special_token(token_id)

        yield from self.drain_text()

        if self.config.ending_pad:
            yield from self.drain_paddings()


class LosslessTokenizer:
    """Tokenizer any text to bytes in a lossless way.

    >>> tokenizer = LosslessTokenizer()
    >>> token_ids = tokenizer.encode("Hello world")
    >>> token_ids
    [256, 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
    >>> tokenizer.decode(token_ids)
    '<|PAD|>Hello world'

    >>> token_ids = tokenizer.encode("سلام به دنیا")
    >>> tokenizer.decode(token_ids)
    '<|PAD|>سلام به دنیا'
    """

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

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self.special_tokens = bidict(
            {
                self.bos_token_id: "<|BOS|>",
                self.eos_token_id: "<|EOS|>",
                self.pad_token_id: "<|PAD|>",
                self.unk_token_id: "<|UNK|>",
            }
        )

    def encode_to_bytes(self, sample: str):
        return bytes(sample, encoding=self.encoding)

    def decode_bytes(self, sample):
        return bytes(sample).decode(self.encoding, errors=self.errors)

    def encode_special_token(self, token: str) -> int:
        """Get id of an special token.

        Args:
            token (str): token

        Raises:
            ValueError: if the token is not in special_tokens

        Returns:
            int: id of the token

        >>> tokenizer = LosslessTokenizer(unk_token_id=0)
        >>> tokenizer.encode_special_token("<|UNK|>")
        0
        >>> tokenizer.encode_special_token("UNK")
        Traceback (most recent call last):
            ...
        ValueError: 'UNK' is not a special token
        """

        token_to_id = self.special_tokens.inv
        if token not in token_to_id:
            raise ValueError(f"{token!r} is not a special token")

        return token_to_id[token]

    def decode_special_token(self, token_id: int) -> str:
        """Get special token corresponding to token_id.

        Tries to find the token in special tokens, if its not found, adds a new
        special token with name='<|UNK-{token_id}|>'.

        Args:
            token_id (int): id of special token

        Returns:
            str: special token

        >>> tokenizer = tokenizer = LosslessTokenizer(
        ...     unk_token_id=0,
        ...     pad_token_id=256,
        ... )
        >>> tokenizer.decode_special_token(0)
        '<|UNK|>'
        >>> tokenizer.decode_special_token(256)
        '<|PAD|>'
        >>> tokenizer.re_encode("<|UNK-2001|>")
        [60, 124, 85, 78, 75, 45, 50, 48, 48, 49, 124, 62]
        >>> tokenizer.decode_special_token(2001)
        '<|UNK-2001|>'
        >>> tokenizer.re_encode("<|UNK-2001|>")
        [2001]

        """

        id_to_token = self.special_tokens
        if token_id not in id_to_token:
            id_to_token[token_id] = f"<|UNK-{token_id}|>"

        return id_to_token[token_id]

    def encode(self, sample: str, config=EncodeConfig()):
        token_ids = []

        if config.add_bos_token:
            token_ids.append(self.bos_token_id)

        token_ids.extend(self.encode_to_bytes(sample))

        if config.add_eos_token:
            token_ids.append(self.eos_token_id)

        if config.output_length is not None:
            if len(token_ids) >= config.output_length:
                token_ids = token_ids[: config.output_length]

            elif config.pad_to_output_length:
                pad_count = config.output_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_count)

        return token_ids

    def re_encode(self, sample: str):
        special_tokens = self.special_tokens.values()
        special_tokens_regex = (re.escape(token_name) for token_name in special_tokens)
        delimiter_pattern = r"|".join(special_tokens_regex)
        pattern = rf"({delimiter_pattern})"

        segments = re.compile(pattern).split(sample)

        tokens = []
        for i, segment in enumerate(segments):
            # Even segments are normal text bytes
            if i % 2 == 0:
                config = EncodeConfig.with_no_special_tokens()
                tokens.extend(self.encode(segment, config))

            # Odd segments are special tokens
            else:
                tokens.append(self.encode_special_token(segment))

        return tokens

    def decode(self, tokens: list[int], config=DecodeConfig()) -> str:
        parser = SegmentedParser(tokenizer=self, config=config)
        return "".join(segment.content for segment in parser.decode(tokens))


def main():  # pragma: no cover
    import sys

    tokenizer = LosslessTokenizer(
        bos_token_id=257,
        pad_token_id=256,
        unk_token_id=0,
    )

    for line in sys.stdin:
        print(f"Input:   {line!r}")
        # token_ids = tokenizer.encode(line)
        token_ids = tokenizer.re_encode(line)

        # token_ids = map(int, line.strip().split())

        print(f"Tokens:  {token_ids}")

        # decoded = tokenizer.decode(token_ids)
        # print(f"Decoded: {decoded!r}")

        parser = SegmentedParser(tokenizer)
        segments = parser.decode(token_ids)
        for segment in segments:
            if segment.is_special_token:
                print("Special:", segment.content)
            else:
                print("Text:   ", repr(segment.content))


if __name__ == "__main__":
    main()
