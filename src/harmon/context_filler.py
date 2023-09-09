from __future__ import annotations

import collections
from typing import Iterable


class ContextFiller:
    """Try to fit as many samples as possible into the context of a transformer.


    Attributes:
        context_size: The context size of the transformer.
        fill: An integer which is used to fill up the empty spaces in context.
        buffer: A buffer for samples to be batched.

    Insert samples into this buffer as they come in but keep this buffer small
    because larger buffers cause slower batches but may fill the context more
    efficiently.

    This class acts as a simple buffer if context_size is None.

    >>> context_filler = ContextFiller(fill=0)
    >>> context_filler.buffer.extend([
    ...     list(range(1, 10)),
    ...     list(range(1, 5)),
    ...     list(range(1, 7)),
    ...     list(range(1, 11)),
    ...     list(range(1, 9)),
    ... ])
    >>> context_filler.get_batch(2)
    [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 0, 0, 0, 0]]
    >>> context_filler.context_size = 16
    >>> context_filler.get_batch(2)
    [[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]]
    """

    def __init__(
        self,
        fill: int,
        sequences: Iterable[list[int]] = [],
        context_size: int | None = None,
    ) -> None:
        self.context_size = context_size
        self.fill = fill
        self.buffer: collections.deque = collections.deque(sequences)

    def get_batch(self, batch_size: int, pad=True):
        lines: list[list[int]] = []

        while self.buffer and len(lines) < batch_size:
            sample = self.buffer.popleft()
            sample = sample[: self.context_size]

            if self.context_size is None:
                lines.append(sample)

            else:
                # try to fit into one of the lines
                for line in lines:
                    if len(line) + len(sample) <= self.context_size:
                        line.extend(sample)
                        break

                else:
                    # if it did'nt fit in any line
                    lines.append(sample)

        if self.context_size is not None:
            self.buffer.append(None)
            while self.buffer[0] is not None:
                sample = self.buffer.popleft()
                sample = sample[: self.context_size]

                for line in reversed(lines):
                    if len(line) + len(sample) <= self.context_size:
                        line.extend(sample)
                        break

                self.buffer.append(sample)

            self.buffer.popleft()

        if pad:
            lines = self.pad_samples(lines)

        return lines

    def pad_samples(self, lines: list[list[int]], target_length: int | None = None):
        """Pad all lines to the same length"""
        if target_length is None:
            target_length = max(len(line) for line in lines)

        for line in lines:
            pad_size = target_length - len(line)
            line.extend([self.fill] * pad_size)

        return lines
