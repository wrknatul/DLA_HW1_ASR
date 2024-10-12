import re
from string import ascii_lowercase

from pyctcdecode import Alphabet, BeamSearchDecoderCTC
from typing import List
import torch
import math
import numpy as np
from collections import defaultdict

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier

def _log_softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    with np.errstate(divide="ignore"):
        out = tmp - np.log(np.sum(np.exp(tmp), axis=axis, keepdims=True))
        return out

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)
        self.default_beam_size = 100

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result_string = ''
        prev = self.EMPTY_TOK
        for ind in inds:
            char = self.ind2char[ind]
            if char == self.EMPTY_TOK:
                prev = char
                continue

            if (prev == self.EMPTY_TOK or char != result_string[-1]):
                result_string += char
                prev = char

        return result_string

    def ctc_beam_search(self, inds: torch.Tensor) -> str:
        return self.decoder.decode(inds.numpy(), beam_width=self.default_beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
