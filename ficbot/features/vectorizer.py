from typing import List, Union

import numpy as np


class Mapper(object):

    def __init__(self, *, char_level=False):
        self.char_level = char_level
        self._token_n = None
        self._n_token = None

    def _tokenize_text(self, text: str) -> Union[str, List[str]]:
        """Bring text to lowercase and split it into word iterable if needed."""
        text = text.lower()
        if not self.char_level:
            text = text.split()
        return text

    def create_text_map(self, text: str):
        """Build {int: token} and {token: int} maps from a single text"""
        text = self._tokenize_text(text)
        tokens = sorted(list(set(text)))

        self._token_n = {token: n for n, token in enumerate(tokens)}
        self._n_token = {n: token for n, token in enumerate(tokens)}
        return text, self._token_n, self._n_token

    def create_corpus_map(self, corpus: List[str]):
        """Build {int: token} and {token: int} maps from a collection of texts"""
        corpus = ' '.join(corpus)
        return self.create_text_map(corpus)

    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self._token_n)


class SequenceVectorizer(Mapper):

    # * requires all the following arguments to be explicitly named on function call

    def __init__(self, corpus: List[str], *, char_level: bool = True):
        super().__init__(char_level=char_level)
        self.create_corpus_map(corpus)

    def sequenize_text(self, text: Union[str, List[str]], *, maxlen: int, step: int = 1):
        """Basic text preprocessing function.

        Converts text into subsequences of maximum length MAXLEN.
        Args:
            text (str | list): A text (iterable) to be sequenized.
            maxlen (int): Maximum subsequence length
            step (int): Sequenization step (we take a sequence every STEP tokens).
        Returns:
            sequences (list[str] | list[list[str]]): List of subsequences of length maxlen.
            next_chars (list[str] | list[list[str]]): List of tokens following after each corresponding sequence.

        """
        text = self._tokenize_text(text)
        sequences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sequences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        return sequences, next_chars

    def vectorize_text(self, text, *, maxlen: int, step: int = 1):

        text_sequences, text_next_chars = self.sequenize_text(text, maxlen=maxlen, step=step)

        vector_seq = np.zeros((len(text_sequences), maxlen, self.get_vocab_size()), dtype=np.bool)
        vector_next = np.zeros((len(text_sequences), self.get_vocab_size()), dtype=np.bool)

        for i, sequence in enumerate(text_sequences):
            for j, token in enumerate(sequence):
                vector_seq[i, j, self._token_n[token]] = 1
            vector_next[i, self._token_n[text_next_chars[i]]] = 1

        return vector_seq, vector_next
