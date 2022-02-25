from typing import List, Union

import numpy as np
import os
import pickle


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

    def create_text_map(self, text: str, ood_token: str = "?"):
        """Build {int: token} and {token: int} maps from a single text"""
        text = self._tokenize_text(text)
        text_set = set(text)
        assert ood_token not in text_set, "Out of Dictionary token is not unique! Choose a different one."
        tokens = sorted(list(text_set))

        self._token_n = {token: n for n, token in enumerate(tokens)}
        self._n_token = {n: token for n, token in enumerate(tokens)}
        self._token_n[ood_token] = len(self._token_n)
        self._n_token[len(self._n_token)] = ood_token
        return text, self._token_n, self._n_token

    def create_corpus_map(self, corpus, *, ood_token: str = "?"):
        """Build {int: token} and {token: int} maps from a collection of texts"""
        if not isinstance(corpus, list):
            corpus = list(corpus)
        corpus = ' '.join(corpus)
        return self.create_text_map(corpus, ood_token=ood_token)

    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self._token_n)

    def maps(self):
        """Return token_n and n_token maps"""
        return self._token_n, self._n_token

    def save_maps(self, save_folder):
        map_path = os.path.join(save_folder, "maps.pkl")
        maps = [self._token_n, self._n_token]
        with open(map_path, "wb") as mp:
            pickle.dump(maps, mp)

    def load_maps(self, token_n, n_token):
        """ Initializes mapper from n_token and token_n maps """
        assert token_n == {value: key for key, value in n_token.items()}, "Indicing maps are not inverses of each other"
        self._token_n = token_n
        self._n_token = n_token


class SequenceVectorizer(Mapper):

    # * requires all the following arguments to be explicitly named on function call

    def __init__(self, *, corpus=None, char_level: bool = True, maps=None, ood_token: str = "?"):
        """

        :param corpus: An iterable of texts
        :param char_level: Whether to do char-level or word-level tokenization
        :param maps: Tuple of (token_n, n_token) maps from token to int and int to token respectively.
                              Maps should be inverses of each other.
        :param ood_token: How to mark out of dictionary words (characters)
        """
        super().__init__(char_level=char_level)
        assert any([corpus, maps]), "Neither corpus, nor maps have been provided."
        self.ood_token = ood_token
        if maps is None:
            self.create_corpus_map(corpus, ood_token=ood_token)
        else:
            token_n, n_token = maps
            self.load_maps(token_n, n_token)

    def sequenize(self, text: Union[str, List[str]], *, maxlen: int, step: int = 1):
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
        for i in range(len(text)):
            if text[i] not in self._token_n:
                text[i] = self.ood_token
        sequences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sequences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        return sequences, next_chars

    def vectorize(self, text, *, maxlen: int, step: int = 1):

        text_sequences, text_next_chars = self.sequenize(text, maxlen=maxlen, step=step)

        vector_seq = np.zeros((len(text_sequences), maxlen, self.get_vocab_size()), dtype=np.bool)
        vector_next = np.zeros((len(text_sequences), self.get_vocab_size()), dtype=np.bool)

        for i, sequence in enumerate(text_sequences):
            for j, token in enumerate(sequence):
                vector_seq[i, j, self._token_n[token]] = 1
            vector_next[i, self._token_n[text_next_chars[i]]] = 1

        return vector_seq, vector_next
