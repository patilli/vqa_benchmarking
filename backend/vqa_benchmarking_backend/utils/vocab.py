import json
from typing import Dict, Union


class Vocabulary:
    def __init__(self, itos: Dict[int, str] = {}, stoi: Dict[str, int] = {}) -> None:
        self._itos = itos
        self._stoi = stoi

    def add_token(self, token: str):
        """
        Add a new token to the vocabulary.
        Will only be added, if it is not already inside the vocabulary.
        """
        if not token in self._stoi:
            token_id = len(self._stoi)
            self._stoi[token] = token_id
            self._itos[token_id] = token

    def save(self, path: str = '.data/vocab.json'):
        """
        Save current state of voocabulary to file.
        """
        with open(path, 'w') as f:
            json.dump({
                'itos': self._itos,
                'stoi': self._stoi
            }, f)

    @classmethod
    def load(cls, path: str = '.data/vocab.json'):
        """
        Load existing vocabulary from file.
        Returns:
            An instance of the loaded vocabulary.
        """
        with open(path, 'r') as f:
            data = json.load(f)
            return Vocabulary(itos={int(token_id): data['itos'][token_id] for token_id in data['itos']},
                              stoi={token: int(data['stoi'][token]) for token in data['stoi']})

    def stoi(self, word: str) -> Union[int, None]:
        """
        String to index
        """
        if word in self._stoi:
            return self._stoi[word]
        return None

    def itos(self, index: str) -> Union[str, None]:
        """
        Index to string
        """
        if index in self._itos:
            return self._itos[index]
        return None

    def exists(self, word: str) -> bool:
        """
        Returns:
            True, if word exists in this vocabulary, else False
        """
        return word in self._stoi
        
    def __len__(self) -> int:
        """
        Returns:
            Size of vocabulary (how many words in vocabulary)
        """
        return len(self._itos)
