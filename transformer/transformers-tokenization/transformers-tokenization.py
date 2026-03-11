import re
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, lowercased, punctuation stripped."""
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Fixed IDs: PAD=0, UNK=1, BOS=2, EOS=3
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        # Unique words from training texts
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.word_to_id:
                    idx = len(self.word_to_id)
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in self._tokenize(text)]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Skip special tokens.
        """
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        words = [
            self.id_to_word[i]
            for i in ids
            if i in self.id_to_word and self.id_to_word[i] not in special
        ]
        return " ".join(words)