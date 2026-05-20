from adapters.tokenizer import Tokenizer
import numpy as np
import pandas as pd

class Corpus:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.tokens = np.ndarray((0, 0))
        self.one_hot = pd.DataFrame()
        self.pairs: list[tuple[int, int]] = list()
        
    
    def update_corpus(self, file: str, lang: str, sliding_window: int):
        self.tokens = self.tokenizer.get_tokens(file, language=lang)
        self.one_hot = self.tokenizer.one_hot_encoding(self.tokens)
        self.pairs = self.tokenizer.create_pairs(self.one_hot, self.tokens, slidingWindow=sliding_window)