import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Tokenizer:
    def __init__(self):
        self.file = self.__read_file_content(
            root = 
            filedialog.askopenfilename(filetypes=[("Text files, *.txt")])
        )

    def __read_file_content(self) -> str:
        with open(self.file, "r") as f:
            content = f.read()
        
        return self.__remove_punctuation(content)
    
    def __remove_punctuation(self, value: str) -> list[str]:
        return "".join(re.findall(r"[\w\s*]+", value))
    
    def get_tokens(self) -> np.ndarray:
        stop_words = stopwords.words("english")
        tokens = word_tokenize(self.__read_file_content())
        tokens = [value for value in tokens if value not in stop_words]
        tokens = pd.unique(tokens)

        return tokens
    
    def one_hot_encoding(self, tokens: np.ndarray) -> pd.DataFrame:
        if tokens == None:
            tokens = self.get_tokens()

        zeros = np.zeros((len(tokens), len(tokens)))
        for i in range(len(tokens)):
            zeros[i][i] = 1
        df = pd.DataFrame(data=zeros, columns=tokens)

        return df
    
    def create_pairs(self, one_hot: pd.DataFrame, tokens: np.ndarray, slidingWindow: int = 2) -> list[tuple[int, int]]:
        results: list[tuple[int, int]] = list()
        sumIds = 0
        for i in range(0, len(tokens)):
            subdf = one_hot.iloc[:, i-sumIds:i + slidingWindow + 1]
            for j in range(len(subdf.columns)):
                if tokens[i] == subdf.columns[j]:
                    continue
                results.append((i, i - sumIds + j))

            # Add one to the sumIds if necessary
            if sumIds < slidingWindow:
                sumIds+=1

        return results
    

    def tf_idf(self, tokens: np.ndarray):
        """
        Calculates the TF-IDF for each one of the tokens
        based on the corpus
        """
        if (tokens == None):
            tokens = self.get_tokens()