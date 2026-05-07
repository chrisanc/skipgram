import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

"""
Christian David Sánchez Sánchez
Universidad de Colima
Facultad de Ingeniería Mecánica y Eléctrica
Ingeniería en Computación Inteligente
6°B
"""

def get_content(filepath: str) -> str:
    with open(filepath) as file:
        data = file.read().upper()
    return data

def remove_punctuation(value: str) -> list[str]:
    return "".join(re.findall(r"[\w\s*]+", value))

def remove_duplicates(arr):
    result = list()

    for value in arr:
        if result.count(value) < 1:
            result.append(value)

    return result

def onehot_encode(tokens: list[str]) -> pd.DataFrame:
    zeros = np.zeros((len(tokens), len(tokens)))
    for i in range(len(tokens)):
        zeros[i][i] = 1
    df = pd.DataFrame(data=zeros, columns=tokens)

    return df

def create_pairs(slidingWindow: int, values: pd.DataFrame, tokens: list[str]) -> list[tuple[int, int]]:
    """
    Returns the pairs of indices of 'slidingWindow' size...
    """
    results: list[tuple[int, int]] = list()
    sumIds = 0
    for i in range(0, len(tokens)):
        subdf = values.iloc[:, i-sumIds:i + slidingWindow + 1]
        for j in range(len(subdf.columns)):
            if tokens[i] == subdf.columns[j]:
                continue
            results.append((i, i - sumIds + j))

        # Add one to the sumIds if necessary
        if sumIds < slidingWindow:
            sumIds+=1

    return results

if __name__ == "__main__":
    """
    Calculations section
    """
    # Load the stop words list
    stop_words = stopwords.words("english")
    # Read the raw content as a string
    content = get_content(filepath="book.txt")
    # Remove punctuation signs using regex
    content = remove_punctuation(content)

    # Create tokens with the NLTK library
    tokenized_content = word_tokenize(content)
    tokens = [value for value in tokenized_content if value not in stop_words]
    tokens = remove_duplicates(tokens)

    encoded_tokens = onehot_encode(tokens)
    pairs = create_pairs(2, encoded_tokens, tokens)

    """
    Pretty print section
    """
    # Print tokens
    print("Conjunto de tokens finales:")
    print(f"Total de tokens: {len(tokens)}")
    for i, token in enumerate(tokens, start=1):
        if i % 8 == 0:
            print(f"\n{token}", end=", ")
        else:
            print(token, end=", ")
    print("\n")

    # Print pairs
    print("Pares finales con window = 2:")
    print(f"Total de pares: {len(pairs)}")
    print("Primeros 200:")
    for pair in pairs[:200]:
        print(pair, end=" ")