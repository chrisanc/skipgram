from gui import GUI
from adapters.tokenizer import Tokenizer
from adapters.embeddings import SkipGram

GUI(Tokenizer(), SkipGram())