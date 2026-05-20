from gui import GUI
from adapters.tokenizer import Tokenizer
from adapters.embeddings import SkipGram
from corpus import Corpus

GUI(Corpus(Tokenizer()), SkipGram())