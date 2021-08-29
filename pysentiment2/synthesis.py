import pandas as pd
from pysentiment2.base import STATIC_PATH, BaseDict


class SYNTHESIS(BaseDict):
    """
    Dictionary class for
    Loughran and McDonald Financial Sentiment Dictionaries.
    
    See also https://www3.nd.edu/~mcdonald/Word_Lists.html
    
    The terms for the dictionary are stemmed by the default tokenizer.
    """
    
    PATH = '%s/SYNTHESIS.csv' % STATIC_PATH
    
    def init_dict(self):
        data = pd.read_csv(self.PATH)
        self._posset = set(data.query('pos > 0')['word'].apply(self.tokenize_first).dropna())
        self._negset = set(data.query('neg > 0')['word'].apply(self.tokenize_first).dropna())
