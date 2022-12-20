import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from typing import Union
from pathlib import Path
from transformers import RobertaTokenizerFast

# Wrapper for RoBERTa tokenizers for use in Spacy pipeline
# https://spacy.io/usage/linguistic-features/#custom-tokenizer-example2
class RobertaTokenizerSpacy:

    def __init__(self, model_path: Union[str, Path]):

        self._tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.vocab = Vocab(strings=list(self._tokenizer.get_vocab().keys())) # setup Vocab object for Doc
    
    def __call__(self, text: str) -> Doc:

        encoding = self._tokenizer(text)
        tokens = self._tokenizer.convert_ids_to_tokens(encoding.input_ids)
        words = []
        spaces = []

        for i, token in enumerate(tokens):
            words.append(token)
            if i < len(tokens)-1:
                # Roberta tokenizer appends 'Ġ' to the subsequent word if there's a space
                spaces.append(tokens[i+1][0]=='Ġ') 
            else:
                spaces.append(False)

        return Doc(self.vocab, words=words, spaces=spaces)


# Add RoBERTa tokenizer to use in training config
@spacy.registry.tokenizers("roberta-tokenizer")
def create_roberta_tokenzer():
    def roberta_tokenizer(nlp):


# Add custom MLFlow logger for Databricks use
# https://spacy.io/usage/training#custom-logging
# https://docs.databricks.com/mlflow/tracking.html
@spacy.registry.loggers("databricks-MlFLow-logger")
def db_mlflow_logger():
    def setup_logger():
        pass
    pass

