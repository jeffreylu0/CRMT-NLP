import spacy
import sys
import pandas as pd
from pathlib import Path
from typing import Union
from spacy.tokens import DocBin

def convert_labeled_csv_to_spacy(csv_path: Union[Path, str],
                                 output_path: Union[Path, str] = './train.spacy') -> None:
    nlp = spacy.blank('en')
    labeled_df = pd.read_csv(csv_path)
    db = DocBin()

    training_data = [(portion,label) for portion,label in zip(labeled_df['Text Portion'], labeled_df['Class'])]
    categories = labeled_df['Class'].unique()

    for text,label in training_data:
        doc = nlp(text)
        doc.cats = {category:0 for category in categories}
        doc.cats[label] = 1
        db.add(doc)

    db.to_disk(output_path)

if __name__ == '__main__':
    convert_labeled_csv_to_spacy(sys.argv[1], sys.argv[2])

