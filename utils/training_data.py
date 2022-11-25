import spacy
import sys
import pandas as pd
from pathlib import Path
from typing import Union
from spacy.tokens import DocBin

# Labels under 'Class' field
def convert_textcat_csv_to_spacy(csv_path: Union[Path, str],
                                 output_path: Union[Path, str] = './textcat_train.spacy') -> None:
    nlp = spacy.blank('en')
    df = pd.read_csv(csv_path)
    db = DocBin()

    training_data = [(portion,label) for portion,label in zip(df['Text Portion'], df['Class'])]
    categories = df['Class'].unique()

    for text,label in training_data:
        doc = nlp(text)
        doc.cats = {category:0 for category in categories}
        doc.cats[label] = 1
        db.add(doc)

    db.to_disk(output_path)

# Entity annotation under 'Entity', 'Start', and 'End' fields
def convert_ner_csv_to_spacy(csv_path: Union[Path, str],
                             output_path: Union[Path, str] = './ner_train.spacy') -> None:
    nlp = spacy.blank('en')
    df = pd.read_csv(csv_path)
    db = DocBin()

    training_data = [(portion,[(entity, start, end)]) 
                     for portion, entity, start, end 
                     in zip(df['Text Portion'], df['Entity'], df['Start'], df['End'])]
    
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for entity, start, end in annotations:
            span = doc.char_span(start, end, label=entity)
            ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)
    
if __name__ == '__main__':
    convert_textcat_csv_to_spacy(sys.argv[1], sys.argv[2])

