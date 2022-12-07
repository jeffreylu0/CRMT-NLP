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

# Entity annotation under 'Start', 'End', 'Label' fields
def convert_ner_csv_to_spacy(csv_path: Union[Path, str],
                             output_path: Union[Path, str] = './ner_train.spacy') -> None:
    nlp = spacy.blank('en')
    df = pd.read_csv(csv_path)
    db = DocBin()

    training_data = [(portion,[(start, end, label)]) 
                        for portion, start, end, label
                        in zip(df['Text Portion'], df['Start'], df['End'], df['Class'])]

    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:

            entity = doc.text[start:end]

            # Check if entity span has leading or trailing whitespaces. Truncate whitespaces to align start and end idx with document tokens
            if entity[0] == ' ':
                start += 1
            if entity[-1] == ' ':
                end -= 1

            # Create Span with label
            span = doc.char_span(start, end, label=label)
            ents.append(span)

        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)
    
if __name__ == '__main__':
    convert_textcat_csv_to_spacy(sys.argv[1], sys.argv[2])

