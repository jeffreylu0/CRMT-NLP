import spacy
import typer
import pandas as pd
from pathlib import Path
from typing import List
from spacy.tokens import DocBin

nlp = spacy.blank('en')

# fields to check for in text classifcation training data
TEXTCAT_FIELDS = ['Document Name', 'Page Number', 'Text Portion', 'Class'] 
# fields to check for in named entity recognition training data
NER_FIELDS = ['Document Name', 'Page Number', 'Text Portion', 'Entity', 'Start', 'End', 'Class'] 

def convert_textcat_csv_to_spacy(input_path: str,
                                 output_path: str = './textcat_train.spacy') -> None:

    df = pd.read_csv(input_path, index_col=False)

    # Check if fields are correct
    input_columns = df.columns.values
    assert (len(input_columns) == len(TEXTCAT_FIELDS)) and (all(input_columns == TEXTCAT_FIELDS)), f"Fields should be {TEXTCAT_FIELDS}"

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
def convert_ner_csv_to_spacy(input_path: str,
                             output_path: str = './ner_train.spacy') -> None:

    df = pd.read_csv(input_path, index_col=False)

    # Check if fields are correct
    input_columns = df.columns.values
    assert (len(input_columns) == len(NER_FIELDS)) and (all(input_columns == NER_FIELDS)), f"Fields should be {NER_FIELDS}"
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

def main(ner: bool = typer.Option(False, help='Whether input labeled data is for NER. Default is text classification'),
         input_path: str = typer.Argument(..., help='Input path to labeled data in CSV format'),
         output_path: str = typer.Argument('./output.spacy', help='Output path for binary .spacy file')):

    assert Path(input_path).suffix == '.csv', "Please use CSV file as input"
    assert Path(output_path).suffix == '.spacy', "Please specify .spacy file as output"

    if ner:
        convert_ner_csv_to_spacy(input_path, output_path)
    else:
        convert_textcat_csv_to_spacy(input_path, output_path)

if __name__ == '__main__':
    typer.run(main)

