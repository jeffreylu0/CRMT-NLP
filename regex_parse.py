import fitz
import regex
import sys
from typing import List
from pathlib import Path
from textacy.preprocessing import pipeline, normalize, remove


class PortionExtractor:

    def __init__(self):

        # Classification markings
        self.markings = ['(U)', '(U//FOUO)', '(S)', '(CUI)', '(S//NF)', '(S//REL TO USA, FVEY)']

        # Pattern for classification markings to split text by
        self.pattern = regex.compile(r"""(\(U\))|
                                         (\(U\/\/FOUO\))|
                                         (\(S\))|
                                         (\(CUI\)|                                                         (\(S\/\/NF\))|
                                         (\(S\/\/REL\ TO\ USA,\ FVEY\))""",
                                         flags=regex.VERBOSE)

        # Textacy text preprocessing pipline
        self.preprocessor = pipeline.make_pipeline(normalize.unicode,
                                              normalize.whitespace,
                                              normalize.bullet_points,
                                              normalize.hyphenated_words,
                                              normalize.quotation_marks,
                                              remove.accents)
    def __call__(self, pdf_path: str):

        doc = fitz.open(pdf_path)
        portions = []
        for page_num,page in enumerate(doc):
            text = page.get_text('text') # get text from each page
            clean_matches = self.filter_matches(regex.split(self.pattern, text)) # filter regex matches
            portions.extend([(page_num,match) for match in self.preprocess_matches(clean_matches)]) #
        
        return portions, Path(doc.name).stem # return text portions, document name

    def filter_matches(self, matches: List[str]) -> List[str]:

        no_content = [None, '', ' '] + self.markings
        filtered = list(filter(lambda x: x not in no_content, matches)) # filter out matches that contain no content (None, empty strings, and classification markings)
        stripped = [match.strip() for match in filtered] # strip starting and ending whitespace  

        return stripped

    def preprocess_matches(self, matches: List[str]) -> List[str]:

        # Preprocess and remove newlines                                    
        return [self.preprocessor(match).replace('\n', '') for match in matches]
    


if __name__ == '__main__':

    extractor = PortionExtractor()
    portions = extractor(sys.argv[1])
    print(portions)
        
    
    
        