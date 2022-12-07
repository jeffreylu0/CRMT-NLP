import fitz
import regex
import sys
import pandas as pd
from typing import List, Dict, Union
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
                                         (\(CUI\))|                                                         
                                         (\(S\/\/NF\))|
                                         (\(S\/\/REL\ TO\ USA,\ FVEY\))""",
                                         flags=regex.VERBOSE)

        # Textacy preprocessing pipeline
        self.preprocessor = pipeline.make_pipeline(normalize.unicode,
                                               normalize.whitespace,
                                               normalize.bullet_points,
                                               normalize.hyphenated_words,
                                               normalize.quotation_marks,
                                               remove.accents)                                         

    def __call__(self, pdf_paths: List[Union[Path, str]]) -> Dict[str, List]:
        
        portions = {'Document Name': [],
                    'Page Number': [], 
                    'Text Portion': []}

        for pdf in pdf_paths:
            doc = fitz.open(pdf)
            doc_name = Path(doc.name).stem

            for page_num,page in enumerate(doc):
                text = page.get_text('text') # get text from each page
                filtered_matches = self.filter_matches(regex.split(self.pattern, text)) # filter regex matches
                clean_matches = self.preprocess_matches(filtered_matches)

                portions['Document Name'].extend([doc_name] * len(clean_matches))
                portions['Page Number'].extend([page_num] * len(clean_matches))
                portions['Text Portion'].extend(clean_matches)
        
        return portions

    def filter_matches(self, matches: List[str]) -> List[str]:

        no_content = [None, '', ' '] + self.markings
        filtered = list(filter(lambda x: x not in no_content, matches)) # filter out matches that contain no content (None, empty strings, and classification markings)
        stripped = [match.strip() for match in filtered] # strip starting and ending whitespace  

        return stripped

    def preprocess_matches(self, matches: List[str]) -> List[str]:

        # Preprocess and remove newlines                                    
        return [self.preprocessor(match).replace('\n', '') for match in matches]

def main(input_path: Union[Path, str], 
         output_path: Union[Path, str] = Path(__file__).parent) -> None:
    
    extractor = PortionExtractor()
    portions_df = pd.DataFrame(extractor([input_path]))
    portions_df.to_csv(output_path)

if __name__ == '__main__':

    main(sys.argv[1])
        
    
    
        