import re
import spacy

from loguru import logger
from typing import List
from textacy import preprocessing
from nltk.corpus import stopwords
import nltk

class GetTopics:
    """
    Class to get topics from each transcript
    """
    def __init__(
        self, 
        transcript_list: List[str]
    ):
        self.transcript_list = transcript_list
        
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        self.raw = {}
        self.ts = {}
        
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
        
        stop_words = set(stopwords.words("english"))
        self.filler_words = stop_words.union({"um", "uh", "like", "sort of", "kinda"})

    def init_nltk(self) -> None:
        """
        Function to initialise nltk
        """
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            logger.info('nltk stop words not found, downloading')
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
        
        self.filler_words = stop_words.union({"um", "uh", "like", "sort of", "kinda"})
        
    def load_transcripts(self) -> None:
        """
        Function to load transcripts
        """
        for i, transcript in enumerate(self.transcript_list):
            logger.info(f'Loading transcript: {transcript}')
            with open(transcript, "r", encoding="utf-8") as file:
                self.raw[i] = file.readlines()
            
        logger.info(f'total transcript count: {len(self.raw.keys())}')


    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts
        """
        logger.info('Cleaning transcripts')
        
        for i, ts in self.raw.items():
            self.ts[i] = []
    
            for line in ts:
                speaker = None
                if line.startswith("Therapist:"):
                    speaker = "Therapist"
                elif line.startswith("Client:"):
                    speaker = "Client"
                
                line = re.sub(r"^(Therapist|Client):", "", line).strip()
                line = re.sub(r"\b(u+h+|u+m+|uh|um)\b", "", line, flags=re.IGNORECASE)
                
                if line == '':
                    continue
                
                doc = self.nlp(line.lower())
                cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

                cleaned_line = " ".join(cleaned_tokens)

                self.ts[i].append(f"{speaker if speaker else 'obfuscated'}: {cleaned_line}")

    def filter_filler_words(
        self, 
        text: str
    ) -> str:
        """
        Function to filter out filler words from text
        :param text: text to filter out filler words from
        """
        doc = self.nlp(text)
        return " ".join([token.text for token in doc if token.text.lower() not in self.filler_words and not token.is_stop])

    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts
        """
        logger.info('Cleaning transcripts')
        
        for i, ts in self.raw.items():
            self.ts[i] = []

            for line in ts:
                speaker = None
                if line.startswith("Therapist:"):
                    speaker = "Therapist"
                elif line.startswith("Client:"):
                    speaker = "Client"
                
                line = re.sub(r"^(Therapist|Client):", "", line).strip()
                line = self.filter_filler_words(line)
                                
                doc = self.nlp(line.lower())
                cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

                cleaned_line = " ".join(cleaned_tokens)
                
                if cleaned_line == '':
                    continue

                self.ts[i].append(f"{speaker if speaker else 'obfuscated'}: {cleaned_line}")

    def runner(self) -> None:
        """
        Main runner function
        """
        self.init_nltk()
        
        self.load_transcripts()
        self.clean_transcripts()
        
        x = 0
    

if __name__ == "__main__":
    a = GetTopics(transcript_list=['/Users/ansonliu/Documents/Github/Anson_ai/data/transcripts/synthetic_test/depression_synthetic.txt'])
    
    a.runner()