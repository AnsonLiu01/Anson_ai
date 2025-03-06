import re
import spacy

from loguru import logger
from typing import List

class GetTopics:
    """
    Class to get topics from each transcript
    """
    def __init__(
        self, 
        transcript_list: List[str]
    ):
        self.transcript_list = transcript_list
        
        self.nlp = spacy.load("en_core_web_sm")

        self.raw = {}
        self.ts = {}
        
    def load_transcripts(self) -> None:
        """
        Function to load transcripts
        """
        for i, transcript in enumerate(self.transcript_list):
            logger.info(f'Loading transcript: {transcript}')
            with open(transcript, "r", encoding="utf-8") as file:
                self.raw[i] = file.readlines()
            
        logger.info(f'total transcript count: {len(self.transcript.keys())}')


    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts
        """
        logger.info('Cleaning transcripts')
        
        for i, ts in self.raw.items():
            self.ts[i] = []
    
            for line in ts:
                line = re.sub(r"^(Therapist|Client):", "", line).strip()

                line = re.sub(r"\b(um|uh|like|you know|I mean)\b", "", line, flags=re.IGNORECASE)

                doc = self.nlp(line.lower())
                tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

                self.ts[i].append(" ".join(tokens))

    
    def runner(self) -> None:
        """
        Main runner function
        """
        self.load_transcripts()
        self.clean_transcripts()
    