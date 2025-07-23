import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
from typing import List, Tuple

import hdbscan
from umap import UMAP

import PyPDF2
import pandas as pd
import spacy
from bertopic import BERTopic
from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from keybert import KeyBERT

from label_topics import TopicLabeller
from src.topic_model.eda_topics import EDATopics
from src.utils import load_yaml


class GetTopics(EDATopics):
    """
    Class to get topics from each transcript
    """
    def __init__(
        self, 
        transcript_list: List[str],
        run_eda: bool
    ):
        self.transcript_list = transcript_list
        self.run_eda = run_eda
        
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
        self.raw = {}
        self.ts = {}
        self.topics = {}
        self.topic_info = {}
        
        self.bert = None
        self.filler_words = None
        self.tokeniser = None
    
    def init_tools(self) -> None:
        """
        Function to initialise tools
        """
        logger.info('Initialising tools')

        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        seed_words_config = load_yaml(config_path=config_path)
        
        self.labeller = TopicLabeller(seed_topics=seed_words_config)
        
        self.umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )

        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        self.bert = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            nr_topics='auto',
            calculate_probabilities=True  # probabilities needed to visualise
        )
        
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            logger.info('nltk stop words not found; downloading')
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info('nltk punkt tokeniser not found; downloading')
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info('nltk punkt_tab tokeniser not found; downloading')
            nltk.download('punkt_tab')
            
        self.tokeniser = PunktSentenceTokenizer()
                    
    def load_transcripts(self) -> None:
        """
        Function to load transcripts
        """
        for i, transcript in enumerate(self.transcript_list):
            logger.info(f'Loading transcript: {transcript}')
            if transcript.endswith('.txt'):
                with open(transcript, "r", encoding="utf-8") as file:
                    self.raw[i] = file.readlines()
            elif transcript.endswith('.pdf'):
                text_dict = {}
                with open(transcript, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(
                        5,  # TEMP current pdf transcript starts from 5
                        len(reader.pages)
                        ):
                        text_dict[page_num] = reader.pages[page_num].extract_text()
                    text_dict
                    self.raw[i] = text_dict
            else:
                raise NotImplementedError(f'File extension {transcript.split(".")[-1]} not supported for parsing')
            
        logger.info(f'total transcript count: {len(self.raw.keys())}')

    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts. Removes spqecial characters, lowers text, remove shorter dialogues and stems words (to their base form)
        """
        logger.info('Cleaning transcripts')
        
        for i, ts in self.raw.items():
            self.ts[i] = []

            for line in ts:
                line = re.sub(r"^(Therapist|Client):", "", line).strip()              
                doc = str(self.nlp(line.lower()))
                
                words_only = [
                    word for word in doc.split(' ') 
                    if (re.search(r'[a-zA-Z0-9]', word) or word in {'.', '!', '?'})  # Keep basic punctuation if needed
                    and word.strip()  # Exclude whitespace-only strings
                ]
                
                formatted_doc = " ".join(words_only)
                
                if formatted_doc:    
                    self.ts[i].append(formatted_doc)
    
    def extract_topics(
        self, 
        cleaned_ts: List[str]
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Applies BERTopic to extract topics from a list of cleaned transcripts.
        :param cleaned_transcripts: list of processed session transcripts
        :return: a list of topic labels assigned to each document and a summary of topics with top words.
        """
        topics, probs = self.bert.fit_transform(cleaned_ts)
        topic_info = self.bert.get_topic_info()
        
        self.get_topic_freq()

        return topics, topic_info

    def runner(self) -> None:
        """
        Main runner function
        """
        self.init_tools()   
        
        self.load_transcripts()
        self.clean_transcripts()
        
        for i, transcript in self.ts.items():
            logger.info(f'Extracting topics for transcript {i + 1}')
            self.topics[i], self.topic_info[i] = self.extract_topics(cleaned_ts=transcript)
            
            self.topic_info[i]['formatted_docs'] = [
                str(doc).strip("[]").replace("'", "")
                for doc in self.topic_info[i]['Representative_Docs']
            ]

            topic_words = self.topic_info[i].set_index("Topic").to_dict()['Representative_Docs']
            labelled_topics = self.labeller.label_with_keywords(
                topic_words=topic_words,
                predefined_labels=self.topic_info[i]['Name'].to_dict()
                )
            
            self.bert.set_topic_labels(labelled_topics)
            self.bert.visualize_barchart(custom_labels=labelled_topics).show()  # TODO: get visualisation to work
            

if __name__ == "__main__":
    a = GetTopics(
        transcript_list=[
            '/Users/ansonliu/Documents/Github/Anson_ai/data/transcripts/synthetic_test/depression_synthetic.txt',
            '/Users/ansonliu/Documents/Github/Other/carl_rogers_therapy_sessions.pdf'
            ],
        run_eda=True
    )
    
    a.runner()