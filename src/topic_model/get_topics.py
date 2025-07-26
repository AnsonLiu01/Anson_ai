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
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize

from topic_model.label_topics import TopicLabeller
from src.topic_model.eda_topics import EDATopics
from src.utils import load_yaml


class GetTopics(EDATopics):
    """
    Class to get topics from each transcript
    """
    def __init__(
        self, 
        transcript_loc: str,
        run_eda: bool
    ):
        self.transcript_loc = transcript_loc
        self.run_eda = run_eda
        
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
        self.raw_ts = None
        self.formatted_ts = None
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
            calculate_probabilities=True
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
        logger.info(f'Loading transcript: {self.transcript_loc}')
        if self.transcript_loc.endswith('.txt'):
            with open(self.transcript_loc, "r", encoding="utf-8") as file:
                self.raw_ts = ''.join(file.readlines())
        elif self.transcript_loc.endswith('.pdf'):
            text_dict = {}
            with open(self.transcript_loc, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(
                    4,  # TEMP current pdf transcript starts from 5
                    len(reader.pages)
                    ):
                    text_dict[page_num] = reader.pages[page_num].extract_text()
                
                self.raw_ts = ''.join(text_dict.values())
        else:
            raise NotImplementedError(f'File extension {self.transcript_loc.split(".")[-1]} not supported for parsing')
        
    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts. Removes spqecial characters, lowers text, remove shorter dialogues and stems words (to their base form)
        """
        logger.info('Cleaning transcript')
        processed = re.sub(r'\b\w+:\s*', '', self.raw_ts)
        processed = processed.replace(r'\\+', '')
        processed = processed.replace('\n', ' ') 
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        doc = str(self.nlp(processed.lower()))
        
        words_only = [
            word for word in doc.split(' ') 
            if (re.search(r'[a-zA-Z0-9]', word) or word in {'.', '!', '?'})
            and word.strip()
        ]
        
        formatted_doc = " ".join(words_only)
        
        if formatted_doc:
            formatted_doc = [formatted_doc]
            documents = sent_tokenize(formatted_doc[0])
            self.formatted_ts = [doc for doc in documents if len(doc.split()) > 3]
        else:
            raise ValueError(f'No formatted words found or recognised in transcript')
    
    def extract_topics(
        self, 
        cleaned_ts: List[str]
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Applies BERTopic to extract topics from a list of cleaned transcripts.
        :param cleaned_transcripts: list of processed session transcripts
        :return: a list of topic labels assigned to each document and a summary of topics with top words.
        """
        logger.info(f'Extracting topics')

        topics, probs = self.bert.fit_transform(documents=cleaned_ts)
        topic_info = self.bert.get_topic_info()
        
        self.get_topic_freq()

        self.topics, self.topic_info = topics, topic_info
    
    def label_topics(self) -> None:
        """
        Function to label topics
        """
        self.topic_info['formatted_docs'] = [
            str(doc).strip("[]").replace("'", "")
            for doc in self.topic_info['Representative_Docs']
            ]

        topic_words = self.topic_info.set_index("Topic").to_dict()['Representative_Docs']
        self.labelled_topics = self.labeller.label_with_keywords(
            topic_words=topic_words,
            predefined_labels=self.topic_info['Name'].to_dict()
            )
            
        self.bert.set_topic_labels(self.labelled_topics)

    def visualisations(self) -> None:
        """
        Function to process all visualisations
        """
        self.bert.visualize_topics(custom_labels=self.labelled_topics).show()
        self.bert.visualize_barchart(custom_labels=self.labelled_topics).show()
        
        self.bert.visualize_heatmap(custom_labels=self.labelled_topics).show()
        self.bert.visualize_document_datamap(docs=self.formatted_ts, custom_labels=self.labelled_topics).show()
        
    def runner(self) -> None:
        """
        Main runner function
        """
        self.init_tools()   
        
        self.load_transcripts()
        self.clean_transcripts()
        
        self.extract_topics(cleaned_ts=self.formatted_ts)
        self.label_topics()
        
        self.visualisations()
        