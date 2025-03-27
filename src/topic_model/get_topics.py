import os
import re
import spacy
import yaml

import pandas as pd
from loguru import logger
from typing import List, Tuple, Dict, Any
from nltk.corpus import stopwords
import nltk
from bertopic import BERTopic
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np

from label_topics import TopicLabeller


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
        self.topics = {}
        self.topic_info = {}
        
        self.model = None
        self.filler_words = None
    
    @staticmethod
    def load_yaml(
        config_path: str
    ) -> Dict[Any, Any]:
        """
        Function to load yaml file
        """
        logger.info('Loading yaml file')
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def init_tools(self) -> None:
        """
        Function to initialise tools
        """
        logger.info('Initialising tools')

        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        seed_words_config = self.load_yaml(config_path=config_path)
        
        self.labeller = TopicLabeller(seed_topics=seed_words_config)
        
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        self.model = BERTopic(
            hdbscan_model=self.hdbscan_model,
            nr_topics=5
        )

        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            logger.info('nltk stop words not found, downloading')
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
                
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
                line = re.sub(r"^(Therapist|Client):", "", line).strip()              
                doc = str(self.nlp(line.lower()))
                
                if doc != '':
                    self.ts[i].append(doc)
    
    def extract_topics(
        self, 
        cleaned_ts: List[str]
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Applies BERTopic to extract topics from a list of cleaned transcripts.
        :param cleaned_transcripts: list of processed session transcripts
        :return: a list of topic labels assigned to each document and a summary of topics with top words.
        """
        topics, _ = self.model.fit_transform(cleaned_ts)
        topic_info = self.model.get_topic_info()

        return topics, topic_info

    def runner(self) -> None:
        """
        Main runner function
        """
        self.init_tools()
        
        self.load_transcripts()
        self.clean_transcripts()
        
        for i, transcript in self.ts.items():
            logger.info(f'Extracting topics for transcript {i}')
            self.topics[i], self.topic_info[i] = self.extract_topics(cleaned_ts=transcript)
            
            topic_words = self.topic_info[i].set_index("Topic").to_dict()['Representative_Docs']
            labelled_topics = self.labeller.label_topics(topic_words)  # TODO: topics are all identical, fix to ensure all topics are different but still accurate

            self.model.visualize_barchart().show()


if __name__ == "__main__":
    a = GetTopics(transcript_list=['/Users/ansonliu/Documents/Github/Anson_ai/data/transcripts/synthetic_test/depression_synthetic.txt'])
    
    a.runner()