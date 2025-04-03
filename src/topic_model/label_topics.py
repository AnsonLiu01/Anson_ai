import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from loguru import logger
from keybert import KeyBERT
import torch

from src.utils import load_yaml


class TopicLabeller:
    """
    Class to label topics based on seed topics
    """
    def __init__(
        self, 
        seed_topics: Dict[str, List[str]]
    ):
        """
        :param seed_topics: Dictionary of seed topics
        """
        self.seed_topics = seed_topics
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        self.predefined_labels = load_yaml(config_path=config_path)

    def label_with_keywords(
        self, 
        topic_words: List[str], 
        predefined_labels: Dict[int, str], 
        threshold: float = 0.3
    ) -> Dict[int, str]:
        """
        Labels topics using predefined labels or extracts keywords dynamically.

        :param topic_words: List of representative documents for each topic.
        :param predefined_labels: Dictionary of predefined topic labels.
        :param threshold: Similarity threshold for predefined labeling.
        :return: Dictionary of topic labels.
        """
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        labels = {}

        for topic_id, doc in topic_words.items():
            # Check predefined labels first
            similarity_scores = {
                label: self.model.similarity(doc, label) 
                for label in predefined_labels.values()
            }
            
            # Use predefined label if similarity is high enough
            best_label = max(similarity_scores, key=similarity_scores.get)
            if similarity_scores[best_label] >= threshold:
                labels[topic_id] = best_label
            else:
                # Extract keywords dynamically
                keywords = kw_model.extract_keywords(
                    doc, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english', 
                    top_n=2
                )
                dynamic_label = ", ".join([kw for kw, _ in keywords])
                labels[topic_id] = dynamic_label or "Uncategorized"

        return labels