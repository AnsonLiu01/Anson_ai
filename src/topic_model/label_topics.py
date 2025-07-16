import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, List

from sentence_transformers import SentenceTransformer
from loguru import logger
from keybert import KeyBERT
import torch
from sentence_transformers import util


from src.utils import load_yaml

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"


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
        self.sentence_trans = SentenceTransformer("all-MiniLM-L6-v2")
        
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        self.predefined_labels = load_yaml(config_path=config_path)

    def label_with_keywords(
        self, 
        topic_words: Dict[int, List[str]], 
        predefined_labels: Dict[int, str], 
        threshold: float = 0.6
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
            doc_embedding = self.sentence_trans.encode(" ".join(val for val in doc), convert_to_tensor=True)
            similarity_scores = {
                predefined_labels[topic_id + 1]: util.pytorch_cos_sim(
                doc_embedding,
                self.sentence_trans.encode(predefined_labels[topic_id + 1], convert_to_tensor=True)
            ).item()
            }
            
            best_label = max(similarity_scores, key=similarity_scores.get)
            if similarity_scores[best_label] >= threshold:
                logger.info(f'topic_id {topic_id}: default topic label sufficient ')
                labels[topic_id] = best_label
            else:
                logger.info(f'topic_id {topic_id}: default topic label not relevant enough, extracting keywords')
                keywords = kw_model.extract_keywords(
                    doc, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english', 
                    top_n=2
                )
                dynamic_label = ", ".join([kw for kw, _ in keywords])
                labels[topic_id] = dynamic_label or "Uncategorized"

        return labels