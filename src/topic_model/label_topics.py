import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, List

from sentence_transformers import SentenceTransformer
from loguru import logger
from keybert import KeyBERT
from sentence_transformers import util
from transformers import pipeline
from textblob import TextBlob

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
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        self.predefined_labels = load_yaml(config_path=config_path)

    def label_with_keywords(
        self, 
        topic_words: Dict[int, List[str]], 
        predefined_labels: Dict[int, str], 
        threshold: float = 0.7
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
            label_id = topic_id + 1 if -1 in topic_words.keys() else topic_id           
            doc_embedding = self.embedding_model.encode(" ".join(val for val in doc), convert_to_tensor=True)
            similarity_scores = {
                predefined_labels[label_id]: util.pytorch_cos_sim(
                doc_embedding,
                self.embedding_model.encode(predefined_labels[label_id], convert_to_tensor=True)
            ).item()
            }
            
            default_label = max(similarity_scores, key=similarity_scores.get)
            if similarity_scores[default_label] >= threshold:
                default_label = ' '.join(default_label.split('_')[1:])
                logger.success(f'topic_id {topic_id}: default topic label sufficient ({default_label})')
                labels[topic_id] = default_label
            else:
                logger.info(f'topic_id {topic_id}: default topic label not relevant enough, extracting keywords')
                keywords = kw_model.extract_keywords(
                    doc, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english',
                    use_mmr=True,
                    nr_candidates=20,
                    top_n=2
                )
                
                top_label, top_similarity = None, -1
                
                for kw_chunk in keywords:
                    if top_similarity < kw_chunk[0][1]:
                        top_label, top_similarity = kw_chunk[0][0], kw_chunk[0][1]
                        logger.debug(f'NEW best label: {top_label} (similarity score: {top_similarity})')
                                
                logger.success(f'Using new keyword label for topic id {topic_id}: {top_label}')

                labels[topic_id] = str(top_label) or 'Uncategorised'

        return labels
    
    @staticmethod
    def refine_label(
        label: str,
        doc: List[str],
        method: str
    ) -> str:
        """
        Function to refine label to be grammatically correct
        :param label: label to refine
        :param doc: main document for additional context
        :param method: method of refinement
        :return: grammatically correct  and refined label
        """
        logger.debug(f'Refining label: {label}, using {method} method')
        
        if method == 'textblob':
            blob = TextBlob(label)
            refined = blob.correct()
        elif method == 't5':
            t5_grammar = pipeline("text2text-generation", model="google/flan-t5-base")
            
            prompt = f"""The string '{label}' is the output of a keyword-focused approach for labels for a topic mapping task focused around therapy.\n 
            Can you correct the grammar or use these keywords to create a coherent and meaningful phrase.\n
            Below is the document from which these keywords have been obtained from, use this as additional context when
            creating a short coherent phrase for the label for the current topic\n
            supporting context document below:\n
            {' '.join(val for val in doc)}
            Your output should only include the new refined phrase as the new label
            """
            
            refined = t5_grammar(prompt, max_length=20)[0]['generated_text']
        else:
            raise ValueError(f'Method must be one of [textblob, t5], got: {method}')
        return refined.strip()  
