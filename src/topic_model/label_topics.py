from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from loguru import logger


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

    def label_topics(
        self, 
        topic_words: Dict[int, List[str]]
    ) -> Dict[int, str]:
        """
        Function to label topics based on seed topics
        :param topic_words: dict containing topics words
        """
        logger.info('Labelling topics')
        
        topic_labels = {}

        for topic_id, words in topic_words.items():
            best_match = None
            highest_score = -1
            
            # Convert topic words into an embedding
            topic_embedding = self.model.encode(" ".join(words))

            for label, seed_words in self.seed_topics.items():
                seed_embedding = self.model.encode(" ".join(seed_words))
                similarity = cosine_similarity([topic_embedding], [seed_embedding])[0][0]
                
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = label
            
            topic_labels[topic_id] = best_match if best_match else "Uncategorized"

        return topic_labels
