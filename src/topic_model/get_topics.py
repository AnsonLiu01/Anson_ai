import os
import re
import sys
from typing import List, Tuple

import hdbscan
import nltk
import numpy as np
import pandas as pd
import PyPDF2
import spacy
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from docx import Document
from kneed import KneeLocator
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.topic_model.eda_topics import EDATopics
from src.utils import load_yaml
from src.topic_model.label_topics import TopicLabeller

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        
        self.labelled_topics = None
        self.representative_docs = {}
    
    def init_tools(self) -> None:
        """
        Function to initialise tools
        """
        logger.info('Initialising tools')

        config_path = os.path.join(os.path.dirname(__file__), 'config', 'seed_words.yaml')
        seed_words_config = load_yaml(config_path=config_path)
        
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            logger.info('nltk stop words not found; downloading')
            nltk.download("stopwords")
            
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

        self.labeller = TopicLabeller(seed_topics=seed_words_config)
        
        self.umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.bert = BERTopic(
            umap_model=self.umap_model,
            ctfidf_model=ctfidf_model,
            nr_topics='auto',
            calculate_probabilities=True
        )
                                        
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
                for page_num in range(0, len(reader.pages)):
                    text_dict[page_num] = reader.pages[page_num].extract_text()
                
                self.raw_ts = ''.join(text_dict.values())
        elif self.transcript_loc.endswith('.docx'):
            raw_doc = Document(self.transcript_loc)
            doc = []
            for paragraph in raw_doc.paragraphs:
                text = paragraph.text.strip()
                doc.append(text)

            self.raw_ts = " ".join(line for line in doc)
        else:
            raise NotImplementedError(f'File extension {self.transcript_loc.split(".")[-1]} not supported for parsing')
        
    def clean_transcripts(self) -> None:
        """
        Function to clean transcripts. Removes spqecial characters, lowers text, remove shorter dialogues and stems words (to their base form)
        """
        logger.info('Cleaning transcript')
        
        processed = (
            re.sub(r'\b[A-Za-z]\s?\d{1,2}\b', '', self.raw_ts)
            .replace('\\', '')
            .translate(str.maketrans('\n\t', '  '))
            .strip()
        )
        processed = re.sub(r'\s{2,}', ' ', processed)
        
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
    
    def tune_hdbscan(self) -> None:
        """
        Function to tune HDBSCAN hyperparameters.
        - Finds optimal epsilon via knee detection
        - Performs grid search across parameters
        - Plots DBCV scores across parameter sets
        """
        logger.info('Tuning HDBSCAN hyperparameters')

        embeddings = self.embedding_model.encode(
            sentences=self.formatted_ts,
            show_progress_bar=True
        )
        norm_embeddings = normalize(embeddings, norm='l2')

        norm_embeddings = UMAP(
            n_components=50,
            metric='cosine',
            random_state=42
        ).fit_transform(norm_embeddings)

        # Estimate epsilon using k-NN distance + knee detection
        neighbors = NearestNeighbors(n_neighbors=10).fit(norm_embeddings)
        distances, _ = neighbors.kneighbors(norm_embeddings)
        distances = np.sort(distances[:, -1])
        knee = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        optimal_epsilon = distances[knee.knee]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(distances)), distances, label="k-distance curve")
        if knee.knee is not None:
            plt.axvline(knee.knee, color="red", linestyle="--", label=f"Knee at {knee.knee}")
        plt.title("Knee Detection for Optimal Epsilon")
        plt.xlabel("Points sorted by distance")
        plt.ylabel("Distance to 10th nearest neighbor")
        plt.legend()
        plt.grid(True)
        plt.show()

        param_grid = {
            "min_cluster_size": [3, 5, 10, 15, 20],
            "cluster_selection_epsilon": [optimal_epsilon, 0.9 * optimal_epsilon, 1.1 * optimal_epsilon],
            "min_samples": [1, 3, 5, 10],
        }

        best_score = -1
        best_params = {}
        scores = []
        params_list = []

        # Grid search
        logger.info('HDBSCAN Parameter Grid Search')
        for params in tqdm(ParameterGrid(param_grid)):
            params = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                    if isinstance(v, (np.integer, int)) else v
                    for k, v in params.items()}

            clusterer = hdbscan.HDBSCAN(
                metric="euclidean",
                cluster_selection_method="eom",
                **params
            ).fit(norm_embeddings)

            norm_embeddings_float64 = np.array(norm_embeddings, dtype=np.float64)
            score = hdbscan.validity.validity_index(norm_embeddings_float64, clusterer.labels_)

            scores.append(score)
            params_list.append(params)

            if score > best_score:
                best_score = score
                best_params = params

        # Fit final model
        self.bert.hdbscan_model = hdbscan.HDBSCAN(
            **best_params,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )
        logger.success(f"Best HDBSCAN params: {best_params} (DBCV: {best_score:.2f})")

        # Plot scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(scores)), scores, marker="o")
        plt.title("DBCV Scores across HDBSCAN parameter sets")
        plt.xlabel("Parameter Set Index")
        plt.ylabel("DBCV Score")
        plt.grid(True)

        # Annotate best point
        best_idx = int(np.argmax(scores))
        plt.scatter(best_idx, scores[best_idx], color="red", s=100,
                    label=f"Best: {scores[best_idx]:.2f}")
        plt.legend()
        plt.show()
    
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
        
        self.eda_get_topic_freq()

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
        logger.info('Collecting all visualisations')
        
        self.bert.visualize_document_datamap(docs=self.formatted_ts, custom_labels=self.labelled_topics).show()
        self.bert.visualize_barchart(custom_labels=self.labelled_topics, top_n_topics=10).show()
        
        self.eda_visual_similarity_heatmap()
                
    def get_best_representative_docs(self) -> None:
        """
        Function to get the best representative docs for each labelled topic
        """
        logger.info('Collecting best representated docs for all topics')
        for n_topic, label in self.labelled_topics.items():
            if n_topic != -1:
                self.representative_docs[label] = self.bert.get_representative_docs(n_topic)
        
    def get_dashboard_info(self) -> None:
        """
        Function to get all data, visualisations and info in preparation for dashboard initialisation
        """
        logger.info('Collecting all dashboard info')
        
        self.get_best_representative_docs()
        
        self.visualisations()
        
    def runner(self) -> None:
        """
        Main runner function
        """
        self.init_tools()   
        
        self.load_transcripts()
        self.clean_transcripts()
        
        self.tune_hdbscan()
        
        self.extract_topics(cleaned_ts=self.formatted_ts)
        self.label_topics()
        
        self.get_dashboard_info()
                
        """
        TODO:
        - epsilon tuning, don't have to pass in maybe? HSBSCAN figures out itself compared to DBSCAN, fact check this
        - topics over time use n entries / session length for artificial timestamp
        - make represntative docs visible per topic
        """