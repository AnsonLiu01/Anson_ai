from loguru import logger


class EDATopics:
    """
    Class for topic eda
    """
    def __init__(self):
        self.bert = None
        self.run_eda: bool = None
    
    def run_eda_check(func):
        """
        Decorator to execute pre-call logic before any function
        """
        def function(self, *args, **kwargs):
            if self.run_eda:
                logger.info(f"⚡EDA enabled, running function {func.__name__}")
                print('============= EDA =============\n')
                result = func(self, *args, **kwargs)
                print('\n=========== EDA EXIT ==========')
            else:
                logger.debug(f'EDA disabled, not running function {func.__name__}')
            
            return result
        return function

    @run_eda_check
    def eda_get_topic_freq(self) -> None:
        """
        Function to get topic frequencies and of which are outliers
        """
        topic_freq = self.bert.get_topic_freq()
        outliers = topic_freq['Count'][topic_freq['Topic'] == -1].iloc[0]
        
        print(f'{outliers} documents have not been classified')
        print(f'The other {topic_freq["Count"].sum() - outliers} documents are {topic_freq["Topic"].shape[0]-1} topics')
        
        print(f'All topic frequencies:\n{topic_freq[["Topic", "Count"]]}')
    
    @run_eda_check
    def eda_visual_similarity_heatmap(self) -> None:
        """
        Function to visualise topic similarity
        """
        self.bert.visualize_heatmap(custom_labels=self.labelled_topics).show()
        