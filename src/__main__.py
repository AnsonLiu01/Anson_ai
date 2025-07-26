from topic_model.get_topics import GetTopics

if __name__ == "__main__":
    topic_model = GetTopics(
        transcript_loc='/Users/ansonliu/Documents/Github/Other/carl_rogers_therapy_sessions.pdf',
        run_eda=True
    )
    
    topic_model.runner()