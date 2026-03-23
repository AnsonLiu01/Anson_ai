from topic_model.get_topics import GetTopics
from dashboard.dashboard_plotly import MainDashboard

if __name__ == "__main__":
    topic_model = GetTopics(
        transcript_loc='/Users/ansonliu/Github/Other/transcripts/carl_rogers_therapy_sessions_gloria.docx',
        run_eda=True,
        hp_tune=False,
        show_local_visualisations=False
    )
    
    topic_model.runner()

    dashboard = MainDashboard(topic_model, port=8050)
    dashboard.runner(debug=False)
