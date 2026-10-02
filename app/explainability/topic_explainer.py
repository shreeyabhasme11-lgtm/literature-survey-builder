from app.metadata.schema import SentenceMeta

def explain_topic(s: SentenceMeta) -> str:
    if s.topic_id < 0:
        return "No topic assigned."
    return f"Topic {s.topic_id}: [{s.topic_keywords}]. Cluster: {s.subtopic_cluster}."