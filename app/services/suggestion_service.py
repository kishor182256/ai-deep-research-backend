class SuggestionService:
    def generate(self, topic: str) -> list[dict[str, str | float]]:
        normalized_topic = topic.strip()
        templates = [
            "What are the most important recent developments in {topic}?",
            "Who are the key players shaping {topic}?",
            "What are the biggest opportunities and risks in {topic}?",
            "How has {topic} changed over the last few years?",
            "What data and statistics best explain {topic}?",
            "What are experts saying about {topic}?",
            "How does {topic} compare across countries, companies, or communities?",
            "What are the strongest arguments for and against {topic}?",
            "What should beginners understand first about {topic}?",
            "What is likely to happen next in {topic}?",
        ]
        return [
            {
                "title": template.format(topic=normalized_topic),
                "summary": f"A focused research direction for {normalized_topic}.",
                "score": round(0.95 - (index * 0.03), 2),
                "reason": "Useful generic angle with broad source availability.",
            }
            for index, template in enumerate(templates)
        ]
