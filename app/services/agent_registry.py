class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, object] = {}

    def register(self, name: str, agent: object) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> object:
        return self._agents[name]
