class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def register(self, capability: str, tool: object) -> None:
        self._tools[capability] = tool

    def get(self, capability: str) -> object:
        return self._tools[capability]
