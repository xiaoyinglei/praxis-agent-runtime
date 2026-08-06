__all__ = ["AgentLoop"]


def __getattr__(name: str) -> object:
    if name == "AgentLoop":
        from agent_runtime.loop.runtime import AgentLoop

        return AgentLoop
    raise AttributeError(name)
