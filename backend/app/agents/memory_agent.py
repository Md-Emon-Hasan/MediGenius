from core.state import AgentState


def MemoryAgent(state: AgentState) -> AgentState:
    history = state.get('conversation_history', [])
    if len(history) > 20:
        history = history[-20:]
    state['conversation_history'] = history
    return state
