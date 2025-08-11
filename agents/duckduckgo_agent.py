# agents/duckduckgo_agent.py
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from core.state import AgentState
from langchain.schema import Document

class DuckDuckGoAgent:
    ddg_search = DuckDuckGoSearchRun()
    
    @classmethod
    def process(cls, state: AgentState) -> AgentState:
        try:
            content = cls.ddg_search.run(state["question"])
            if content:
                state["documents"] = [Document(page_content=content)]
                state["ddg_success"] = True
                state["conversation_history"].append("AI: Retrieved information from DuckDuckGo.")
            else:
                state["documents"] = []
                state["ddg_success"] = False
        except Exception:
            state["documents"] = []
            state["ddg_success"] = False

        state["ddg_attempted"] = True
        return state