# agents/wikipedia_agent.py
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from core.state import AgentState
from langchain.schema import Document

class WikipediaAgent:
    wiki = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=2000,
        load_all_available_meta=True
    )
    
    @classmethod
    def process(cls, state: AgentState) -> AgentState:
        try:
            content = cls.wiki.run(state["question"])
            if content:
                state["documents"] = [Document(page_content=content)]
                state["wiki_success"] = True
                state["conversation_history"].append("AI: Retrieved information from Wikipedia.")
            else:
                state["documents"] = []
                state["wiki_success"] = False
        except Exception:
            state["documents"] = []
            state["wiki_success"] = False

        state["wiki_attempted"] = True
        return state