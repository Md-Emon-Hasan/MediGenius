from core.state import AgentState
from langchain_core.documents import Document
from tools.search_tools import get_wikipedia_wrapper


def WikipediaAgent(state: AgentState) -> AgentState:
    wiki = get_wikipedia_wrapper()

    if not wiki:
        state["documents"] = []
        state["wiki_success"] = False
        state["wiki_attempted"] = True
        return state

    # Search with medical context
    search_query = f"{state['question']} medical symptoms treatment"
    content = wiki.run(search_query)

    if not content or len(content.strip()) < 100:
        # Fallback to simpler search
        content = wiki.run(state['question'])

    if content and len(content.strip()) > 100:
        state["documents"] = [Document(page_content=content)]
        state["wiki_success"] = True
        state["source"] = "Wikipedia Medical Information"
        print("Wikipedia: Found relevant content")
    else:
        state["documents"] = []
        state["wiki_success"] = False
        print("Wikipedia: No relevant content found")

    state["wiki_attempted"] = True
    return state
