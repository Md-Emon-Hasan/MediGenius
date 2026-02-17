from core.state import AgentState
from langchain_core.documents import Document
from tools.search_tools import get_tavily_search


def TavilyAgent(state: AgentState) -> AgentState:
    tavily_search = get_tavily_search()

    if not tavily_search:
        state["documents"] = []
        state["tavily_success"] = False
        state["tavily_attempted"] = True
        return state

    # Add medical context to search
    search_query = f"{state['question']} medical health treatment symptoms"
    results = tavily_search.invoke(search_query)

    if results and len(results) > 0:
        valid_results = []
        for res in results:
            if isinstance(res, dict) and res.get("content") and len(res["content"].strip()) > 50:
                valid_results.append(res)

        if valid_results:
            docs = [Document(
                page_content=res["content"],
                metadata={"url": res.get("url", ""), "title": res.get("title", "")}
            ) for res in valid_results]
            state["documents"] = docs
            state["tavily_success"] = True
            state["source"] = "Current Medical Research & News"
            print(f"Tavily: Found {len(valid_results)} results")
        else:
            state["documents"] = []
            state["tavily_success"] = False
            print("Tavily: No valid results")
    else:
        state["documents"] = []
        state["tavily_success"] = False
        print("Tavily: No results found")

    state["tavily_attempted"] = True
    return state
