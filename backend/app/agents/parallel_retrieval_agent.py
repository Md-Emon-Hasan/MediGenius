"""
MediGenius — agents/parallel_retrieval_agent.py
ParallelRetrievalAgent: fans RAG, Wikipedia, and Tavily out concurrently instead of
trying them one at a time. I/O-bound retrieval only — no LLM call happens here.
Supersedes the old sequential retriever -> wikipedia -> tavily chain; those agent
files stay in place and unit-tested, they're just no longer wired into the graph
(the same pattern Phase 4 used for llm_client.py).
"""

import asyncio

from langchain_core.documents import Document

from app.core import cache
from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools.tavily_search import get_tavily_search
from app.tools.vector_store import get_retriever
from app.tools.wikipedia_search import get_wikipedia_wrapper

BRANCH_TIMEOUT = 8.0

# authority order when more than one branch succeeds: curated PDF > live web > general encyclopedia
_PRIORITY = ["Medical Literature Database", "Current Medical Research & News", "Wikipedia Medical Information"]


def _rag_sync(query: str):
    # get_retriever() itself is blocking (loads embeddings/Chroma on a cold start) — must not run on the event loop
    retriever = get_retriever()
    if not retriever:
        return None, "rag"
    cached = cache.get_retrieval(query)
    docs = cached if cached is not None else retriever.invoke(query)
    if cached is None:
        cache.set_retrieval(query, docs)
    valid = [d for d in (docs or []) if len(d.page_content.strip()) > 50]
    return (valid, "Medical Literature Database") if valid else (None, "rag")


def _wiki_sync(query: str):
    wiki = get_wikipedia_wrapper()
    if not wiki:
        return None, "wiki"
    search_query = f"{query} medical symptoms treatment"
    cached = cache.get_retrieval(search_query)
    content = cached if cached is not None else wiki.run(search_query)
    if cached is None:
        cache.set_retrieval(search_query, content)
    if content and len(content.strip()) > 100:
        return [Document(page_content=content)], "Wikipedia Medical Information"
    return None, "wiki"


def _tavily_sync(query: str):
    tavily = get_tavily_search()
    if not tavily:
        return None, "tavily"
    search_query = f"{query} medical health treatment symptoms"
    try:
        results = tavily.invoke(search_query)
    except Exception as e:
        logger.error("parallel_retrieval: tavily failed: %s", str(e))
        return None, "tavily"
    valid = [r for r in (results or []) if isinstance(r, dict) and r.get("content") and len(r["content"].strip()) > 50]
    if not valid:
        return None, "tavily"
    docs = [
        Document(page_content=r["content"], metadata={"url": r.get("url", ""), "title": r.get("title", "")})
        for r in valid
    ]
    return docs, "Current Medical Research & News"


async def _with_timeout(sync_fn, query: str, name: str):
    try:
        return await asyncio.wait_for(asyncio.to_thread(sync_fn, query), timeout=BRANCH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("parallel_retrieval: %s branch timed out", name)
        return None, name
    except Exception as e:
        logger.error("parallel_retrieval: %s branch failed: %s", name, str(e))
        return None, name


async def ParallelRetrievalAgent(state: AgentState) -> AgentState:
    """Fan out RAG, Wikipedia, and Tavily concurrently and keep the highest-authority hit."""
    query = state["question"]
    results = await asyncio.gather(
        _with_timeout(_rag_sync, query, "rag"),
        _with_timeout(_wiki_sync, query, "wiki"),
        _with_timeout(_tavily_sync, query, "tavily"),
    )
    by_source = {source: docs for docs, source in results if docs}

    for source in _PRIORITY:
        if source in by_source:
            state["documents"] = by_source[source]
            state["source"] = source
            state["rag_success"] = True
            state["rag_attempted"] = True
            logger.info("parallel_retrieval: served from %s", source)
            return state

    state["documents"] = []
    state["rag_success"] = False
    state["rag_attempted"] = True
    logger.info("parallel_retrieval: no branch returned usable content")
    return state
