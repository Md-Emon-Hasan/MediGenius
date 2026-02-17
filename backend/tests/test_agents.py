import os
import sys
from unittest.mock import MagicMock, patch

from agents.executor_agent import ExecutorAgent
from agents.explanation_agent import ExplanationAgent
from agents.llm_agent import LLMAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.tavily_agent import TavilyAgent
from agents.wikipedia_agent import WikipediaAgent
from core.state import initialize_conversation_state
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))


# Import agents


# --- Planner Agent Tests ---
def test_planner_agent_medical():
    state = initialize_conversation_state()
    # "fever" is in the medical keyword list
    state["question"] = "I have a high fever"
    new_state = PlannerAgent(state)
    assert new_state["current_tool"] == "retriever"


def test_planner_agent_general():
    state = initialize_conversation_state()
    state["question"] = "Hello there"
    new_state = PlannerAgent(state)
    assert new_state["current_tool"] == "llm_agent"


# --- Retriever, LLM, Wiki, Tavily Tests (Existing) ---
def test_retriever_agent_success():
    state = initialize_conversation_state()
    state["question"] = "fever"

    with patch('agents.retriever_agent.get_retriever') as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="Fever details " * 10)]
        mock_get_retriever.return_value = mock_retriever

        new_state = RetrieverAgent(state)
        assert new_state["rag_success"] is True
        assert len(new_state["documents"]) > 0


def test_retriever_agent_failure():
    state = initialize_conversation_state()
    state["question"] = "unknown"
    with patch('agents.retriever_agent.get_retriever') as mock_get:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get.return_value = mock_retriever

        new_state = RetrieverAgent(state)
        assert new_state["rag_success"] is False


def test_llm_agent():
    state = initialize_conversation_state()
    state["question"] = "Hi"
    with patch('agents.llm_agent.get_llm') as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Hello there my friend, this is a long enough response."
        mock_get.return_value = mock_llm

        new_state = LLMAgent(state)
        assert new_state["llm_success"] is True
        assert new_state["generation"] == "Hello there my friend, this is a long enough response."


def test_wikipedia_agent():
    state = initialize_conversation_state()
    state["question"] = "Flu"
    with patch('agents.wikipedia_agent.get_wikipedia_wrapper') as mock_get:
        mock_wiki = MagicMock()
        mock_wiki.run.return_value = "Flu information is very important to know. " * 10
        mock_get.return_value = mock_wiki
        new_state = WikipediaAgent(state)
        assert new_state["wiki_success"] is True


def test_tavily_agent():
    state = initialize_conversation_state()
    state["question"] = "News"
    with patch('agents.tavily_agent.get_tavily_search') as mock_get:
        mock_tav = MagicMock()
        mock_tav.invoke.return_value = [
            {"content": "News about medical discoveries is important. " * 5, "url": "http://news.com"}]
        mock_get.return_value = mock_tav
        new_state = TavilyAgent(state)
        assert new_state["tavily_success"] is True


# --- New Agent Tests ---

def test_memory_agent():
    state = initialize_conversation_state()
    # Create a history > 20 items
    state["conversation_history"] = [{"role": "user", "content": str(i)} for i in range(25)]

    new_state = MemoryAgent(state)

    # Should be truncated to last 20
    assert len(new_state["conversation_history"]) == 20
    assert new_state["conversation_history"][-1]["content"] == "24"


def test_executor_agent_with_docs():
    state = initialize_conversation_state()
    state["question"] = "What is X?"
    state["documents"] = [Document(page_content="X is Y.")]

    with patch('agents.executor_agent.get_llm') as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "X is likely Y based on docs."
        mock_get_llm.return_value = mock_llm

        new_state = ExecutorAgent(state)

        assert new_state["generation"] == "X is likely Y based on docs."
        # Should append to history
        assert len(new_state["conversation_history"]) == 2  # user + assistant


def test_executor_agent_no_llm():
    state = initialize_conversation_state()
    with patch('agents.executor_agent.get_llm') as mock_get_llm:
        mock_get_llm.return_value = None
        new_state = ExecutorAgent(state)
        assert "temporarily unavailable" in new_state["generation"]


def test_explanation_agent():
    state = initialize_conversation_state()
    new_state = ExplanationAgent(state)
    assert new_state == state
