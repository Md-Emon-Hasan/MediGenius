import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Global LLM instance
_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not found in environment variables")
            return None

        _llm_instance = ChatGroq(
            api_key=api_key,
            model_name="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=2048
        )
    return _llm_instance
