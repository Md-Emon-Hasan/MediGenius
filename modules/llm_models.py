import logging
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config.settings import (
    GROQ_API_KEY, 
    LLM_MODEL_NAME, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    EMBEDDING_MODEL
)

logger = logging.getLogger("medical_chatbot")

class LLMService:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()

    def _initialize_llm(self):
        logger.info("Initializing LLM service")
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

    def _initialize_embeddings(self):
        logger.info("Initializing embeddings model")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def generate_response(self, prompt: str):
        try:
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise