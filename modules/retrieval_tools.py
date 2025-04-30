import logging
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from config.settings import PERSIST_DIR, COLLECTION_NAME, WIKI_TOP_K, WIKI_MAX_CHARS

logger = logging.getLogger("medical_chatbot")

class RetrievalSystem:
    def __init__(self, embeddings):
        self.vectorstore = self._initialize_vectorstore(embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 3})
        self.wiki_tool = self._initialize_wiki_tool()
        self.ddg_tool = DuckDuckGoSearchRun()

    def _initialize_vectorstore(self, embeddings):
        logger.info("Initializing vector store")
        return Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def _initialize_wiki_tool(self):
        return WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=WIKI_TOP_K,
                doc_content_chars_max=WIKI_MAX_CHARS,
                load_all_available_meta=True
            )
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        try:
            logger.debug(f"Retrieving documents for query: {query[:100]}...")
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Document retrieval error: {str(e)}")
            raise

    def retrieve_wikipedia(self, query: str) -> Document:
        try:
            logger.debug(f"Querying Wikipedia: {query[:100]}...")
            content = self.wiki_tool.run(query)
            return Document(page_content=content)
        except Exception as e:
            logger.error(f"Wikipedia retrieval error: {str(e)}")
            raise

    def retrieve_web(self, query: str) -> Document:
        try:
            logger.debug(f"Searching web: {query[:100]}...")
            content = self.ddg_tool.run(query)
            return Document(page_content=content)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            raise