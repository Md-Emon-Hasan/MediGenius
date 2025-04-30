import logging
import logging.config
from pathlib import Path
from config.settings import BASE_DIR
from modules.document_processing import DocumentProcessor
from modules.llm_models import LLMService
from modules.retrieval_tools import RetrievalSystem
from modules.state_management import StateManager
from modules.workflow import MedicalWorkflow
import sys

# Ensure log directory exists
log_dir = BASE_DIR / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.config.fileConfig(BASE_DIR / 'config' / 'logging.conf')
logger = logging.getLogger("medical_chatbot")  # <- Correct logger name

def initialize_system():
    logger.info("Initializing Medical Chatbot System")
    
    # Initialize components
    llm_service = LLMService()
    doc_processor = DocumentProcessor()
    
    # Process documents
    doc_splits = doc_processor.load_and_split_documents(
        str(BASE_DIR / "data" / "medical_book.pdf")
    )
    
    # Initialize retrieval system
    retrieval_system = RetrievalSystem(llm_service.embeddings)
    
    # Create workflow
    workflow = MedicalWorkflow(llm_service, retrieval_system)
    app = workflow.compile()
    
    return app, StateManager()

def main():
    app, state_manager = initialize_system()
    
    print("=== MEDICAL CONVERSATION LOG ===")
    print("Type your question or 'exit' to end\n")
    
    while True:
        try:
            query = input().strip()
            
            if query.lower() == 'exit':
                state_manager.reset_conversation()
                print("\n=== Consultation Ended ===")
                break

            print(f"\n[You] {query}")
            
            # Update state
            state_manager.update_state({
                "question": query,
                "documents": [],
                "generation": "",
                "source": ""
            })
            
            # Process query
            result = app.invoke(state_manager.get_state())
            state_manager.update_state(result)
            
            # Show response
            if "generation" in result:
                response = result["generation"].strip()
                print(f"[AI] {response}")
            else:
                print("[AI] I couldn't generate a response.")
                
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            logger.info("Session terminated by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print("[AI] Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main()