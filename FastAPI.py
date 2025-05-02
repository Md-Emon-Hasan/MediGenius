import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from modules.document_processing import DocumentProcessor
from modules.llm_models import LLMService
from modules.retrieval_tools import RetrievalSystem
from modules.state_management import StateManager
from modules.workflow import MedicalWorkflow
from config.settings import BASE_DIR

# Initialize FastAPI app
app = FastAPI(title="MediGenius: AI Medical Assistant API")

# Initialize system components (cached to avoid reloading on every request)
def initialize_system():
    llm_service = LLMService()
    doc_processor = DocumentProcessor()
    doc_processor.load_and_split_documents(str(BASE_DIR / "data" / "medical_book.pdf"))
    retrieval_system = RetrievalSystem(llm_service.embeddings)
    workflow = MedicalWorkflow(llm_service, retrieval_system)
    return workflow.compile(), StateManager()

# Session state init
app_state = {
    "conversation": [],
    "app": None,
    "state_manager": None
}

# Initialize the system only once
if app_state["app"] is None or app_state["state_manager"] is None:
    app_state["app"], app_state["state_manager"] = initialize_system()

# Pydantic model for request body
class UserQuery(BaseModel):
    question: str

# Route to send query to the medical assistant
@app.post("/ask/")
async def ask_medical_assistant(query: UserQuery):
    try:
        user_input = query.question

        # Update state
        app_state["state_manager"].update_state({
            "question": user_input,
            "documents": [],
            "generation": "",
            "source": ""
        })

        # Process the query
        result = app_state["app"].invoke(app_state["state_manager"].get_state())
        app_state["state_manager"].update_state(result)

        response = result.get("generation", "I couldn't generate a response to that question.")

        # Store the conversation history
        app_state["conversation"].append({"role": "user", "content": user_input})
        app_state["conversation"].append({"role": "doctor", "content": response.strip()})

        return {"response": response.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

# Route to get the entire conversation history
@app.get("/conversation/")
async def get_conversation():
    return {"conversation": app_state["conversation"]}

# Route to reset the conversation
@app.post("/reset/")
async def reset_conversation():
    app_state["state_manager"].reset_conversation()
    app_state["conversation"] = []
    return {"message": "Conversation reset successfully"}

if __name__ == "__main__":
    # To run the FastAPI app with Uvicorn (when executing this script directly)
    uvicorn.run(app, host="0.0.0.0", port=8000)
