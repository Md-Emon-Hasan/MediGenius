# api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from core.langgraph_workflow import setup_workflow
from core.state import initialize_state
from pydantic import BaseModel

app = FastAPI(title="Medical AI Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = setup_workflow()

# Session storage (in production, use Redis or database)
sessions = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    conversation_id: str

@app.post("/chat")
async def chat_handler(chat_request: ChatRequest):
    try:
        # Get or create conversation
        if chat_request.conversation_id and chat_request.conversation_id in sessions:
            conversation_data = sessions[chat_request.conversation_id]
        else:
            conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
            conversation_data = {
                "history": [],
                "state": initialize_state()
            }
            sessions[conversation_id] = conversation_data
            chat_request.conversation_id = conversation_id

        # Update conversation history
        conversation_data["history"].append(f"User: {chat_request.message}")
        
        # Prepare state
        conversation_data["state"].update({
            "question": chat_request.message,
            "conversation_history": conversation_data["history"]
        })

        # Process through workflow
        result = workflow.invoke(conversation_data["state"])
        
        # Update history with response
        conversation_data["history"].append(f"Doctor: {result.get('generation', '')}")
        
        return JSONResponse({
            "response": result.get("generation", "I couldn't generate a response."),
            "timestamp": datetime.now().strftime("%H:%M"),
            "conversation_id": chat_request.conversation_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)