import streamlit as st
from pathlib import Path
from config.settings import BASE_DIR
from modules.document_processing import DocumentProcessor
from modules.llm_models import LLMService
from modules.retrieval_tools import RetrievalSystem
from modules.state_management import StateManager
from modules.workflow import MedicalWorkflow

# CSS styling for messenger-style chat
st.markdown("""
<style>
    .chat-container { display: flex; flex-direction: column; }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 75%;
        font-size: 16px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #dcf8c6;
        align-self: flex-end;
        text-align: right;
        margin-left: auto;
    }
    .doctor-message {
        background-color: #f1f0f0;
        align-self: flex-start;
        text-align: left;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the system
@st.cache_resource
def initialize_system():
    llm_service = LLMService()
    doc_processor = DocumentProcessor()
    doc_processor.load_and_split_documents(str(BASE_DIR / "data" / "medical_book.pdf"))
    retrieval_system = RetrievalSystem(llm_service.embeddings)
    workflow = MedicalWorkflow(llm_service, retrieval_system)
    return workflow.compile(), StateManager()

# Session state init
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'app' not in st.session_state:
    st.session_state.app, st.session_state.state_manager = initialize_system()

# Header
st.title("🩺 MediGenius: AI Medical Consultation")
st.markdown("Ask your medical questions to our AI doctor.")
st.markdown("<small>Developed by <b>Md Emon Hasan</b></small>", unsafe_allow_html=True)

# Reset
if st.button("🔁 Reset Conversation"):
    st.session_state.state_manager.reset_conversation()
    st.session_state.conversation = []
    st.rerun()

# Show conversation in styled bubbles
chat_container = st.container()
chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.conversation:
    role_class = "user-message" if message["role"] == "user" else "doctor-message"
    sender = "" if message["role"] == "user" else ""
    chat_container.markdown(
        f'<div class="chat-bubble {role_class}"><b>{sender}</b>{message["content"]}</div>',
        unsafe_allow_html=True
    )

chat_container.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your medical question here...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})

    try:
        st.session_state.state_manager.update_state({
            "question": user_input,
            "documents": [],
            "generation": "",
            "source": ""
        })

        result = st.session_state.app.invoke(st.session_state.state_manager.get_state())
        st.session_state.state_manager.update_state(result)

        response = result.get("generation", "I couldn't generate a response to that question.")
        st.session_state.conversation.append({"role": "doctor", "content": response.strip()})
        st.rerun()

    except Exception:
        st.session_state.conversation.append({
            "role": "doctor",
            "content": "Sorry, I encountered an error processing your request."
        })
        st.rerun()

# Disclaimer
st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; background-color: #fff8e6; border-radius: 10px;">
    <small>
    <b>Disclaimer:</b> This AI medical assistant provides general health information and 
    should not be considered a substitute for professional medical advice, diagnosis, 
    or treatment. Always consult with a qualified healthcare provider for any medical concerns.
    </small>
</div>
""", unsafe_allow_html=True)
