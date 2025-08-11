# main.py
from dotenv import load_dotenv
from core.langgraph_workflow import setup_workflow
from core.state import initialize_state

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the workflow and state
    app = setup_workflow()
    conversation_state = initialize_state()
    
    print("=== Medical AI Assistant (Type 'exit' to quit) ===")
    
    while True:
        query = input("\nAsk your medical question: ").strip()
        
        if query.lower() == "exit":
            conversation_state = initialize_state()  # Reset state
            print("\n=== Consultation Ended. Conversation history cleared. ===")
            break
            
        # Update state with new question
        conversation_state.update({
            "question": query,
            "documents": [],
            "generation": "",
            "source": "",
            "search_query": None,
            "llm_attempted": False,
            "llm_success": False,
            "rag_attempted": False,
            "rag_success": False,
            "wiki_attempted": False,
            "wiki_success": False,
            "ddg_attempted": False,
            "ddg_success": False,
            "current_tool": None,
            "retry_count": 0
        })
        
        # Run the workflow
        result = app.invoke(conversation_state)
        conversation_state.update(result)
        
        # Print response
        if result.get("generation"):
            print(f"\n[Doctor AI] {result['generation']}")
        else:
            print("\n[Doctor AI] Sorry, I couldn't generate a response.")
            
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()