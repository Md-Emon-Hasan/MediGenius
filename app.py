# app.py
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import session
from core.langgraph_workflow import setup_workflow
from core.state import initialize_state
from datetime import datetime
import os
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize the workflow
workflow = setup_workflow()

@app.route('/')
def home():
    session['conversation_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    conversation_state = initialize_state()
    
    if 'history' not in session:
        session['history'] = []
    
    session['history'].append(f"User: {user_input}")
    
    conversation_state.update({
        "question": user_input,
        "conversation_history": session['history']
    })
    
    result = workflow.invoke(conversation_state)
    session['history'].append(f"Doctor: {result.get('generation', '')}")
    
    return jsonify({
        'response': result.get("generation", "I couldn't generate a response."),
        'timestamp': datetime.now().strftime("%H:%M")
    })

if __name__ == '__main__':
    app.run(debug=True)