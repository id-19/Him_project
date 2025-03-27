from flask import Flask, request, jsonify
from flask_cors import CORS
from processing_module import Conversationalist
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

groq_key = os.getenv("GROQ_API_KEY")
model = "gemma2-9b-it"
talker = Conversationalist(groq_api_key=groq_key)
# talker = Conversationalist()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message'}), 400
    
    response = talker.process_query(data["message"])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)