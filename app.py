# File: app.py
from flask import Flask, request, jsonify
import logging
import nlp_processing
import retrieval_module
import augmented_generation
import personalization_engine
import knowledge_graph
import anti_hallucination

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return '''
        <form action="/query" method="post">
            <input type="text" name="query" placeholder="Enter your query">
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.form['query']
    logging.info(f"Received query: {user_query}")
    
    try:
        processed_query = nlp_processing.process_query(user_query)
        documents, images = retrieval_module.retrieve(processed_query)
        response = augmented_generation.generate_response(processed_query, documents, images)
        
        if not anti_hallucination.validate_response(response):
            response = "I'm sorry, I couldn't find reliable information on that topic."
        
        personalization_engine.update_preferences(user_query, response)
        knowledge_graph.update_graph(user_query, response)
        
        return jsonify(response=response, images=images)
    
    except Exception as e:
        logging.error(f"Error handling query: {e}")
        return jsonify(error="An error occurred while processing your query."), 500

if __name__ == '__main__':
    app.run(debug=True)
