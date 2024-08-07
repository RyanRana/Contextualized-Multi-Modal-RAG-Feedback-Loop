from flask import Flask, request, jsonify
import logging
import spacy
import requests
import openai
import json
import networkx as nx
from bs4 import BeautifulSoup
import urllib.parse

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Initialize OpenAI API (make sure to set your OpenAI API key)
openai.api_key = 'your-openai-api-key'

# In-memory storage for user preferences (for simplicity)
user_preferences = {}

# Create a directed graph for the knowledge graph
knowledge_graph = nx.DiGraph()

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
        # Query Processing
        processed_query = process_query(user_query)
        
        # Retrieval Module
        documents, images = retrieve(processed_query)
        
        # Augmented Generation
        response = generate_response(processed_query, documents, images)
        
        # Anti-Hallucination
        if not validate_response(response):
            response = "I'm sorry, I couldn't find reliable information on that topic."
        
        # Personalization Engine
        update_preferences(user_query, response)
        
        # Knowledge Graph
        update_graph(user_query, response)
        
        return jsonify(response=response, images=images)
    
    except Exception as e:
        logging.error(f"Error handling query: {e}")
        return jsonify(error="An error occurred while processing your query."), 500

def process_query(query):
    try:
        doc = nlp(query)
        processed_query = {
            'tokens': [token.text for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentiment': doc.sentiment
        }
        return processed_query
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise

def retrieve(processed_query):
    try:
        search_query = " ".join(processed_query['tokens'])
        documents = [{"title": "Example Document", "content": "This is an example document relevant to the query."}]
        images = scrape_google_images(search_query)
        return documents, images
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        raise

def scrape_google_images(query):
    try:
        query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_urls = []
        for img in soup.find_all('img', {'src': True}):
            image_urls.append(img['src'])
            if len(image_urls) >= 5:  # Limit to 5 images for simplicity
                break
        return image_urls
    except Exception as e:
        logging.error(f"Error scraping images: {e}")
        raise

def generate_response(processed_query, documents, images):
    try:
        prompt = f"Query: {processed_query['tokens']}\nDocuments: {documents}\nImages: {images}\nGenerate a comprehensive response:"
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        raise

def update_preferences(query, response):
    try:
        user_preferences[query] = response
        with open('user_preferences.json', 'w') as f:
            json.dump(user_preferences, f)
    except Exception as e:
        logging.error(f"Error updating preferences: {e}")
        raise

def update_graph(query, response):
    try:
        knowledge_graph.add_node(query)
        knowledge_graph.add_edge(query, response)
        nx.write_gpickle(knowledge_graph, "knowledge_graph.gpickle")
    except Exception as e:
        logging.error(f"Error updating knowledge graph: {e}")
        raise

def validate_response(response):
    try:
        return True
    except Exception as e:
        logging.error(f"Error validating response: {e}")
        raise

if __name__ == '__main__':
    app.run(debug=True)
