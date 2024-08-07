# File: nlp_processing.py
import spacy

# Load a pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

def process_query(query):
    # Use NLP techniques to preprocess and understand the query
    doc = nlp(query)
    processed_query = {
        'tokens': [token.text for token in doc],
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'sentiment': doc.sentiment
    }
    return processed_query
