import os
import json
import time
import math
from tokenizer import tokenize
from collections import Counter

# --- CONFIGURATION ---
DOC_MAP_FILE = 'doc_map.json'
SPLIT_INDEX_DIR = 'split_indexes'
VOCAB_DIR = 'split_vocabs'
DOC_LENGTH_FILE = 'doc_lengths.json'

def load_doc_map():
    """
    Loads the Document-ID to URL mapping into memory
    """
    try:
        with open(DOC_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {DOC_MAP_FILE} missing. Run indexer.py first.")
        return {}

def search(query, doc_map):
    """
    Processes a user query, retrieves matching documents from the disk index,
    performs Boolean AND intersection, and prints the top 5 URLs
    """
    # Start the stopwatch to prove we meet the < 300ms requirement
    start_time = time.time()

    # Tell method to use global
    global total_docs
    
    # Query Processing
    # We must tokenize and stem the query using the exact same logic we used 
    # for the documents, otherwise the words won't match the index
    tokens = tokenize(query)
    
    if not tokens:
        print("Please enter a valid query.")
        return

    # Cache to reuse needed vocabs
    v_cache = {}

    # Counter to hold dot product for each doc
    dot_products = Counter()
    # Counter to store weight for each term in query
    q_weights = Counter()
    # Counter for query tf counts
    q_tf = Counter(tokens)
    # Sum of query term weights squared (for query vector length)
    sum_q_weights_squared = float()
    
    for token in tokens:
        # Determine which split file contains this token 
        first_char = token[0] if token[0].isalnum() else '_'
        file_path = os.path.join(SPLIT_INDEX_DIR, f"{first_char}.json")
        vocab_f_path = os.path.join(VOCAB_DIR, f"vocab_{first_char}.json")

         # If vocab not in cache, load vocab for containing term
        if(f"vocab_{first_char}" not in v_cache):
            with open(vocab_f_path, 'r', encoding='utf-8') as file:
                v_cache[f"vocab_{first_char}"] = json.load(file) # Add vocab dict to cache
        
        # Get current vocab dict needed for token
        terms = v_cache[f"vocab_{first_char}"]

        try:
            # Only load the specific letter file we need into memory
            with open(file_path, 'r', encoding='utf-8') as f:
                
                if token in terms:
                    # Calculate byte position of term
                    f.seek(terms[token][0])

                    # Get postings for that term
                    term_dict = json.loads(f.readline())
                    # Each posting is valid for scoring since it contains at least one query term
                    postings_list = term_dict[token]

                    idf = terms[token][2] # Get idf for term

                    # Calculate token weight in terms of query
                    query_freq = q_tf[token] # Frequency of term in query
                    qt_weight = (1 + math.log(query_freq, 10)) * idf # Calculate query term tf-idf weight
                    q_weights[token] += qt_weight # Inserts or updates weight for current query term
                    sum_q_weights_squared += qt_weight**2 # Update tracker for query vector length

                    # Calculate dot product for each document in postings dict
                    for id, d_weight in postings_list.items():
                        current_q_weight = q_weights[token] # Retrieve query weight
                        total_weight = d_weight * current_q_weight # Calculate dot product
                        dot_products[int(id)] += total_weight # Insert or update dot product for current doc

                else:
                    # If token is not found, move onto next token
                    continue
        except FileNotFoundError:
            print(f"0 results found. (Index file for '{first_char}' missing).")
            return
    
    # Get doc vector lengths
    with open(DOC_LENGTH_FILE, 'r', encoding='utf-8') as doc_file:
        d_lengths = json.load(doc_file) # Dict to hold doc vectors lengths

    # Dict to hold cosine score for each doc
    similarity_scores = dict()

    # Find square root of calculated squared query vector length
    q_length = math.sqrt(sum_q_weights_squared)

    # Calculate cosine similarity score for each dot product
    for id, dot in dot_products.items():
        d_length = d_lengths[str(id)] # Get current doc length
        normalization = (d_length * q_length) # Calculate normalization of vectors
        similarity_scores[int(id)] = (dot / normalization) # Cosine(q,d) = Dot product / |d|*|q|

    # Stop the stopwatch and calculate milliseconds
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    # Formatting & Output
    print(f"\n--- Search Results ---")
    print(f"Query: '{query}'")
    print(f"Found {len(similarity_scores)} valid documents in {elapsed_ms:.2f} ms")
    
    #if not result_list:
        #print("No documents contained ALL query terms.")
        #return

    #print(f"\nTop 5 URLs:")
    # Loop through the first 5 document IDs and look up their real URLs
    #for i, doc_id in enumerate(result_list[:5]):
        #url = doc_map.get(str(doc_id), "URL not found")
        #print(f"{i + 1}. {url}")
    #print("-" * 35)

# Main ui
if __name__ == "__main__":
    print("Loading Document Map into memory...")
    doc_map = load_doc_map()
    
    if doc_map:
        print("\nSearch Engine Ready (Type 'quit' to exit)")
        while True:
            user_query = input("\nEnter search query: ")
            if user_query.lower() == 'quit':
                break
            search(user_query, doc_map)