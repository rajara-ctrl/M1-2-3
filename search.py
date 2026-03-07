import os
import json
import time
from tokenizer import tokenize

# --- CONFIGURATION ---
DOC_MAP_FILE = 'doc_map.json'
SPLIT_INDEX_DIR = 'split_indexes'
VOCAB_DIR = 'split_vocabs'

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
    
    # Query Processing
    # We must tokenize and stem the query using the exact same logic we used 
    # for the documents, otherwise the words won't match the index
    tokens = tokenize(query)
    
    if not tokens:
        print("Please enter a valid query.")
        return

    # Disk Retrieval
    # We will store the sets of Document IDs for each query term here
    postings_lists = []

    # Create a cache to reuse needed vocabs
    v_cache = {}
    
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
                    f.seek(terms[token])
                    term_dict = json.loads(f.readline())
                    postings_lists.append(term_dict[token]) # Add term postings list to total postings lists
                else:
                    # If token is not found, move onto next token
                    continue
        except FileNotFoundError:
            print(f"0 results found. (Index file for '{first_char}' missing).")
            return

    #Query Optimization & Boolean AND Intersection
    # Optimization: Sort the lists by size (smallest to largest).
    # Intersecting the smallest sets first eliminates the most candidates immediately,
    # drastically speeding up the query
    postings_lists.sort(key=len)
    
    # Start with the smallest set of documents
    result_set = postings_lists[0]
    
    # Perform the Boolean AND by intersecting with the remaining sets
    for s in postings_lists[1:]:
        result_set = result_set.intersection(s)

    # Stop the stopwatch and calculate milliseconds
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    # Formatting & Output
    result_list = list(result_set)
    print(f"\n--- Search Results ---")
    print(f"Query: '{query}'")
    print(f"Found {len(result_list)} valid documents in {elapsed_ms:.2f} ms")
    
    if not result_list:
        print("No documents contained ALL query terms.")
        return

    print(f"\nTop 5 URLs:")
    # Loop through the first 5 document IDs and look up their real URLs
    for i, doc_id in enumerate(result_list[:5]):
        url = doc_map.get(str(doc_id), "URL not found")
        print(f"{i + 1}. {url}")
    print("-" * 35)

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