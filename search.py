import os
import glob
import json
import time
import math
import re
import heapq
from tokenizer import tokenize
from collections import Counter

# --- CONFIGURATION ---
DOC_MAP_FILE = 'doc_urls.json' # Fixed to match the document map update
SPLIT_INDEX_DIR = 'split_indexes'
VOCAB_DIR = 'split_vocabs'
DOC_LENGTH_FILE = 'doc_lengths.json'
DOC_CHAMPION_LISTS_FILE = 'doc_champion_lists.json'
DOC_ND_FILE = 'doc_near_duplicates.json'

# LOAD DATA INTO MEMORY
# Get champions list for all terms
with open(DOC_CHAMPION_LISTS_FILE, 'r', encoding='utf-8') as f:
    champion_dict = json.load(f) # Load champion list dict into memory

# Get doc vector lengths
with open(DOC_LENGTH_FILE, 'r', encoding='utf-8') as doc_file:
    d_lengths = json.load(doc_file) # Dict to hold doc vectors lengths

# Get near duplicate lists
with open(DOC_ND_FILE, 'r', encoding='utf-8') as nd_file:
    doc_nd = json.load(nd_file)

# Load vocabs into memory
vocabs = {} # Dict to hold vocabs

files = sorted(glob.glob(f"{VOCAB_DIR}/vocab_*.json")) # Get every vocab file name

for file in files:
    with open(file, 'r', encoding='utf-8') as vocab_file:
        # os.path.basename grabs just "vocab_m.json" for os independence
        # Then split by the period to just keep "vocab_m"
        clean_name = os.path.basename(file).split('.')[0]
        vocabs[clean_name] = json.load(vocab_file) # Add vocab to dict

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
    performs ranked retrieval, and prints the top 5 URLs
    """
    # Start the stopwatch to prove we meet the < 300ms requirement
    start_time = time.time()
    
    # Query Processing
    # We must tokenize and stem the query using the exact same logic we used 
    # for the documents, otherwise the words won't match the index
    tokens = tokenize(query)
    
    if not tokens:
        return {"results": [], "time": 0.0, "count": 0} # No more terminal printing

    # Counter to hold dot product for each doc
    dot_products = Counter()
    # Counter to store weight for each term in query
    q_weights = Counter()
    # Counter for query tf counts
    q_tf = Counter(tokens)
    # Sum of query term weights squared (for query vector length)
    sum_q_weights_squared = float()
    # Sum of query term weights (for threshold calculation)
    weight_threshold = float()
    # List to track valid query tokens
    valids = []
    
    # INDEX ELIMINATION
    # Find valid tokens and average weight of them
    for token in tokens:
        # Determine which vocab file would contain this token 
        first_char = token[0] if token[0].isalnum() else '_'
        
        # Get current vocab dict needed for token
        terms = vocabs[f"vocab_{first_char}"]

        # Check if term is in vocab (if not, term is not valid)
        if token in terms:
            valids.append(token) # Marks token as valid
            weight_threshold += terms[token][2] # Adds token weight to tracker

    # Calculate average token tf-idf weight for threshold
    if len(valids) != 0: weight_threshold /= len(valids)

    # Make list of query terms with high weights
    high_weights_list = []

    # Filter out query terms that do not meet threshold
    for term in valids:
        first_char = term[0] if term[0].isalnum() else '_'
        terms = vocabs[f"vocab_{first_char}"]
        if terms[term][2] >= weight_threshold: # Add term to list if it meets threshold
            high_weights_list.append(term)

    if len(high_weights_list) >= 4: # Make sure there are enough terms for effective search
        valids = high_weights_list

    # PROCESSING VALID TOKENS
    for token in valids:
        # Determine first char of token
        first_char = token[0] if token[0].isalnum() else '_'
        
        # Use vocab containg token
        terms = vocabs[f"vocab_{first_char}"]

        #CAN USE THIS TO PRODUCE MORE RESULTS IF NEEDED (currently only use champion lists)
        # Only load the specific letter file we need into memory
        #file_path = os.path.join(SPLIT_INDEX_DIR, f"{first_char}.json")
        # Calculate byte position of term
        #f.seek(terms[token][0])

        # Get postings for that term
        #term_dict = json.loads(f.readline())
        # Each posting is valid for scoring since it contains at least one query term
        #postings_list = term_dict[token]

        champion_list = champion_dict[token] # Load current token's champion list

        if len(valids) > 1: # Multiple valid tokens in query

            idf = terms[token][2] # Get idf for term

            # CALCULATE W_QT
            query_freq = q_tf[token] # Frequency of term in query
            qt_weight = (1 + math.log(query_freq, 10)) * idf # Calculate query term tf-idf weight
            q_weights[token] += qt_weight # Inserts or updates weight for current query term
            sum_q_weights_squared += qt_weight**2 # Update tracker for query vector length

            # CALCULATE DOT PRODUCT for each document in postings dict
            for d_weight, id in champion_list:
                total_weight = d_weight * qt_weight # Calculate dot product
                dot_products[int(id)] += total_weight # Insert or update dot product for current doc

        elif len(valids) == 1: # Only one valid token in query
            break

        else: # If no tokens are valid
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            return {"results": [], "time": round(elapsed_ms, 2), "count": 0}  # No more terminal printing
    
     # Helpers for near duplicate elimination
    results = []
    traversed = set() # Holds docs to skip like near duplicates

    if(len(valids) > 1):
        # Find square root of calculated squared query vector length
        q_length = math.sqrt(sum_q_weights_squared)
        
        # CALCULATE COSINE SIMILARITY SCORE for each dot product
        for id, dot in dot_products.items():
            if id in traversed: # Skips doc if already calculated
                continue

            d_length = d_lengths[str(id)] # Get current doc length
            normalization = (d_length * q_length) # Calculate normalization of vectors
            score = (dot / normalization) # Cosine(q,d) = Dot product / |d|*|q|

            if len(results) < 20: # Add to results heap if not full
                heapq.heappush(results, (score, id))
            else: # Add if greater than min score
                heapq.heappushpop(results, (score, id))

            # Mark duplicates
            if id in doc_nd:
                traversed.update(doc_nd[id])

            traversed.add(id) # Add id to traversed set

    # If only one term, can skip query vectory length calculations
    elif(len(valids) == 1):
        # Get term
        term = valids[0]
        # Get champions list
        c_list = champion_dict[term]
        # Skip near duplicates
        for weight_, d_id in c_list:
            if d_id in traversed: # Skips doc if already calculated
                continue

            # Calculate score for each doc in postings list
            d_length = d_lengths[str(d_id)] # Get current doc length
            score = weight_ / d_length # # Cosine(q,d) = d_weight / |d|

            # Push champions list
            heapq.heappush(results, (score, d_id))

            # Mark duplicates
            if d_id in doc_nd:
                traversed.update(doc_nd[d_id])

            traversed.add(d_id) # Add id to traversed set

    # Sorts results heap by highest similarity score
    results = sorted(results, key=lambda x: x[0],reverse=True)


    #Deleted the print results since the output will no longer be terminal-based

    # Stop the stopwatch and calculate milliseconds
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

<<<<<<< HEAD
    # Since a regular html file is not allowed to run scripts or read the JSON off the hard drive, 
    # the flask engine links the scripts with the webpage
    # ----------------------- FLASK OUTPUT---------------------------
    # Instead of printing to the terminal, the results are packaged into a dictionary
    # Flask automatically converts this dictionary into a JSON format to send to the webpage

    # If the search yielded zero valid documents, return an empty package
    if not results:
        return {"results": [], "time": round(elapsed_ms, 2), "count": 0}

    final_results = []
    
    # Loop through the top 5 document IDs (already sorted by Cosine Similarity)
    for i, pair in enumerate(results[:5]):
        # pair[0] is the DocID, pair[1] is the Cosine Score.
        # We look up the real URL using the DocID from our doc_map
        url = doc_map.get(str(pair[0]), "URL not found")
        
        # Add this specific result (URL and Score) to our list
        final_results.append({"url": url, "score": round(pair[1], 4)})
        
    # Return the final package of data back to the Flask server
    return {"results": final_results, "time": round(elapsed_ms, 2), "count": len(similarity_scores)}
=======
    # Formatting & Output
    print(f"\n--- Search Results ---")
    print(f"Query: '{query}'")
    print(f"Found {len(results)} valid documents in {elapsed_ms:.2f} ms")
    
    if not results:
        print("No documents contained query terms.")
        return

    print(f"\nTop {len(results)} URLs:")
    # Loop through the first N document IDs and look up their real URLs
    for i, pair in enumerate(results[:20]):
        url = doc_map.get(str(pair[1]), "URL not found")
        print(f"{i + 1}. {url}")
    print("-" * 35)
>>>>>>> c9ad354a67220e9fa5caf2c680ea3b012f4c98c4

# Main ui
# Flask is now running as the main program, so we no longer need this
"""
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
"""