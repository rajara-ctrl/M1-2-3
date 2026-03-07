import os
import json
from bs4 import BeautifulSoup as bs
from tokenizer import tokenize
import glob
import math
from collections import Counter

# This simply ignores the warning about parsing XML documents
# "XMLParsedAsHTMLWarning: It looks like you're using an HTML parser to parse an XML document."
# This is harmless and we can keep treating these files as HTML
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

#CONFIGURATION
DEV_DIR = os.path.join('developer', 'DEV') #Path to the data
PARTIAL_INDEX_DIR = 'partial_indexes' # Where we save the small index chunks to avoid running out of RAM
OFFLOAD_THRESHOLD = 15000 # How many documents to process before dumping memory to disk

def build_inverted_index():
    #Create the output folder if it doesn't exist
    if not os.path.exists(PARTIAL_INDEX_DIR):
        os.makedirs(PARTIAL_INDEX_DIR)

    inverted_index = {} # The main map
    # Structure: { "token": { doc_id_1: frequency, doc_id_2: frequency } }

    doc_map = {}  # Maps our integer IDs back to the real URLs
    
    doc_id = 0  # Counter for assigning unique IDs to documents
    partial_index_count = 1  # Counter for naming our partial files (index_1, index_2...)

    unique_tokens = set() # Set for tracking unique tokens
    total_index_size = int() # int for tracking total index size in bytes
    
    print(f"--- STARTING INDEXING from '{DEV_DIR}' ---") 

    # Walk through the developer folder and all of the JSON
    for root, dirs, files in os.walk(DEV_DIR):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    # Read the JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    url = data.get('url', '')
                    content = data.get('content', '')
                    
                    # Map the Document ID
                    doc_map[doc_id] = url
                    
                    # Parse HTML using bs
                    soup = bs(content, 'lxml')
                    text = soup.get_text()
                    
                    # Tokenize and stem, track unique tokens
                    tokens = tokenize(text)
                    #Add any unique tokens to tracker
                    unique_tokens.update(tokens)
                    
                    # Add to Index 
                    for token in tokens:
                        if token not in inverted_index:
                            inverted_index[token] = {}
                        
                        # Calculate Term Frequency 
                        if doc_id not in inverted_index[token]:
                            inverted_index[token][doc_id] = 1
                        else:
                            inverted_index[token][doc_id] += 1
                            
                    doc_id += 1

                    # Check progress every 1000 docs
                    if doc_id % 1000 == 0:
                        print(f"Processed {doc_id} documents...")

                    # Offload to disk
                    if doc_id % OFFLOAD_THRESHOLD == 0:
                        total_index_size += dump_partial_index(inverted_index, partial_index_count) #Update size tracker
                        print(f"Index size is {total_index_size} bytes...") #Displays total size for testing
                        print(f"Tracked {len(unique_tokens)} unique tokens...") #Displays amount of unique tokens for testing
                        inverted_index.clear() # Wipe memory
                        partial_index_count += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Dump any remaining data after the loop
    if inverted_index:
        total_index_size += dump_partial_index(inverted_index, partial_index_count)
    
    #  Save the Document Map (ID -> URL)
    with open("doc_map.json", "w") as f:
        json.dump(doc_map, f)
    
    # Convert bytes to kilobytes
    total_KB_size = round((total_index_size / 1000), 2)

    #Store stats in dict
    stats = {"Document Count":doc_id, 
    "Partial Indexes Count":partial_index_count,
    "Unique Tokens":len(unique_tokens),
    "Size in Bytes":total_index_size,
    "Size in KB":total_KB_size}

    STATS_FILE = 'stats_index.json' # Name of stats file

    #Unload indexing statistics into file
    print(f"   --> Offloading index stats to {STATS_FILE}...")
    with open(STATS_FILE, 'w', encoding='utf-8') as stats_file:
            json.dump(stats, stats_file, indent=2)

    print(f"\n---INDEXING COMPLETE---")

# Helper that saves the current dictionary to a JSON and returns total index size
def dump_partial_index(index_data, count):
    filename = os.path.join(PARTIAL_INDEX_DIR, f"index_{count}.json")
    print(f"   --> Offloading partial index to {filename}...")
    with open(filename, 'w') as f:
        json.dump(index_data, f)
    #Return current size of file in bytes
    return os.path.getsize(filename)

#Merges partial indexes into one dict and sorts postings list for each term
def mergeIndexes():
    """
    Merges all partial indexes into one, sorts the postings, and then 
    splits the final index into smaller alphabetical files
    """
    print("\n--- STARTING MERGE---")
    import string

    FINAL_INDEX_DIR = 'split_indexes'
    VOCAB_DIR = 'split_vocabs'
    # Create the folder for our final split files if it doesn't exist
    if not os.path.exists(FINAL_INDEX_DIR):
        os.makedirs(FINAL_INDEX_DIR)

     # Create the folder for our vocab files if it doesn't exist
    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)

    # Find all the partial index files we created during indexing
    files = sorted(glob.glob("partial_indexes/index_*.json"))
    merged = {}
    
    # Load and combine indexes
    print("Loading partial indexes into memory...")
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            index = json.load(f)
            
            # Combine the postings lists for each term
            for term, postings in index.items():
                if term in merged:
                    merged[term].update(postings)
                else:
                    merged[term] = postings

    # Sort postings
    # We must sort the document IDs as integers to allow for efficient 
    # linear-time O(x+y) intersection during search.
    print("Sorting postings lists...")
    for term in merged:
        merged[term] = dict(sorted(merged[term].items(), key=lambda x: int(x[0])))

    # Alphabetical splitting
    print("Splitting final index alphabetically and saving to disk...")
    
    # Create a set of valid starting characters (a-z and 0-9)
    valid_chars = set(string.ascii_lowercase + string.digits)
    
    # Initialize empty dictionaries for each letter/number, plus '_' for symbols
    split_data = {char: {} for char in valid_chars}
    split_data['_'] = {} 

    # Distribute every term into its corresponding bucket based on its first letter
    for term, postings in merged.items():
        if not term: continue
        first_char = term[0]
        
        # If it starts with a normal letter/number, put it in that bucket
        if first_char in split_data:
            split_data[first_char][term] = postings
        # Otherwise, throw it in the '_' bucket
        else:
            split_data['_'][term] = postings

    # Initialize empty bookmarking dictionaries for each split index
    vocab_dict = {char: {} for char in valid_chars}
    vocab_dict['_'] = {}

    STATS_FILE = 'stats_index.json'

    # Get total number of docs for idf calculation
    with open(STATS_FILE, 'r', encoding='utf-8') as stats_:
        stat = json.load(stats_) # Load stats as dict
        total_docs = stat["Document Count"] # Get total docs in index

    # Counter for tracking document vector lengths
    d_lengths = Counter()

    # Save split indexes to disk
    # Write each letter's dictionary to its own JSON file 
    for char, data in split_data.items():
        if data: # Only create the file if there are actually words in this bucket
            file_path = os.path.join(FINAL_INDEX_DIR, f"{char}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                for term, posting in sorted(data.items()):
                    position = f.tell() # Get byte position
    
                    df = len(posting) # Gets total number of documents that contain term
                    idf = math.log((total_docs / df), 10) # Calculate idf
                    term_stats = [position, df, idf] # List that holds term statistics
                    vocab_dict[char][term] = term_stats # Add term and byte position to vocabulary
                
                    for id, tf in posting.items(): # Calculate document vector length
                        weight = (1 + math.log(tf, 10)) * idf # Calculate document weight
                        posting[id] = weight # Change tf to tf-idf weight
                        d_lengths[int(id)] += weight**2 # Add to sum of doc weight squared
                    
                    json.dump({term:posting}, f) # Add term and updated postings list to split index file
                    f.write("\n")

    # Let user know vocabs are being saved
    print("Saving index vocabs to disk...")
    
    # Save vocabs to disk
    for char, dictionary in vocab_dict.items():
        if dictionary: # Only create file if vocab contains items
            file_path = os.path.join(VOCAB_DIR, f"vocab_{char}.json")
            with open (file_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_dict[char], f) # Write vocab dict to file
    
    # Let user know final document vector lengths are being calculated
    print("Calculating final document vector lengths...")

    #Calculate final document vector lengths
    for d_id, length in d_lengths.items():
        d_lengths[d_id] = math.sqrt(length)
    
    # Let user know final document vector lengths are being saved
    print("Saving final document vector lengths to disk...")
    
    DOC_LENGTH_FILE = 'doc_lengths.json'

    # Save lengths to file
    with open(DOC_LENGTH_FILE, 'w', encoding='utf-8') as d_file:
        json.dump(d_lengths, d_file)


    print(f"Saved indexes to '{FINAL_INDEX_DIR}' folder, " +
        f"Saved vocabs to '{VOCAB_DIR}' folder, " +
        f"Saved document vector lengths to '{DOC_LENGTH_FILE}' file" +
        f"\n--- MERGE COMPLETE ---")

if __name__ == "__main__":
    #build_inverted_index()
    mergeIndexes()