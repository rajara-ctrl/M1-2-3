import os
import json
from bs4 import BeautifulSoup as bs
from tokenizer import tokenize
import glob

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

    unique_tokens = set() #Set for tracking unique tokens
    total_index_size = int() #int for tracking total index size in bytes
    
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
    
    #Convert bytes to kilobytes
    total_KB_size = total_index_size / 1000

    print(f"\n---INDEXING COMPLETE---")
    print(f"Total Documents Indexed: {doc_id}")
    print(f"Partial Indexes Created: {partial_index_count}")
    print(f"Total Unique Tokens: {len(unique_tokens)}")
    print(f"Total Index Size in Bytes: {total_index_size}")
    print(f"Total Index Size in Kilobytes: {total_KB_size:.2f}")

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
    # Create the folder for our final split files if it doesn't exist
    if not os.path.exists(FINAL_INDEX_DIR):
        os.makedirs(FINAL_INDEX_DIR)

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

    # Save to disk
    # Write each letter's dictionary to its own JSON file 
    for char, data in split_data.items():
        if data: # Only create the file if there are actually words in this bucket
            file_path = os.path.join(FINAL_INDEX_DIR, f"{char}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)

    print(f"--- MERGE COMPLETE, Saved to '{FINAL_INDEX_DIR}' folder ---")

if __name__ == "__main__":
    #build_inverted_index()
    mergeIndexes()
