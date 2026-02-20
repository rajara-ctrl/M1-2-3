import os
import json
from bs4 import BeautifulSoup as bs
from tokenizer import tokenize

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
                    tokens = tokenize(text, unique_tokens)
                    
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
                    #Check amount of unique tokens tracked every 10000 docs
                    if doc_id % 10000 == 0:
                        print(f"Tracked {len(unique_tokens)} unique tokens...")

                    # Offload to disk
                    if doc_id % OFFLOAD_THRESHOLD == 0:
                        dump_partial_index(inverted_index, partial_index_count)
                        inverted_index.clear() # Wipe memory
                        partial_index_count += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Dump any remaining data after the loop
    if inverted_index:
        dump_partial_index(inverted_index, partial_index_count)
    
    #  Save the Document Map (ID -> URL)
    with open("doc_map.json", "w") as f:
        json.dump(doc_map, f)

    print(f"\n---INDEXING COMPLETE---")
    print(f"Total Documents Indexed: {doc_id}")
    print(f"Partial Indexes Created: {partial_index_count}")
    print(f"Total Unique Tokens: {len(unique_tokens)}")

# Helper that saves the current dictionary to a JSON
def dump_partial_index(index_data, count):
    filename = os.path.join(PARTIAL_INDEX_DIR, f"index_{count}.json")
    print(f"   --> Offloading partial index to {filename}...")
    with open(filename, 'w') as f:
        json.dump(index_data, f)

if __name__ == "__main__":
    build_inverted_index()