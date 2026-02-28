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

#Global tracker for number of docs
doc_id = int()

def build_inverted_index():
    #Let function know to use global variable
    global doc_id
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
    #Display start message
    print("Merging indexes...")
    #Get name of all files in a list
    files = sorted(glob.glob("partial_indexes/index_*.json"))
    #Dictionary to hold combined postings
    merged = {}
    #Go through every partial index file
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            #Load index into a dictionary in memory
            index = json.load(f)
            #For each term in index, add posting to merged dictionary
            for term, postings in index.items():
                #If term was in a previous index
                if term in merged:
                    #Add postings to end of postings list
                    merged[term].update(postings)
                else:
                    #Add new term with postings list
                    merged.update({term: postings})
    #Sort postings list for each term
    for term in merged:
        #Convert each doc_id to int for proper sorting logic
        merged[term] = dict(sorted(merged[term].items(), key=lambda x: int(x[0])))
    #Let user know merge is done
    print("Merge is done...")
    #Return sorted merged dict
    return merged
