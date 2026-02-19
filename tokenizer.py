import re
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

# Parses a string into a list of stemmed tokens
def tokenize(text):
    tokens = []
    
    # Lowercase and find all alphanumeric characters (a-z, 0-9)
    # This regex automatically ignores punctuation like "!" or ","
    raw_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    # Apply Stemming
    for token in raw_tokens:
        stemmed = stemmer.stem(token)
        tokens.append(stemmed)
        
    return tokens