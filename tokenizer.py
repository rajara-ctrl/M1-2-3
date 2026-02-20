import re
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

# Parses a string into a list of stemmed tokens and updates set of unique tokens
def tokenize(text, unique_tokens):
    tokens = []
    
    # Lowercase and find all alphanumeric characters (a-z, 0-9)
    # This regex automatically ignores punctuation like "!" or ","
    raw_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    # Apply Stemming
    for token in raw_tokens:
        stemmed = stemmer.stem(token)
        tokens.append(stemmed)

    #Add any unique tokens to tracker
    unique_tokens.update(tokens)
    #Return tokens list
    return tokens