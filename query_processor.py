from tokenizer import tokenize
from indexer import *
from math import log

def booleanSearchForQuery():
    #Merge all partial indexes together
    index = mergeIndexes()
    #Get query and tokenize it
    query = tokenizeQuery()
    #Make dict to store all query token postings
    token_dict = dict()
    #Add every token posting_list to dict
    for term in query:
        token_dict.update({term: index[term]})

    #Make merged postings list for all terms
    merged_postings = list(token_dict.values())
    #Store postings of first posting list in a set
    intersected_postings = set(merged_postings[0].keys())
    #Loop through postings lists in merged list and intersect them
    for postings_list in merged_postings[1:]:
        intersected_postings = intersected_postings.intersection(postings_list.keys())
    
    #Calculate tf-idf scores for intersected docs
    scores = calculateScore(token_dict, intersected_postings)
    #Return list of docs only 
    return scores.keys()

def tokenizeQuery():
    #Get user query
    query = getQuery()
    #Seperate and simplify query terms
    query_tokens = tokenize(query)
    #Return tokenized query
    return query_tokens

def getQuery():
    #Get user input
    query = input("What would you like to search for? ")
    #Let user know results are on the way
    print("Searching for results...")
    return query

#Takes query tokens dict and set of docs that contain all query terms as arguements
def calculateScore(query, intersected):
    #Calculate tf-idf score for each posting in term posting's list
    n = doc_id
    #Dict to store idf scores for each
    idf = dict()
    #Dict to store tf-idf scores for each doc
    scores = dict()
    #Calculate the idf for each term in query
    for term in query:
        df = len(query[term]) #total number of docs per term
        idf.update({term: log((55301 / df))})
    #Calculate tf-idf score for each posting that contains all terms
    for posting in intersected:
        score = 0
        for term in query:
            score += query[term][posting] * idf[term] #Multiply tf * idf for each term
        scores.update({posting: score}) #Add posting with its score
    
    #Sort docs descending by tf-idf scores
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))