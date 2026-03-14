# Import Flask tools... turn Python dictionaries into JSON data for the web
# pip install Flask for this
from flask import Flask, render_template, request, jsonify

from search import search, load_doc_map

# Initialize the Flask application
app = Flask(__name__)

# Load the doc map into memory when the server starts
print("Loading Document Map for Web Server...")
doc_map = load_doc_map()

# Route 1: The Homepage (localhost:5000)
@app.route('/')
def home():
    # render_template automatically looks inside a folder named 'templates' for this file
    return render_template('index.html')

# The API endpoint 
@app.route('/api/search', methods=['GET'])
def do_search():
    # request.args.get grabs the 'q' parameter from the URL (for example, ?q=machine+learning)
    query = request.args.get('q', '')
    
    # If the user submitted an empty search bar, return empty data
    if not query:
        return jsonify({"results": [], "time": 0, "count": 0})
    
    # Run the search engine logic
    data = search(query, doc_map)
    
    # if search() returns None, send empty data to prevent a crash
    if not data:
        data = {"results": [], "time": 0, "count": 0}
        
    # Send the Python dictionary back to the browser as a JSON object
    return jsonify(data)

# This block starts the local web server when running python app.py
if __name__ == '__main__':
    # debug=True allows the server to auto-update if you change the code
    app.run(debug=True, port=5000)