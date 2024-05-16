import tensorflow as tf
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import spacy 
import random 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
stop_words = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")
def extract_entities(question):
    question = question.replace("-", " ")
    doc = nlp(question)
    entities = {}
    rules = []
    operations = []
    dsa = []

    custom_rules = {
        "purpose": ["purpose"],
        "implementation": ["implement", "implementation", "implemented", "how do", "how to", "perform", "process"],
        "description": ["what", "explain", "describe", "state", "enumerate", "elaborate", "description", "define"],
        "complexity": ["time complexity"],
        "applications": ["applications", "real-world applications", "practical applications", "use cases", "use", "used", "uses"],
        "differences": ["difference", "differences", "differ", "compare", "contrast"],
        "advantages": ["advantages", "benefits"],
        "types": ["types", "kinds", "type"],
        "algorithm": ["algorithm", "algorithms"],
        "examples": ["example", "examples", "sample"],
        "property": ["property", "properties"]
    }

    custom_operations = {
        "detect": ["detect"],
        "find": ["find"],
        "compute": ["compute"],
        "handle": ["handle"],
        "reverse": ["reverse", "reversal"],
        "insert": ["insert", "add", "insertion", "addition", "new"],
        "delete": ["deletion", "delete", "remove", "removal", "removed", "deleting"],
        "access": ["access", "get"],
        "end": ["end", "last", "append"],
        "beginning": ["beginning", "first", "front"],
        "push" : ["push"],
        "pop" : ["pop"],
        "enqueue" : ["enqueue"],
        "dequeue" : ["dequeue"],
        "union" : ["union"]
    }

    for rule_entity, keywords in custom_rules.items():
        for keyword in keywords:
            if keyword in question.lower():
                rules.append(rule_entity)
                break
    if not rules:
        rules.append("description")

    for rule_entity, keywords in custom_operations.items():
        for keyword in keywords:
            if keyword in question.lower():
                operations.append(rule_entity)
                break

    data_structures = ["array", "linked list", "doubly linked list", "singly linked list", "circular linked list", "stack", "heap", "fibonacci heap", "graph", "tree", "binary tree", "balanced binary tree", "min heap", "max heap", "bubble sort", "binary search", "cycle", "inorder traversal", "preorder traversal", "post order traversal", "postorder traversal", "level order traversal", "pre order traversal", "in order traversal", "prefix", "postfix", "breadth first search", "bfs", "dfs", "depth first search","binary search tree", "queue", "priority queue", "hash table", "trie", "doubly linked list", "suffix array", "B tree", "k d tree", "disjoint set", "avl tree", "complete binary tree", "two pointer", "full binary tree", "perfect binary tree", "balanced binary tree", "undirected graph", "weighted graph", "connected graph", "disconnected graph", "cyclic graph", "acyclic graph", "hamiltonian cycle", "eulerian cycle", "kanpsack problem", "prefix sum", "traveling salesman problem", "suffix sum", "expression tree", "sliding window", "two pointer", "longest common subsequence", "longest increasing subsequence", "segment tree", "fenwick tree", "bloom filter", "skip list", "treap", "greedy", "knut morris pratt",  "bit manipulation", "quicksort", "mergesort", "lazy propagation", "heap sort", "heapsort", "radix sort", "radixsort", "counting sort", "longest common subsequences", "treap", "quick sort", "merge sort", "selection sort", "insertion sort", "shell sort", "sorting", "searching", "recursion", "backtracking", "dynamic programming", "hashing", "topological sort", "adjacency matrix", "directed graph", "linear serach", "graph traversal", "tree traversal", "spanning tree", "minimum spanning tree", "spaning tree", "minimum spaning tree", "dijkstra", "floyd warshall", "kruskal", "hashing", "hash", "hash function", "set", "subset", "subset sum", "knapsack", "bellman ford", "heapify", "hash collision", "collision", "a* search", "a*", "self balancing tree", "adjacency list", "edmonds karp", "collision resolution", "floyd cycle", "quick select", "tarjan", "floyd warshall matrix", "johnson", "prim", "kosaraju", "load factor", "vitebri", "ford fulkerson", "probing", "hopcraft karp", "rehashing", "linear probing", "dinic", "quadratic probing", "karger", "double hashing", "cuckoo hashing", "hungarian", "bellman ford moore", "seperate chaining", "divide and conquer", "robinhood hashing", "robin hood hashing",  "boyer moore", "karp rabin", "karp miller roesnberg", "hopscotch hashing", "aho corasick", "rabin karp", "karp miller", "smith waterman", "mccreight", "time complexity", "space complexity", "big o notation", "big o", "divide and conquer", "memoization", "circular queue", "red black", "suffix tree", "linear search", "deque", "hash map", "digraph",  "chaining", "open addressing", "matrix", "binary heap", "dynamic array", ]
    for data_structure in data_structures:
        if data_structure in question.lower():
            dsa.append(data_structure)
    entities["rules"] = rules
    entities["operations"] = operations
    entities["dsa"] = dsa
    return entities
# Load the saved model
model = tf.keras.models.load_model('chatbot_model_real.h5')

# Load the tokenizer from the file
with open('tokenizer_real.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_len_real.pkl', 'rb') as f:
    max_len = pickle.load(f)

def generate_response(model, tokenizer, user_query, max_len):
    # Tokenize and preprocess the user query
    user_query_sequence = tokenizer.texts_to_sequences([user_query])
    user_query_padded = pad_sequences(user_query_sequence, maxlen=max_len)

    # Generate response using the trained model
    response_sequence = model.predict(user_query_padded)
    response_sequence = np.argmax(response_sequence, axis=-1)

    # Decode the model's output to get the text response
    response_text = decode_response(response_sequence, tokenizer)

    # Perform post-processing
    response_text = post_process_response(response_text)

    return response_text

def decode_response(response_sequence, tokenizer):
    # Initialize an empty list to store the decoded tokens
    decoded_tokens = []

    # Iterate over each token index in the response sequence
    for token_index in response_sequence[0]:
        # Convert the token index to its corresponding word
        word = tokenizer.index_word.get(int(token_index), '')  # Cast token_index to int

        # Append the word to the list of decoded tokens
        decoded_tokens.append(word)

    # Join the decoded tokens into a single string
    response_text = ' '.join(decoded_tokens)

    return response_text

def post_process_response(response_text):
    # Remove repeated words
    words = response_text.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    response_text = ' '.join(unique_words)

    # Capitalize the first letter

    response_text = response_text.capitalize()

    # Remove leading and trailing whitespace
    response_text = response_text.strip()

    return response_text

def generate_response_handler(user_query):
    entities = extract_entities(user_query)
    altered_question = " ".join(entities["rules"]) + " ".join(entities["operations"]) + " " + " ".join(entities["dsa"])
    response = generate_response(model, tokenizer, altered_question, max_len)
    return response

def generate_hint(answer, correct_answer):
    remove_words = ["use", "used", "the", "of", "various", "user"]

    correct_answer = re.sub(r'[^\w\s]', '', correct_answer)

    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words and token not in remove_words]
        return tokens

    tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = tfidf_vectorizer.fit_transform([correct_answer])

    feature_names = tfidf_vectorizer.get_feature_names_out()
    keyword_indices = tfidf_matrix.sum(axis=0).A1.argsort()[-4:]
    keywords = [feature_names[idx] for idx in keyword_indices]
    keywords = [hint for hint in keywords if hint.lower() not in answer.lower()]

    print(correct_answer, answer, keywords)
    if not keywords:
        return "Correct answer"
    
    hint_sentences = ["You may want to reconsider with ", "It seems like you've missed out on ", "Have you considered ", "Consider thinking of ", "Quite wrong! think of ", "Explore on "]

    return random.choice(hint_sentences) + random.choice(keywords)