import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from stellargraph import StellarGraph


# Load the vocabulary dictionary
with open('vocabulary100.pkl', 'rb') as vocab_file100:
    vocabulary100 = pickle.load(vocab_file100)

# Initialize structures for StellarGraph
node_features = {}  # Dictionary to store cumulative node embeddings
edge_weights = {}   # Dictionary to store cumulative edge weights
word_to_idx = {}    # Mapping words to their node indices
current_index = 0

# Loop over each year
for year in range(1985, 2025):
    print(f"Processing year {year}...")

    # Load the embedding matrix and tokenized articles for the current year
    embedding_matrix = np.load(f"embedding_matrix100_{year}.npy")
    with open(f"tokenized_articles_{year}.pkl", 'rb') as token_file:
        tokenized_articles = pickle.load(token_file)
    
    # Get the vocabulary for the current year and map words to their indices
    vocabulary_year = vocabulary100[year]
    word_to_embedding_idx = {word: idx for idx, word in enumerate(vocabulary_year)}

    # Update node features for each word in the vocabulary
    for word in vocabulary_year:
        if word not in word_to_idx:
            # If word is new, assign an index and add its embedding
            word_to_idx[word] = current_index
            current_index += 1
            embedding_idx = word_to_embedding_idx[word]
            node_features[word] = embedding_matrix[embedding_idx]
        else:
            # Update embedding for existing word node
            embedding_idx = word_to_embedding_idx[word]
            node_features[word] = embedding_matrix[embedding_idx]

    # Build edges based on co-occurrence in articles
    for tokens in tokenized_articles:
        unique_tokens = set(tokens)  # Remove duplicate words in the same article
        for word1, word2 in combinations(unique_tokens, 2):
            word1, word2 = sorted((word1, word2))  # Ensure unique (word1, word2) pair ordering
            if word1 != word2:  # Ignore self-loops
                if (word1, word2) in edge_weights:
                    edge_weights[(word1, word2)] += 1  # Increment weight if edge exists
                else:
                    edge_weights[(word1, word2)] = 1  # Initialize edge with weight 1

    # Convert edges to list format for StellarGraph
    edges = [(word1, word2, weight) for (word1, word2), weight in edge_weights.items()]

    # Create DataFrames for nodes and edges
    nodes_df = pd.DataFrame.from_dict(node_features, orient="index")
    nodes_df.columns = [f"dim_{i}" for i in range(nodes_df.shape[1])]  # Rename columns as embedding dimensions
    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])

    # Create StellarGraph
    graph = StellarGraph(nodes=nodes_df, edges=edges_df, edge_weight_column="weight", node_type_default="word", edge_type_default="co-occurrence")

    # Save the StellarGraph for the current year
    with open(f"stellar_graph100_{year}.pkl", 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Graph for year {year} saved with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

# Save node vocabulary by year
with open('node_vocabulary_stellargraph100.pkl', 'wb') as vocab_file100:
    pickle.dump(word_to_idx, vocab_file100)

print("All cumulative yearly StellarGraph graphs created and saved.")
