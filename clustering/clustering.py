from sklearn.cluster import KMeans
import pickle
from munkres import Munkres
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns



# clustering
# Load the datasets
with open('embedding_matrices100.pkl', 'rb') as f100: 
    embedding_matrices100 = pickle.load(f100)

with open('vocabulary100.pkl', 'rb') as vocab_file100:
    vocabulary100 = pickle.load(vocab_file100)


# Dictionary to store clusters for each year
yearly_clusters = {}

# Dictionary to store centroids for each year
yearly_centroids = {}

# Perform clustering for years 1987 to 2023
for year in range(1987, 2024):
    # Get the embeddings and vocabulary for the year
    word_embeddings = embedding_matrices100[year]  # Shape: (num_words, embedding_dim)
    words = vocabulary100[year]                   # List of words for the year
    
    # Perform clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(word_embeddings)
    
    # Save centroids
    yearly_centroids[year] = kmeans.cluster_centers_
    
    # Create a dictionary to store words in each cluster
    clusters = {i: [] for i in range(10)}  # Initialize 10 empty clusters
    for word, cluster_label in zip(words, cluster_labels):
        clusters[cluster_label].append(word)
    
    # Save clusters for the year
    yearly_clusters[year] = clusters

# Perform clustering for the year 2024
# Use actual embeddings and words in testset
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels_2024 = kmeans.fit_predict(actual_embeddings_2024)

# Save centroids for 2024
yearly_centroids[2024] = kmeans.cluster_centers_

# Create a dictionary to store words in each cluster for 2024
clusters_2024 = {i: [] for i in range(10)}  # Initialize 10 empty clusters
for word, cluster_label in zip(words2024, cluster_labels_2024):
    clusters_2024[cluster_label].append(word)

# Save clusters for the year 2024
yearly_clusters[2024] = clusters_2024

# Save yearly clusters and centroids to pickle files
file_path = "..."

with open(f'{file_path}/yearly_clusters.pkl', 'wb') as clusters_file:
    pickle.dump(yearly_clusters, clusters_file)

with open(f'{file_path}/yearly_centroids.pkl', 'wb') as centroids_file:
    pickle.dump(yearly_centroids, centroids_file)


# mapping 
# Initialize Munkres instance
munkres = Munkres()

# Initialize cluster name mappings
cluster_names = {}
cluster_names[1987] = {i: f"Cluster_{i}" for i in range(10)}  # Initial names for 1990

# Dictionary to store mappings across years
yearly_cluster_mapping = {}

# Iterate through each year to map clusters to the previous year
for year in range(1988, 2025):
    current_centroids = yearly_centroids[year]
    previous_centroids = yearly_centroids[year - 1]

    current_clusters = yearly_clusters[year]
    previous_clusters = yearly_clusters[year - 1]

    # Compute cosine similarity matrix (centroid similarity)
    centroid_similarity = cosine_similarity(current_centroids, previous_centroids)

    # Compute Jaccard similarity matrix (word overlap)
    word_overlap_similarity = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            words_1 = set(current_clusters[i])
            words_2 = set(previous_clusters[j])
            if len(words_1 | words_2) > 0:  # Avoid division by zero
                word_overlap_similarity[i, j] = len(words_1 & words_2) / len(words_1 | words_2)

    # Combine similarities (weighted sum)
    combined_similarity = 0.7 * centroid_similarity + 0.3 * word_overlap_similarity

    # Convert similarity to a cost matrix (negate values for Munkres)
    cost_matrix = -combined_similarity

    # Compute optimal mapping using Munkres
    indices = munkres.compute(cost_matrix)  # List of (current_cluster, previous_cluster) pairs

    # Map clusters and save the mapping
    cluster_mapping = {}
    for current_cluster, previous_cluster in indices:
        cluster_mapping[current_cluster] = previous_cluster
    yearly_cluster_mapping[year] = cluster_mapping

    # Assign consistent cluster names based on the mapping
    cluster_names[year] = {}
    for current_cluster, previous_cluster in cluster_mapping.items():
        cluster_names[year][current_cluster] = cluster_names[year - 1][previous_cluster]

# Save cluster mappings and consistent names to files
with open(f'{file_path}/yearly_cluster_mapping.pkl', 'wb') as f:
    pickle.dump(yearly_cluster_mapping, f)

with open(f'{file_path}/consistent_cluster_names.pkl', 'wb') as f:
    pickle.dump(cluster_names, f)


# Dictionary to store updated clusters with consistent names
updated_yearly_clusters = {}

# Iterate through each year to rename clusters
for year in range(1987, 2025):
    # Get current year's clusters and cluster names
    clusters = yearly_clusters[year]
    cluster_name_mapping = cluster_names[year]  # Consistent cluster names for the year

    # Rename clusters based on consistent names
    renamed_clusters = {}
    for cluster_id, words in clusters.items():
        renamed_cluster_name = cluster_name_mapping[cluster_id]
        renamed_clusters[renamed_cluster_name] = words

    # Save renamed clusters for the year
    updated_yearly_clusters[year] = renamed_clusters

# Save updated yearly clusters to a pickle file
with open(f'{file_path}/updated_yearly_clusters.pkl', 'wb') as f:
    pickle.dump(updated_yearly_clusters, f)





# compute relationship between clusters
# Load the required data
with open(f'{file_path}/updated_yearly_clusters.pkl', 'rb') as f:
    updated_yearly_clusters = pickle.load(f)

def load_graph(year):
    # load the StellarGraph for the current year
    with open(f"stellar_graph100_{year}.pkl", 'rb') as f:
        graph = pickle.load(f)
        return graph


# Function to compute cluster-to-cluster relationship matrix
def compute_cluster_relationship_matrix(year, clusters, edges_with_weights):
    # Get clusters for the current year
    cluster_to_words = clusters[year]  # {cluster_id: [word1, word2, ...]}
    
    # Map words to their cluster IDs
    word_to_cluster = {}
    for cluster_id, words in cluster_to_words.items():
        for word in words:
            word_to_cluster[word] = cluster_id
    
    # Initialize cluster-to-cluster weight aggregation
    cluster_matrix = defaultdict(lambda: defaultdict(float))
    
    # Process edges
    for edge, weight in edges_with_weights.items():  # Use `.items()`
        word1, word2 = sorted(edge)  # Unpack the edge
        cluster1 = word_to_cluster.get(word1)
        cluster2 = word_to_cluster.get(word2)
        
        if cluster1 is not None and cluster2 is not None:
            cluster_matrix[cluster1][cluster2] += weight
            # Avoid adding weight twice for self-loops
            if cluster1 != cluster2:
                cluster_matrix[cluster2][cluster1] += weight
    
    # Convert to a DataFrame
    cluster_ids = sorted(cluster_to_words.keys())
    matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
    
    for i, cluster1 in enumerate(cluster_ids):
        for j, cluster2 in enumerate(cluster_ids):
            matrix[i, j] = cluster_matrix[cluster1].get(cluster2, 0)
    
    relationship_df = pd.DataFrame(matrix, index=cluster_ids, columns=cluster_ids)
    return relationship_df

cluster_relationship_matrices = {}
# Loop through the years from 1987 to 2024
for year in range(1987, 2025):  
    print(f"Processing year: {year}")
    
    # Load the graph for the current year
    graph = load_graph(year)
    
    # Extract edges and weights from the graph
    edges, weights = graph.edges(include_edge_weight=True)
    edges_with_weights = {}
    for i in range(len(edges)):
        source, target = edges[i]
        weight = weights[i]
        edges_with_weights[(source, target)] = weight
    
    # Compute the cluster relationship matrix for the current year
    cluster_relationship_matrices[year] = compute_cluster_relationship_matrix(year, updated_yearly_clusters, edges_with_weights)
# Save the cluster relationship matrices for all years
with open(f'{file_path}/cluster_relationship_matrices.pkl', 'wb') as f:
    pickle.dump(cluster_relationship_matrices, f)


# plot heatmap fot each of relationships
# Function to load and visualize the matrix for a given year
def visualize_relationship_matrix(year):
    # Load the matrix from the CSV file
    matrix = cluster_relationship_matrices[year]
   
    
    # Print the DataFrame
    print(f"Cluster Relationship Matrix for Year {year}:\n")
    print(matrix)
    
    # Plot the matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=True, yticklabels=True)
    plt.title(f"Cluster Relationship Matrix ({year})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.show()

# Loop through the years and visualize each matrix
for year in range(1987, 2025):  # 1990 to 2000 inclusive
    visualize_relationship_matrix(year)




# Aggregate weights to create time series with symmetry and no duplicates
aggregated_relationships = defaultdict(list)
for year, matrix in cluster_relationship_matrices.items():
    cluster_ids = matrix.index
    for i, cluster_1 in enumerate(cluster_ids):
        for j, cluster_2 in enumerate(cluster_ids):
            # Process only the upper triangular matrix (excluding the diagonal)
            if i >= j:
                continue
            
            # Sort clusters to ensure symmetry
            key = tuple(sorted((cluster_1, cluster_2)))
            
            # Append the year and weight
            aggregated_relationships[key].append((year, matrix.loc[cluster_1, cluster_2]))

# Convert to a structured time series dataframe
time_series_data = {}
for (cluster_1, cluster_2), values in aggregated_relationships.items():
    years, weights = zip(*values)
    time_series_data[(cluster_1, cluster_2)] = pd.Series(data=weights, index=years)

# Print the resulting time series data
print("time_series_data: ", time_series_data)

# Save the time series data for all pairs of clusters
with open(f'{file_path}/cluster_relationship_time_series.pkl', 'wb') as f:
    pickle.dump(time_series_data, f)

time_window_length = 4
windowed_data = []

# Iterate over all time series in the data
for (cluster_1, cluster_2), series in time_series_data.items():
    years = series.index.to_list()  # Extract years
    weights = series.to_list()     # Extract weights
    
    # Create sliding window samples
    for i in range(len(years) - time_window_length + 1):
        window_weights = weights[i:i + time_window_length]
        start_year = years[i]
        sample = {
            'cluster_pair': (cluster_1, cluster_2),
            'start_year': start_year,
            'weights': window_weights
        }
        windowed_data.append(sample)

# Convert to a DataFrame for easier analysis and storage
windowed_df = pd.DataFrame(windowed_data)

# Show a preview of the DataFrame
print(windowed_df.head())

windowed_df.to_csv(f'{file_path}/windowed_cluster_relationships.csv', index=False)



# Function to apply Z-score normalization
def z_score_normalize(group):
    # Convert weights into a numpy array for calculations
    weights_array = np.array(group['weights'].tolist())  # List of lists of weights
    # Calculate the mean and standard deviation
    mean_weights = np.mean(weights_array, axis=1)  # Mean of each row (cluster pair)
    std_weights = np.std(weights_array, axis=1)    # Std dev of each row (cluster pair)
    
    # Normalize each weight
    normalized_weights = (weights_array - mean_weights[:, None]) / std_weights[:, None]  # Apply z-score
    group['normalized_weights'] = normalized_weights.tolist()  # Add normalized weights to the DataFrame
    
    return group

# Apply Z-score normalization by group (cluster_pair)
normalized_df = windowed_df.groupby('cluster_pair').apply(z_score_normalize)

# Show a preview of the DataFrame with normalized weights
print(normalized_df.head())

# Save the normalized DataFrame to a CSV file
normalized_df.to_csv(f'{file_path}/normalized_windowed_cluster_relationships.csv', index=False)
