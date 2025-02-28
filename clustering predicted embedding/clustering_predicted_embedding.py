from sklearn.cluster import KMeans
import pickle
from munkres import Munkres
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



# Perform clustering for the year 2024
# Use predicted embeddings and words
kmeans = KMeans(n_clusters=10, random_state=42)
predicted_cluster_labels_2024 = kmeans.fit_predict(predicted_embeddings)

# predicted centroids for 2024
predicted_centroids_2024 = kmeans.cluster_centers_

# Create a dictionary to store words in each predicted cluster for 2024
predicted_clusters_2024 = {i: [] for i in range(10)}  # Initialize 10 empty clusters
for word, cluster_label in zip(words2024, predicted_cluster_labels_2024):
    predicted_clusters_2024[cluster_label].append(word)






# Initialize Munkres instance
munkres = Munkres()


current_centroids = predicted_centroids_2024
previous_centroids = yearly_centroids[2023]

current_clusters = predicted_clusters_2024
previous_clusters = yearly_clusters[2023]

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
predicted_cluster_mapping_2024 = cluster_mapping

# Assign consistent cluster names based on the mapping
predicted_cluster_names_2024 = {}
for current_cluster, previous_cluster in cluster_mapping.items():
        predicted_cluster_names_2024[current_cluster] = cluster_names[2023][previous_cluster]

# Get clusters and cluster names for actual 2024
clusters = predicted_clusters_2024
cluster_name_mapping = predicted_cluster_names_2024  # Consistent cluster names for the year

# Rename clusters based on consistent names
renamed_clusters = {}
for cluster_id, words in clusters.items():
    renamed_cluster_name = cluster_name_mapping[cluster_id]
    renamed_clusters[renamed_cluster_name] = words

# Save renamed clusters for the year
predicted_updated_yearly_clusters_2024 = renamed_clusters
# --- Example: Access updated clusters ---
print(f"Updated predicted clusters for 2024:", predicted_updated_yearly_clusters_2024)
