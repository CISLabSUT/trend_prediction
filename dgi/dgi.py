import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, DeepGraphInfomax
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from stellargraph.utils import plot_history
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import random
import matplotlib.cm as cm



node_embedding_matrices_DGI100 = {}
vocabulary_DGI100= {}


# Loop over each year and process the graph
for year in range(1985, 2025):
    print(f"Processing DGI for year {year}...")

    # Load the StellarGraph for the current year
    with open(f'stellar_graph100_{year}.pkl', 'rb') as f:
        graph = pickle.load(f)

    # Ensure the graph is loaded with edge weights
    fullbatch_generator = FullBatchNodeGenerator(graph, sparse=True, weighted=True)
    gcn_model = GCN(layer_sizes=[100], activations=["relu"], generator=fullbatch_generator)

    # Set up CorruptedGenerator for Deep Graph Infomax
    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(graph.nodes())
    
    dgi_model = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = dgi_model.in_out_tensors()

    # Compile DGI model
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

    # Train DGI model
    gen = corrupted_generator.flow(graph.nodes())
    epochs = 100
    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
    plot_history(history)

    # Extract embeddings from the trained GCN model
    x_emb_in, x_emb_out = gcn_model.in_out_tensors()
    # for full batch models, squeeze out the batch dim (which is 1)
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = Model(inputs=x_emb_in, outputs=x_out)
    
    # Predict embeddings for all nodes
    node_embeddings = emb_model.predict(fullbatch_generator.flow(graph.nodes()))
    
    # Map embeddings to their corresponding words (nodes)
    embedding_dict = {node: embedding for node, embedding in zip(graph.nodes(), node_embeddings)}
    
    vocabulary_DGI100[year] = list(embedding_dict.keys())
    
    
    node_embedding_matrix_year = np.array([embedding_dict[word] for word in vocabulary_DGI100[year]])
    
    # Save the node embedding matrix for each year
    np.save(f"node_embedding_matrix_DGI100_{year}.npy", node_embedding_matrix_year)
    
    node_embedding_matrices_DGI100[year] = node_embedding_matrix_year

    print(f"Embeddings for year {year} processed and saved.")

# save vocabulary DGI
with open('vocabulary_DGI100.pkl', 'wb') as DGI_vocab_file100:
    pickle.dump(vocabulary_DGI100, DGI_vocab_file100)

# Save the node_embedding_matrices_DGI
with open('node_embedding_matrices_DGI100.pkl', 'wb') as node_matrix_DGI_file100:
    pickle.dump(node_embedding_matrices_DGI100, node_matrix_DGI_file100)    
     

print("All yearly embeddings saved successfully.")



# plot node embedding over 10 years with 10 random node.

# Load the vocabulary and embedding matrices
with open('vocabulary_DGI100.pkl', 'rb') as DGI_vocab_file100:
    vocabulary_DGI100 = pickle.load(DGI_vocab_file100)
    
with open('node_embedding_matrices_DGI100.pkl', 'rb') as node_matrix_DGI_file100:
    node_embedding_matrices_DGI100 = pickle.load(node_matrix_DGI_file100)
       

# Function to find nodes appearing in all selected years
def get_common_nodes(years, vocabulary):
    common_nodes = set(vocabulary[years[0]])
    for year in years[1:]:
        common_nodes &= set(vocabulary[year])
    return list(common_nodes)

# Select the years and get common nodes
years = range(2015,2025)  # Example: 10 consecutive years
common_nodes = get_common_nodes(years, vocabulary_DGI100)
selected_nodes = common_nodes[:10]  # Select 10 nodes for visualization

# Prepare the data for dimensionality reduction
embeddings = []
labels = []
node_indices = []  # To keep track of which node corresponds to which embedding
for year in years:
    node_embedding_matrix = node_embedding_matrices_DGI100[year]
    node_to_index = {node: i for i, node in enumerate(vocabulary_DGI100[year])}
    for node in selected_nodes:
        if node in node_to_index:
            embeddings.append(node_embedding_matrix[node_to_index[node]])
            labels.append(f"{node}_{year}")
            node_indices.append(selected_nodes.index(node))  # Index of the node

embeddings = np.array(embeddings)

# Dimensionality reduction
svd = TruncatedSVD(n_components=2)
reduced_embeddings = svd.fit_transform(embeddings)

# Use the 'tab20' colormap for more distinct colors
cmap = cm.get_cmap('tab20', len(selected_nodes))

# Shuffle colors to make them more distinct
colors = [cmap(i) for i in range(len(selected_nodes))]
random.shuffle(colors)
node_colors = [colors[i] for i in node_indices]

# Plot the embeddings with a larger figure size
plt.figure(figsize=(18, 12))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, color=node_colors)

# Align labels more accurately with the nodes
for i, label in enumerate(labels):
    x, y = reduced_embeddings[i]
    plt.text(x, y, label, fontsize=9, ha='center', va='center', color=node_colors[i])

plt.title("node Embeddings DGI100 Over 10 years(2015-2025) with 10 node")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.grid(False)

# Save the chart as an image file
plt.savefig('node_embeddings_DGI100_chart 2015-2025 WITH 10 words.jpg', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

plt.show()
