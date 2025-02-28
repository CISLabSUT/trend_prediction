import numpy as np
import pickle


# Define the directory on one Drive where I want to save the files
file_path = "..."


# Load the saved Word2Vec embedding matrices and vocabulary
with open('embedding_matrices100.pkl', 'rb') as f100:
    embedding_matrices100 = pickle.load(f100)

with open('vocabulary100.pkl', 'rb') as vocab_file100:
    vocabulary100 = pickle.load(vocab_file100)

# Load the saved DGI embedding matrices and vocabulary
with open('vocabulary_DGI100.pkl', 'rb') as DGI_vocab_file100:
    vocabulary_DGI100 = pickle.load(DGI_vocab_file100)

with open('node_embedding_matrices_DGI100.pkl', 'rb') as node_matrix_DGI_file100:
    node_embedding_matrices_DGI100 = pickle.load(node_matrix_DGI_file100)

# Initialize dictionaries to store combined embeddings and vocabulary
combined_embeddings_matrices100 = {}
common_vocabulary100 = {}

# Find common vocabulary between Word2Vec and dgi models and combine embeddings
for year in range(1985, 2025):
    vocab_word2vec = set(vocabulary100[year])
    vocab_dgi = set(vocabulary_DGI100[year])
    
    print(f'len_vocab_word2vec100 {year}:', len(vocab_word2vec))
    print(f'len_vocab_dgi100 {year}:', len(vocab_dgi))
    
    common_vocab = list(vocab_word2vec.intersection(vocab_dgi))
    common_vocabulary100[year] = common_vocab
    print(f'len_common_vocab100 {year}:', len(common_vocab))

    # Extract combined embeddings for the current year
    combined_embeddings_year = {}
    for word in common_vocab:
        # Get the index of the word in the Word2Vec vocabulary
        word2vec_index = vocabulary100[year].index(word)
        embedding_word2vec = embedding_matrices100[year][word2vec_index]
        
        # Get the index of the word in the dgi vocabulary
        dgi_index = vocabulary_DGI100[year].index(word)
        embedding_dgi = node_embedding_matrices_DGI100[year][dgi_index]

        # Combine the embeddings by concatenation
        combined_embedding = np.concatenate((embedding_word2vec, embedding_dgi))
        combined_embeddings_year[word] = combined_embedding

    # Create a combined embedding matrix for the current year
    combined_embedding_matrix_year = np.array([combined_embeddings_year[word] for word in common_vocab])
    
    # Save the combined embedding matrix for each year
    np.save(f"{file_path}/combined_embedding_matrix100_{year}.npy", combined_embedding_matrix_year)
    
    # Print dimensions of the combined embedding matrix and size of vocabulary for the current year
    print(f"Year: {year}")
    print(f"Dimension of combined embedding matrix: {combined_embedding_matrix_year.shape}")
    print(f"Size of vocabulary: {len(common_vocab)}\n")

    # Store the combined embedding matrix in the dictionary
    combined_embeddings_matrices100[year] = combined_embedding_matrix_year
    
# Save the common vocabulary dictionary
with open(f'{file_path}/common_vocabulary100.pkl', 'wb') as common_vocabulary100_file:
    pickle.dump(common_vocabulary100, common_vocabulary100_file)

# Save the combined embeddings matrices dictionary
with open(f'{file_path}/combined_embeddings_matrices100.pkl', 'wb') as combined_matrices100_file:
    pickle.dump(combined_embeddings_matrices100, combined_matrices100_file)


