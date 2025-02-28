from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import gc
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import random
import matplotlib.cm as cm


# Initialize an empty dictionary to store word embedding matrices for each year
embedding_matrices100 = {}
vocabulary100 = {}

# Filter the DataFrame to include articles published between 1985 and 2024
articles_1985_to_2024 = new_df[new_df['Publication Year'].between(1985, 2024)]

# Initialize the Word2Vec model (global variable)
word2vec_model = None

# Function to process and save data for a specific year
def process_year(year):
    # Indicate that we are using the global variable
    global word2vec_model
       
    # Filter articles for the current year
    articles_year = articles_1985_to_2024[articles_1985_to_2024['Publication Year'] == year]

    # Tokenize the text for articles in the current year
    tokenized_articles_year = []
    for article in articles_year['Filtered Text']:
        tokens = word_tokenize(article)
        tokenized_articles_year.append(tokens)
          
    # Initialize or update the Word2Vec model
    if year == 1985:
        word2vec_model = Word2Vec(tokenized_articles_year, vector_size=100, window=5, min_count=1, workers=1)
        print(word2vec_model)
    else:
        word2vec_model.build_vocab(tokenized_articles_year, update=True)
        word2vec_model.train(tokenized_articles_year, total_examples=len(tokenized_articles_year), epochs=10)
        print(word2vec_model)
    
    # Extract word embeddings  for the current year
    word_embeddings_year = {}
    for word in word2vec_model.wv.index_to_key:
        if word in word2vec_model.wv.key_to_index:
            word_embeddings_year[word] = word2vec_model.wv[word]
            
    # Create a word embedding matrix for the current year        
    vocabulary_year = list(word_embeddings_year.keys())
    vocabulary100[year] = vocabulary_year
    embedding_matrix_year = np.array([word_embeddings_year[word] for word in vocabulary_year])
    
    # Store the embedding matrix in the dictionary
    embedding_matrices100[year] = embedding_matrix_year

    # Save the embedding matrix
    np.save(f"embedding_matrix100_{year}.npy", embedding_matrix_year)

    # Print dimensions and size of the embedding matrix for the current year
    print(f"Year: {year}")
    print(f"Dimension of embedding matrix: {embedding_matrix_year.shape}")
    print(f"Size of vocabulary: {len(vocabulary_year)}\n")

    # Clear variables to free memory
    del tokenized_articles_year
    del word_embeddings_year
    del embedding_matrix_year
    gc.collect()
    
# Process each year from 1985 to 2024
for year in range(1985, 2025):
    process_year(year)  
 
# Save the vocabulary dictionary
with open('vocabulary100.pkl', 'wb') as vocab_file100:
    pickle.dump(vocabulary100, vocab_file100)

# Save the embedding matrices dictionary
with open('embedding_matrices100.pkl', 'wb') as f100:
    pickle.dump(embedding_matrices100, f100)

# Plot the number of words per year
# Define the years range
years = range(1985, 2025)

# List to store the number of words per year
num_words_per_year = []

# Extract the number of words for each year
for year in years:
    # Load the embedding matrix
    embedding_matrix = np.load(f"embedding_matrix100_{year}.npy")
    num_words_per_year.append(embedding_matrix.shape[0])  # Number of words (rows)

# Plot the number of words per year
plt.figure(figsize=(12, 6))
plt.plot(years, num_words_per_year, marker='o', linestyle='-', color='b', label='Number of Words')
plt.title("Number of Words Per Year (1985-2024)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Words", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as an image
output_image = "number_of_words_per_year.png"
plt.savefig(output_image)
plt.show()

print(f"Chart saved as {output_image}")


# plot the word embedding over 35 years with 5 random word

# Load the embedding matrices dictionary
with open('embedding_matrices100.pkl', 'rb') as f100:
    embedding_matrices100 = pickle.load(f100)

# Load the vocabulary dictionary
with open('vocabulary100.pkl', 'rb') as vocab_file100:
    vocabulary100 = pickle.load(vocab_file100)

# Function to find words appearing in all selected years
def get_common_words(years, vocabulary):
    common_words = set(vocabulary[years[0]])
    for year in years[1:]:
        common_words &= set(vocabulary[year])
    return list(common_words)

# Select the years and get common words
years = range(1990, 2025)  # Example: 10 consecutive years
common_words = get_common_words(years, vocabulary100)
selected_words = common_words[:5]  # Select 20 words for visualization

# Prepare the data for dimensionality reduction
embeddings = []
labels = []
word_indices = []  # To keep track of which word corresponds to which embedding
for year in years:
    embedding_matrix = embedding_matrices100[year]
    word_to_index = {word: i for i, word in enumerate(vocabulary100[year])}
    for word in selected_words:
        if word in word_to_index:
            embeddings.append(embedding_matrix[word_to_index[word]])
            labels.append(f"{word}_{year}")
            word_indices.append(selected_words.index(word))  # Index of the word

embeddings = np.array(embeddings)

# Dimensionality reduction
svd = TruncatedSVD(n_components=2)
reduced_embeddings = svd.fit_transform(embeddings)

# Use the 'tab20' colormap for more distinct colors
cmap = cm.get_cmap('tab20', len(selected_words))

# Shuffle colors to make them more distinct
colors = [cmap(i) for i in range(len(selected_words))]
random.shuffle(colors)
word_colors = [colors[i] for i in word_indices]

# Plot the embeddings with a larger figure size
plt.figure(figsize=(18, 12))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, color=word_colors)

# Align labels more accurately with the nodes
for i, label in enumerate(labels):
    x, y = reduced_embeddings[i]
    plt.text(x, y, label, fontsize=9, ha='center', va='center', color=word_colors[i])

plt.title("Word Embeddings with 100 dimenssion Over 35 years(1990-2025) with 5 word")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.grid(False)

# Save the chart as an image file
plt.savefig('word embeddings(with 100 dimenssion) chart 1990-2025.jpg', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

plt.show()

