# Intoroduction
This project focuses on analyzing and predicting trends in scientific topics by leveraging text analysis, graph-based techniques, and deep learning models. By processing a large corpus of research articles, we aim to uncover how scientific concepts evolve over time and how different fields interact.

We have two main objectives:

Predicting Semantic Changes in Scientific Concepts
We develop a method that utilizes both semantic and structural information from past research articles to forecast how the meanings of words and concepts will shift in the future. This helps identify the emergence of new scientific fields and track the evolution of knowledge.

Predicting Relationships Between Scientific Clusters
We aim to model and predict the relationships between clusters of scientific words based on their historical interactions. This allows us to identify converging scientific fields, highlight emerging topics, and anticipate which research areas may gain or lose prominence.

To achieve these objectives, we break the process into several key steps:

Data Collection: Gathering scientific articles from reliable databases.

Feature Extraction: Developing methods to extract and combine semantic and structural information from texts.

Clustering & Trend Analysis: Identifying clusters of scientific words and tracking their relationships over time.

Deep Learning Modeling: Training neural networks to quantitatively model and predict changes in word meanings and connections between scientific topics.

This project provides valuable insights into the evolution of scientific knowledge and helps researchers, policymakers, and institutions better understand emerging trends in research.

# Project Workflow & Execution

1️⃣ Data Collection & Preprocessing

The dataset, collected from the Web of Science database, focuses on artificial intelligence and related sciences.
📌[data.csv](data.csv)

We preprocess the text to clean and structure the data.
📌[Preprocessing Code](preprocessing/data_preprocessing.py)

2️⃣ Feature Extraction

✨ Semantic Features (Word Embeddings): We use Word2Vec to generate word embeddings that capture meaning in scientific texts.
📌[word2vec Code](word2vec/word2vec.py)

🔗 Structural Features (Graph-Based Embeddings): We generate word co-occurrence graphs for each year and apply Deep Graph Infomax (DGI) to extract node embeddings.

📌 Graph Creation: [graph Code](graph/graph.py)

📌 DGI Embeddings:  [dgi Code](dgi/dgi.py)

3️⃣ combine Embeddings & Predictions

We combine Word2Vec and DGI embeddings to incorporate both semantic and structural word representations.
📌[Combine Word2Vec and DGI Code](word2vec%20and%20dgi/combine_word2vec_and_dgi.py)

An LSTM model predicts word embeddings for the next year based on three previous years.
📌[lstm1 Code](lstm1/lstm1.py)

4️⃣ Clustering and mapping

Clustering scientific words based on their embeddings to identify topics.To track topic evolution over time, we map clusters across years and extract their relationships
📌[clustering Code](clustering/clustering.py)

5️⃣ Predicting Scientific Topic Connections

We use an LSTM model to predict the future relationships between scientific topics.
📌[lstm2 Code](lstm2/lstm2.py)


📌clustering predicted embedding: [clustering predicted embedding Code](clustering%20predicted%20embedding/clustering_predicted_embedding.py)


# Technologies & Libraries

Python

Gensim (Word2Vec)

StellarGraph (Graph-based embeddings)

TensorFlow / Keras (LSTM models)

Scikit-learn 

Munkres (mapping)

Matplotlib & Seaborn (Visualization)



