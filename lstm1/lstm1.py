import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error




# create dataset for lstm1
# Initialize an empty dataset to store samples
combined_dataset_dgi100 = []
current_year = 2024  # Starting point

# Iterate through each word in the common vocabulary for the current year
for word in common_vocabulary100[current_year]:
    # Initialize the starting year for each word as the current year
    start_year = 1985

    while start_year <= 2024:
        # Initialize a flag for checking embeddings for four consecutive years
        combined_embeddings_exist_for_consecutive_years = True
        word_sample = {'word and start year': [], 'embeddings': []}
        word_sample['word and start year'].append(word)
        word_sample['word and start year'].append(start_year)

        # Check embeddings for four consecutive years from start_year to start_year +3
        for year in range(start_year, start_year + 3):
            # Check if the year exists in common_vocabulary100 and combined_embeddings_matrices100
            if year in common_vocabulary100 and year in combined_embeddings_matrices100:
                # Check if the word exists in the common_vocabulary100 for the current year
                if word in common_vocabulary100[year]:
                    # Add the embedding for the year to the sample
                    word_index = common_vocabulary100[year].index(word)
                    word_sample['embeddings'].append(combined_embeddings_matrices100[year][word_index])
                else:
                    # If the word is not in the vocabulary for the year, set the flag to False
                    combined_embeddings_exist_for_consecutive_years = False
                    break
            else:
                # If the year does not exist in common_vocabulary100 or combined_embeddings_matrices100, break the loop
                combined_embeddings_exist_for_consecutive_years = False
                break

        # Check embedding for the fourth year
        if combined_embeddings_exist_for_consecutive_years:
          year = start_year + 3
          # Check if the year exists in vocabulary100 and embedding_matrices100
          if year in vocabulary100 and year in embedding_matrices100:
            if word in vocabulary100[year]:
              # Add the embedding for the year to the sample
              word_index = vocabulary100[year].index(word)
              word_sample['embeddings'].append(embedding_matrices100[year][word_index])
            else:
              # If the word is not in the vocabulary100 for the year, set the flag to False
              combined_embeddings_exist_for_consecutive_years = False

          else:
            # If the year does not exist in vocabulary100 or embedding_matrices100, set the flag to False
            combined_embeddings_exist_for_consecutive_years = False
          

        # If embeddings exist for all four consecutive years, add the sample to the dataset
        if combined_embeddings_exist_for_consecutive_years:
            combined_dataset_dgi100.append(word_sample)
            start_year += 1
        else:
            # Move to the next year
            start_year += 1



file_path = "..."


# Save the combined_dataset_dgi100 to a file
with open(f'{file_path}/combined_dataset_dgi100.pkl', 'wb') as combined_dataset_dgi_file:
    pickle.dump(combined_dataset_dgi100, combined_dataset_dgi_file)

# prepare data for lstm

# Initialize lists for input and output sequences train and validation
combined_input_sequences_node_train_dgi100 = []
combined_output_sequences_node_train_dgi100 = []
combined_train_words_and_start_year_dgi100 = []


# Initialize lists for input and output sequences test
combined_input_sequences_node_test_dgi100 = []
combined_output_sequences_node_test_dgi100 = []
combined_test_words_and_start_year_dgi100 = []


# Split data into training and test based on years
for sample in combined_dataset_dgi100:# my dataset_dgi100 here includes samples from 1985-2024
    start_year = sample['word and start year'][1]

    if start_year <= 2020: 
        combined_input_sequences_node_train_dgi100.append(sample['embeddings'][:3])  # Input embeddings for 3 years
        combined_output_sequences_node_train_dgi100.append(sample['embeddings'][3])  # Target embedding for the 4th year
        combined_train_words_and_start_year_dgi100. append(sample['word and start year'])
    elif start_year == 2021:
        combined_input_sequences_node_test_dgi100.append(sample['embeddings'][:3])
        combined_output_sequences_node_test_dgi100.append(sample['embeddings'][3])
        combined_test_words_and_start_year_dgi100.append(sample['word and start year'])
        
        

# Convert the lists to numpy arrays
combined_input_sequences_node_train_dgi100 = np.array(combined_input_sequences_node_train_dgi100)
combined_output_sequences_node_train_dgi100 = np.array(combined_output_sequences_node_train_dgi100)

combined_input_sequences_node_test_dgi100 = np.array(combined_input_sequences_node_test_dgi100)
combined_output_sequences_node_test_dgi100 = np.array(combined_output_sequences_node_test_dgi100)


# Reshape the train input sequences to (num_samples, sequence_length, embedding_dim)
num_samples_combined_input_node_train_dgi100 = combined_input_sequences_node_train_dgi100.shape[0]
print('num_samples_combined_input_node_train_dgi100: ', num_samples_combined_input_node_train_dgi100)

sequence_length_combined_input_node_train_dgi100= combined_input_sequences_node_train_dgi100.shape[1]
print('sequence_length_combined_input_node_train_dgi100: ', sequence_length_combined_input_node_train_dgi100)

embedding_dim_combined_input_node_train_dgi100 = combined_input_sequences_node_train_dgi100.shape[2]
print('embedding_dim_combined_input_node_train_dgi100: ', embedding_dim_combined_input_node_train_dgi100)

combined_input_sequences_node_train_dgi100 = combined_input_sequences_node_train_dgi100.reshape(num_samples_combined_input_node_train_dgi100, sequence_length_combined_input_node_train_dgi100, embedding_dim_combined_input_node_train_dgi100)


# output_sequences_node_train_dgi are already 2D arrays in the shape of (num_samples, embedding_dim), so no reshaping is required.
num_samples_combined_output_node_train_dgi100 = combined_output_sequences_node_train_dgi100.shape[0]
print('num_samples_combined_output_node_train_dgi100: ', num_samples_combined_output_node_train_dgi100)

embedding_dim_combined_output_node_train_dgi100 = combined_output_sequences_node_train_dgi100.shape[1]
print('embedding_dim_combined_output_node_train_dgi100: ', embedding_dim_combined_output_node_train_dgi100)

combined_output_sequences_node_train_dgi100 = combined_output_sequences_node_train_dgi100.reshape(num_samples_combined_output_node_train_dgi100, embedding_dim_combined_output_node_train_dgi100)




# Reshape the input sequences test to (num_samples, sequence_length, embedding_dim)
num_samples_combined_input_node_test_dgi100 = combined_input_sequences_node_test_dgi100.shape[0]
print('num_samples_combined_input_node_test_dgi100: ',num_samples_combined_input_node_test_dgi100)
sequence_length_combined_input_node_test_dgi100 = combined_input_sequences_node_test_dgi100.shape[1]
print('sequence_length_combined_input_node_test_dgi100: ',sequence_length_combined_input_node_test_dgi100)
embedding_dim_combined_input_node_test_dgi100 = combined_input_sequences_node_test_dgi100.shape[2]
print('embedding_dim_combined_input_node_test_dgi100: ',embedding_dim_combined_input_node_test_dgi100)

combined_input_sequences_node_test_dgi100 = combined_input_sequences_node_test_dgi100.reshape(num_samples_combined_input_node_test_dgi100, sequence_length_combined_input_node_test_dgi100, embedding_dim_combined_input_node_test_dgi100)

# output_sequences are already 2D arrays in the shape of (num_samples, embedding_dim), so no reshaping is required.
num_samples_combined_output_node_test_dgi100 = combined_output_sequences_node_test_dgi100.shape[0]
print('num_samples_combined_output_node_test_dgi100: ',num_samples_combined_output_node_test_dgi100)
embedding_dim_combined_output_node_test_dgi100 = combined_output_sequences_node_test_dgi100.shape[1]
print('embedding_dim_combined_output_node_test_dgi100: ',embedding_dim_combined_output_node_test_dgi100)

combined_output_sequences_node_test_dgi100 = combined_output_sequences_node_test_dgi100.reshape(num_samples_combined_output_node_test_dgi100, embedding_dim_combined_output_node_test_dgi100)


# Save the mapping of words and start years for the test set and train set
with open(f"{file_path}/train_words_and_start_years.pkl", "wb") as file:
    pickle.dump(combined_train_words_and_start_year_dgi100, file)

with open(f"{file_path}/test_words_and_start_years.pkl", "wb") as file:
    pickle.dump(combined_test_words_and_start_year_dgi100, file)

# save train input and output
with open(f"{file_path}/train_input_lstm1.pkl", "wb") as file:
    pickle.dump(combined_input_sequences_node_train_dgi100, file)

with open(f"{file_path}/train_output_lstm1.pkl", "wb") as file:
    pickle.dump(combined_output_sequences_node_train_dgi100, file)

# save test input and output
with open(f"{file_path}/test_input_lstm1.pkl", "wb") as file:
    pickle.dump(combined_input_sequences_node_test_dgi100, file)

with open(f"{file_path}/test_output_lstm1.pkl", "wb") as file:
    pickle.dump(combined_output_sequences_node_test_dgi100, file)



# train lstm model for predicted word2vec embedding

# Define the LSTM model with adjusted parameters
def build_lstm_model(sequence_length, embedding_dim, embedding_dim_out, lstm_units=300, l2_reg=1e-4):
    model = Sequential()
    model.add(
        LSTM(
            lstm_units,
            activation='relu',
            input_shape=(sequence_length, embedding_dim),
            kernel_regularizer=l2(l2_reg)
        )
    )
    model.add(Dense(embedding_dim_out, kernel_regularizer=l2(l2_reg)))  # Output layer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Get dimensions from training data
sequence_length = combined_input_sequences_node_train_dgi100.shape[1]
embedding_dim = combined_input_sequences_node_train_dgi100.shape[2]
embedding_dim_out = combined_output_sequences_node_train_dgi100.shape[1]

# Build the model
combined_model100_3 = build_lstm_model(
    sequence_length, embedding_dim, embedding_dim_out, lstm_units=300, l2_reg=1e-4
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

# Train the model with adjusted callbacks
history3 = combined_model100_3.fit(
    combined_input_sequences_node_train_dgi100,
    combined_output_sequences_node_train_dgi100,
    validation_split=0.2,
    epochs=100,
    batch_size=128,  # Reduced batch size
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, reduce_lr]
)

# Plot convergence diagrams for loss and MAE
plt.figure(figsize=(12, 5))

# Loss convergence
plt.subplot(1, 2, 1)
plt.plot(history3.history['loss'], label='Training Loss (MSE)')
plt.plot(history3.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()

# MAE convergence
plt.subplot(1, 2, 2)
plt.plot(history3.history['mae'], label='Training MAE')
plt.plot(history3.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training and Validation MAE')
plt.legend()

plt.tight_layout()
plt.show()


# save the lstm model 
model_filename = "lstm_model.h5"

combined_model100_3.save(os.path.join(file_path, model_filename))
print(f"Model saved at: {os.path.join(file_path, model_filename)}")


# Use the trained model to make predictions on the test data
predictions1 = combined_model100_3.predict(combined_input_sequences_node_test_dgi100)
print("Predictions1 made successfully.")

# Save the predictions1
with open(f"{file_path}/predictions1.pkl", "wb") as file:
    pickle.dump(predictions1, file)

print("Predictions saved successfully.")


# Compute the Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(combined_output_sequences_node_test_dgi100, predictions1)
mae = mean_absolute_error(combined_output_sequences_node_test_dgi100, predictions1)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Compute individual errors for plotting
errors = np.linalg.norm(predictions1 - combined_output_sequences_node_test_dgi100, axis=1)

# Plot the histogram of errors
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# load the predictions1
with open(f"{file_path}/predictions1.pkl", "rb") as file:
    predictions1 = pickle.load(file)

# Create separate lists for predicted embeddings
predicted_embeddings = []

# Iterate through predictions and extract the words and embeddings
for prediction in predictions1:
    predicted_embeddings.append(prediction)


# Load the mapping of words and start years for the test set
with open(f"{file_path}/test_words_and_start_years.pkl", "rb") as file:
    test_words_and_start_years = pickle.load(file)

# Create separate lists for words and their corresponding predicted embeddings
words2024 = []
actual_embeddings_2024 = []

# Iterate through predictions and extract the words and embeddings
for i, embedding in enumerate(combined_output_sequences_node_test_dgi100):
    word, start_year = test_words_and_start_years[i]
    words2024.append(word)
    actual_embeddings_2024.append(embedding)

