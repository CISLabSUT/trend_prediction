import ast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score



# Define the year thresholds for splitting
training_threshold = 2020
testing_year = 2021

# Split the data into training and testing sets
training_data_lstm = normalized_df[normalized_df['start_year'] <= training_threshold]
testing_data_lstm = normalized_df[normalized_df['start_year'] == testing_year]

# Reset indices (optional, for cleaner DataFrame output)
training_data_lstm = training_data_lstm.reset_index(drop=True)
testing_data_lstm = testing_data_lstm.reset_index(drop=True)

# Print summary of the split
print("Training Data:")
# print(training_data.head())
print(training_data_lstm)

print("\nTesting Data:")
# print(testing_data.head())
print(testing_data_lstm)

print(training_data_lstm['normalized_weights'].head())
print(type(training_data_lstm['normalized_weights'].iloc[0]))


training_data_lstm['normalized_weights'] = training_data_lstm['normalized_weights'].apply(ast.literal_eval)



# Prepare input and output
X_lstm = np.array([sample[:-1] for sample in training_data_lstm['normalized_weights']])
y_lstm = np.array([sample[-1] for sample in training_data_lstm['normalized_weights']])



# Reshape X for LSTM input: (samples, timesteps, features)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
y_lstm = y_lstm.reshape((y_lstm.shape[0], 1))

# Define the LSTM model
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
    Dense(1)
])

# Compile the model
model_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history_lstm = model_lstm.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)

# Plotting training and validation loss for lstm model
plt.figure(figsize=(12, 6))
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# save the lstm model
file_path = "..."    
model_filename = "lstm_model2.h5"

model_lstm.save(os.path.join(file_path, model_filename))
print(f"Model saved at: {os.path.join(file_path, model_filename)}")


print(testing_data_lstm['normalized_weights'].head())
print(type(testing_data_lstm['normalized_weights'].iloc[0]))
testing_data_lstm['normalized_weights'] = testing_data_lstm['normalized_weights'].apply(ast.literal_eval)



# Prepare test input and output
X_test_lstm = np.array([sample[:-1] for sample in testing_data_lstm['normalized_weights']])
y_test_lstm = np.array([sample[-1] for sample in testing_data_lstm['normalized_weights']])

# Reshape X for LSTM input: (samples, timesteps, features)
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
y_test_lstm = y_test_lstm.reshape((y_test_lstm.shape[0], 1))


# Use the trained model to make predictions on the test data
predictions_lstm = model_lstm.predict(X_test_lstm)
print("Predictions made successfully.")

# Compute the Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse_lstm = mean_squared_error(y_test_lstm, predictions_lstm)
mae_lstm = mean_absolute_error(y_test_lstm, predictions_lstm)
print(f"Mean Squared Error: {mse_lstm}")
print(f"Mean Absolute Error: {mae_lstm}")


# Calculate R² for the predictions
r2_lstm = r2_score(y_test_lstm, predictions_lstm)

print(f"R² (Coefficient of Determination): {r2_lstm}")

# Plot predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_lstm, label='Actual')
plt.plot(predictions_lstm, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Weight')
plt.title(f'Predictions vs Actual Values (R²: {r2_lstm:.4f})')
plt.legend()
plt.show()



predicted_weights_lstm = predictions_lstm  # The predictions from the LSTM model
test_cluster_pairs_lstm = testing_data_lstm['cluster_pair']  # The actual cluster pairs from the test set

# Create a dictionary to store predicted weights for each cluster pair
predicted_cluster_weights_lstm = {}

# Map predicted weights to cluster pairs
for cluster_pair, weight in zip(test_cluster_pairs_lstm, predicted_weights_lstm):
    predicted_cluster_weights_lstm[cluster_pair] = weight

# Now we have a dictionary with cluster pairs as keys and predicted weights as values
print(predicted_cluster_weights_lstm)


# Step 1: Assuming the normalized test data contains 3 years' weights and we predict the 4th year's weight.
# We need to reverse the normalization for each cluster pair.

# Prepare the data for inverse normalization
original_weights = []
for i, (cluster_pair, predicted_weight) in enumerate(zip(test_cluster_pairs_lstm, predictions_lstm)):
    # Extract the actual normalized weights for the 3 years
    normalized_weights_3_years = np.array(ast.literal_eval(testing_data_lstm[testing_data_lstm['cluster_pair'] == cluster_pair]['weights'].values[0]))

    # Calculate the mean and standard deviation of the 4 data points (3 from test data + 1 predicted)
    # Adding the predicted weight as the 4th data point
    all_weights = np.append(normalized_weights_3_years, predicted_weight)
    mean_weight = np.mean(all_weights)
    std_weight = np.std(all_weights)

    # Step 2: Reverse the normalization using the formula
    original_weight = predicted_weight * std_weight + mean_weight
    original_weights.append((cluster_pair, original_weight))

# Now `original_weights` contains the cluster pair and the predicted weight on the original scale.
# Convert the list into a dictionary for easy reference.
predicted_original_weights_lstm = dict(original_weights)

# Print out the predicted weights on the original scale
print(predicted_original_weights_lstm)


# Sort the dictionary by weight values in descending order
sorted_weights_lstm = sorted(predicted_original_weights_lstm.items(), key=lambda x: x[1], reverse=True)

# Get the top 5 cluster pairs with the highest weights
top_5_cluster_pairs_lstm = sorted_weights_lstm[:5]

# Display the top 5 cluster pairs and their weights
for (cluster_pair, weight) in top_5_cluster_pairs_lstm:
    print(f"Cluster Pair: {cluster_pair}, Weight: {weight}")



