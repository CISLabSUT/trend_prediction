import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load dataset
raw_df = pd.read_csv(r"data.csv", low_memory=False)
print("raw dataframe head: ", raw_df.head())

# Keep only relevant columns
relevant_columns = ['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus', 'Publication Year']
raw_df = raw_df[relevant_columns]
print("raw dataframe head with relevent column: ", raw_df.head())
      
# Convert all relevant text columns to lowercase
text_columns = ['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus']
raw_df[text_columns] = raw_df[text_columns].apply(lambda x: x.str.lower() if x.dtype == "object" else x)
print("raw dataframe head with lowercase text column: ", raw_df.head())

# Remove duplicate rows
df = raw_df.drop_duplicates()

# Display the first few rows of the DataFrame without duplicates
print(df.head())

# Optionally, save the DataFrame without duplicates to a new CSV file
df.to_csv(r"web of science 100000_no_duplicates.csv", index=False)


# Combine text columns into a single column
df['Combined Text'] = df[['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus']].astype(str).agg(' '.join, axis=1)

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove numerical terms
    tokens = [word for word in tokens if not word.isdigit()]

    # Part-of-speech tagging
    tagged_tokens = nltk.pos_tag(tokens)

    # Filter out nouns
    nouns = [word for word, pos in tagged_tokens if pos.startswith('N')]

    # Convert to lowercase
    nouns = [word.lower() for word in nouns]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    nouns = [word for word in nouns if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    nouns = [lemmatizer.lemmatize(word) for word in nouns]


    return ' '.join(nouns)

# Apply preprocessing to combined text column
df['Preprocessed_Text'] = df['Combined Text'].apply(preprocess_text)

# Display the result
print(df['Preprocessed_Text'])


# Create a CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the 'Preprocessed_Text' column
X = vectorizer.fit_transform(df['Preprocessed_Text'])

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Calculate the count of each word
word_counts = X.sum(axis=0)

# Filter out words that occur less than three times
filtered_words = [word for word, count in zip(feature_names, word_counts.tolist()[0]) if count >= 3]

# Filter the 'Processed Text' column based on the selected words
df['Filtered Text'] = df['Preprocessed_Text'].apply(lambda text: ' '.join([word for word in text.split() if word in filtered_words]))

# Display the result
print(df['Filtered Text'])

print(df.head)
# Drop rows with NaN values in specific columns ( 'Filtered Text')
df.dropna(subset=['Filtered Text'], inplace=True)

# Display the first few rows of the DataFrame after removing NaN values
print(df.head())
# new dataframe
new_df = pd.concat([df[['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus', 'Publication Year', 'Combined Text', 'Preprocessed_Text', 'Filtered Text']]], axis=1)

# Display the result
print(new_df)

# save new_df
new_df.to_csv(r"web of science 100000_no_duplicates_and_cleaned_preprocessed.csv", index=False)

# Check for null values in the 'Filtered Text' column
null_filtered_text = new_df['Filtered Text'].isnull().sum()
print(f"Number of null values in 'Filtered Text': {null_filtered_text}")

# Remove rows with null values in the 'Filtered Text' column
new_df = new_df.dropna(subset=['Filtered Text'])

# Display the result
print(new_df)
