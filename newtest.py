import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Load the dataset from a CSV file
df = pd.read_csv('hamro.csv')

# Filter out "Bike" and "Car" categories from the dataset
df = df[~df['Category'].isin(['Bike', 'Car'])]

# Preprocess text data by tokenizing, converting to lowercase, and removing stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Apply the preprocessing function to the 'Problem Statement' column and create a new 'Processed_Text' column
df['Processed_Text'] = df['Problem Statement'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Processed_Text'], df['Category'], test_size=0.2, random_state=42
)

# Create a text classification model using TfidfVectorizer and LogisticRegression in a pipeline
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Fit the model on the training data
model.fit(X_train, y_train)

# Function to classify user input and provide relevant category, solution, probabilities, and deviations
def classify_user_input(user_input):
    processed_input = preprocess_text(user_input)
    category = model.predict([processed_input])[0]
    probabilities = model.predict_proba([processed_input]).max(axis=1)
    return category, probabilities

# Lists to store predicted and actual categories
predicted_categories = []
actual_categories = []

# Simple chatbot loop
while True:
    # Take user input
    user_input = input("You: ")
    
    # Handle common greetings and goodbyes
    if user_input.lower() in ['hi', 'hello']:
        print("Bot: Hello! How can I assist you today?")
    elif user_input.lower() in ['bye', 'exit']:
        print("Bot: Goodbye! Have a great day!")
        break
    elif user_input.lower() in ['thank you', 'thanks']:
        print("Bot: You're welcome! If you have any more questions, feel free to ask.")
    else:
        # Classify user input and provide relevant information
        category, probabilities = classify_user_input(user_input)
        predicted_categories.append(category)
        
        # Print relevant information
        print(f"Bot: The most relevant category is: {category}")
        print(f"Bot: Predicted probabilities: {probabilities}")
        
        # Ask if the user has more queries
        more_queries = input("Bot: Do you have any other queries? (yes/no): ").lower()
        
        # Handle user responses to determine further interaction
        if more_queries == 'no':
            print("Bot: Goodbye! Have a great day!")
            break
        elif more_queries == 'yes':
            print("Bot: How may I assist you?")
        else:
            print("Bot: I'm sorry, I didn't understand that. How may I assist you?")

# Convert labels to binary for each category
binary_y_test = np.zeros_like(y_test)
for i, category in enumerate(model.classes_):
    binary_y_test[y_test == category] = i

# Calculate ROC curve and AUC for each category
plt.figure(figsize=(10, 8))
for i, category in enumerate(model.classes_):
    binary_y_scores = model.predict_proba(X_test)[:, i]
    fpr, tpr, thresholds = roc_curve(binary_y_test == i, binary_y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{category} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Each Category')
plt.legend(loc='lower right')
plt.show()
