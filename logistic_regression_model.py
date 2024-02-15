import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns



# load data
data = pd.read_csv('Womens_Clothing_E-Commerce_Reviews.csv')
data = data.dropna(subset=['Review Text', 'Recommended IND'])

# shuffling for better accuracy
def shuffling(training_images, labels):
    np.random.seed(1111) 
    ind_shuffled = np.random.permutation(len(training_images))

    images_shuffled = [training_images[i] for i in ind_shuffled]
    labels_shuffled = [labels[i] for i in ind_shuffled] # shuffle labels in the same order
    return np.array(images_shuffled), np.array(labels_shuffled)

# splitting data into train and validation
def partitioning(shuffled_training_images, shuffled_training_labels):
    
    num_valid = int((len(shuffled_training_images))/ 100)*20 # set aside 20% for validation

    images_valid = shuffled_training_images[0:num_valid]
    labels_valid = shuffled_training_labels[0:num_valid]

    images_train = shuffled_training_images[num_valid:]
    labels_train = shuffled_training_labels[num_valid:]

    return np.array(images_valid), np.array(labels_valid), np.array(images_train), np.array(labels_train)


# function for checking accuracy
def evaluation_metric(true_labels, predicted_labels):
    n = len(true_labels)
    s = (1/n) * np.sum(true_labels == predicted_labels)
    return s

def words_in_texts(words, texts):
    # Convert each review text to lowercase before checking for word presence
    indicator_array = np.array([texts.str.lower().str.contains(word.lower(), case=False, na=False) for word in words]).T.astype(int)
    return indicator_array

# Shuffle and partition the data
data_text_shuffled, data_labels_shuffled = shuffling(data['Review Text'].values, data['Recommended IND'].values)
text_validation, labels_validation, text_train, labels_train = partitioning(data_text_shuffled, data_labels_shuffled)


selected_words_bar = ['cheap', 'itchy', 'why', 'return', 'look', 'beautiful', 'go-to', 'not worth', 'quality'] # Generate features for the bar plot


indicator_data = pd.DataFrame(words_in_texts(selected_words_bar, data['Review Text']), columns=selected_words_bar)
indicator_data['Recommended'] = data['Recommended IND'].replace({1: 'Recommended', 0: 'Not Recommended'})

# Melt the DataFrame for the bar plot
melted_data = indicator_data.melt('Recommended', var_name='Words', value_name='Presence')

# Calculate proportions for the bar plot
word_proportions = (
    melted_data
    .groupby(['Words', 'Recommended'])
    .mean()
    .reset_index()
)

# Plot the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=word_proportions, x='Words', y='Presence', hue='Recommended')
plt.xlabel("Words")
plt.ylabel("Proportion of Reviews Containing Word")
plt.title("Frequency of Words in Recommended/Not Recommended Review Texts")
plt.tight_layout()
plt.show()

selected_words_model = ['cheap', 'itchy', 'why', 'return', 'look', 'beautiful', 'go-to', 'not worth', 'quality']  # You can change this list as needed


# Create feature matrices for the model training and predictions

X_train = words_in_texts(selected_words_model, pd.Series(text_train))
X_train = pd.DataFrame(X_train, columns=selected_words_model)
Y_train = np.array(labels_train)


X_validation = words_in_texts(selected_words_model, pd.Series(text_validation))
X_validation = pd.DataFrame(X_validation, columns=selected_words_model)
Y_validation = np.array(labels_validation)


# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Predictions and evaluation on training data
train_predictions = model.predict(X_train)
training_accuracy = evaluation_metric(Y_train, train_predictions)
print(f"Training Accuracy: {training_accuracy}")

# Predictions and evaluation on validation data
validation_predictions = model.predict(X_validation)
validation_accuracy = evaluation_metric(Y_validation, validation_predictions)
print(f"Validation Accuracy: {validation_accuracy}")
