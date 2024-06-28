# Phishing Email Detection

A machine learning project to classify emails as safe or phishing based on their content.

## Description

This project uses Natural Language Processing (NLP) and a Naive Bayes classifier to detect phishing emails. The dataset consists of emails labeled as either "Safe Email" or "Phishing Email".

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

## Installation

Instructions on how to install and set up the project. Include any prerequisites and dependencies.

```bash
# Clone the repository
git clone https://github.com/ohene-a-n/Phishing-Email-Detector.git

# Navigate to the project directory
cd Phishing-Email-Detector

# Install dependencies
pip install -r requirements.txt
```

## Usage

Instructions on how to use the project. Include code examples and screenshots if necessary.

### Running the Phishing Email Detection Script

```python
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the dataset
df1 = pd.read_csv('Phishing_Email.csv')

# Removing rows that have empty values and unnecessary columns
df1 = df1.dropna()
df1.drop(columns=['Unnamed: 0'], inplace=True)

# Creating a new column with boolean values for email type and removing the obsolete column
df1['Safe'] = df1['Email Type'] == 'Safe Email'
df1.drop(columns=['Email Type'], inplace=True)

# Function for preprocessing text
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def text_processor(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    store = []
    for word in text:
        if word isalnum():
            store.append(word)
    text = store[:]
    store.clear()
    for word in text:
        if word not in stop_words and word not in punctuations:
            store.append(word)
    return ' '.join(store)

# Add pre-processed text to the dataframe
df1['Processed_Text'] = df1['Email Text'].apply(text_processor)

# Split data into training set and testing set
train_x, test_x, train_y, test_y = train_test_split(df1['Processed_Text'], df1['Safe'], test_size=0.25)

# Feature extraction
cvec = CountVectorizer()
train_x_vectorised = cvec.fit_transform(train_x)

# Initializing the model
model = MultinomialNB()

# Train the model
model.fit(train_x_vectorised, train_y)

# Test the model
test_x_vectorised = cvec.transform(test_x)
result_y = model.predict(test_x_vectorised)

# Evaluation
print('Accuracy:', accuracy_score(test_y, result_y))
```

## Dataset

The dataset used in this project is sourced from [Phishing Email Detection](https://www.kaggle.com/datasets/subhajournal/phishingemails/data). It consists of emails labeled as "Safe Email" or "Phishing Email".

## Methodology

### Problem Statement

The goal of this project is to develop a machine learning model that can accurately classify emails as either safe or phishing. Phishing emails pose significant security risks, and an automated system for detecting them can help protect users from fraud and cyber threats.

### Approach

1. **Data Collection**:
   The dataset was sourced from [Phishing Email Detection](https://www.kaggle.com/datasets/subhajournal/phishingemails/data). It includes emails labeled as "Safe Email" or "Phishing Email".

2. **Data Preprocessing**:
   - **Cleaning**: Removed rows with empty values and unnecessary columns.
   - **Labeling**: Created a boolean column to indicate whether an email is safe.
   - **Text Processing**: Applied text preprocessing techniques such as tokenization, removal of stop words, and punctuation to clean and standardize the email text.

3. **Feature Extraction**:
   Used `CountVectorizer` to convert the processed text into numerical features suitable for machine learning.

4. **Model Selection**:
   Chose the `MultinomialNB` (Multinomial Naive Bayes) algorithm, which is well-suited for text classification problems.

5. **Training and Testing**:
   - Split the dataset into training and testing sets.
   - Trained the model on the training set.
   - Evaluated the model's performance on the testing set using accuracy as the metric.

### Results

The model achieved an accuracy of `96%` on the test set, indicating its effectiveness in distinguishing between safe and phishing emails. 

## Contributing

Guidelines for contributing to the project. Include information about submitting pull requests and issues.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:

- Name: Ohene Amankwaah Nkansah
- Email: oheneamankwaah210@gmail.com
- GitHub: [ohene-a-n](https://github.com/yourusername)
```
