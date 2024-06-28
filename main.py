import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Loading in the dataset
df1 = pd.read_csv('Phishing_Email.csv')

#Removing rows that have empty values and column that are needed
df1 = df1.dropna()
df1.drop(columns=['Unnamed: 0'], inplace=True)

#Make a new column where email type has a boolean value and removing obselete column
df1['Safe'] = df1['Email Type'] == 'Safe Email'
df1.drop(columns=['Email Type'], inplace=True)

#Function for removing preprocessing text

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def text_processor(text: str) -> str:
    
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    store = []
    
    for word in text:
        
        if word.isalnum():
            
            store.append(word)
            
            
    text = store[:]
    store.clear()
    
    for word in text:
        
        if word not in stop_words and word not in punctuations:
            
            store.append(word)

    return ' '.join(store)

#Add pre-processed text into dataframe
df1['Processed_Text'] = df1['Email Text'].apply(text_processor)

#Split data into training set and testing set
train_x, test_x, train_y, test_y = train_test_split(df1['Processed_Text'], df1['Safe'], test_size=0.25)

#Feature extraction
cvec = CountVectorizer()

train_x_vectorised = cvec.fit_transform(train_x)

#Initializing Model
model = MultinomialNB()

#Train Model
model.fit(train_x_vectorised, train_y)

#Test Model

test_x_vectorised = cvec.transform(test_x)

result_y = model.predict(test_x_vectorised)

#Evaluation
print('Accuracy:', accuracy_score(test_y, result_y))


