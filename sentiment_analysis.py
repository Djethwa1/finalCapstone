# loading necessary Libraries

import numpy as np

import pandas as pd

import spacy

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

# loading spacy model for text preocessing

nlp = spacy.load('en_core_web_sm')

# load csv file into pandas dataframe

df = pd.read_csv('amazon_product_reviews.csv', low_memory=False , encoding = 'utf-8' )

# Display the first 5 rows in the dataset

print(df.head())

# Display all the columns in the dataset to get an overview of what is in them

print(df.info('review.text'))

# Find all null values in the dataset

print(df.isnull().sum())

# Preprocessing the Text Data

# Selecting the 'reviews.text'column from the dataset and retrieve its data

reviews_data = df['reviews.text']

# Removing all missing values from the 'reviews.text column

clean_data = df.dropna(subset=['reviews.text'])

# Function to preprocess text to remove stopwords, punctuations
# converting uppercase text to lowercase, 
# Also Removing spaces at the begining and end of the text

def preprocess_reviews(text):
    doc=nlp(text)
    return" ".join([token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()])

# Funtion to convert preprocessed text to spacy doc
def get_doc(text):
     return nlp(text)

# Applying preprocessing

# df['reviews.text'] = df['reviews.text'].apply(preprocess_reviews)

# Taken a smaller sample of the dataframe to increase run time
# The dataframe is large and the code above takes a very long time to process

df = df.sample(2000, random_state=42)
df['doc'] = df['reviews.text'].apply(get_doc)

print(df.head())


# Analyse sentiment with Textblob

from textblob import TextBlob

# Function to measure the strength of the sentiment in a product review

def analyse_polatirty(text):
    doc = nlp(text)
    blob = TextBlob(text)
    polatity = blob.sentiment.polarity
    
    return polatity

# Test Usage and get polarity score

text = " I love this product. It does evertything that I want it to do"
print("\n")
print(" ")

#Alernative text message to test for negative polarity score

#text = " I hate this item. It is not to my expectations"

print("\n")
print(" ")

polarity_score = analyse_polatirty(text)

if polarity_score > 0:
    sentiment = 'positive'

elif polarity_score < 0:
    sentiment = 'negative'

else:
    sentiment = 'neutral'


print(f"Text:{text}\n The Polarity score is: {polarity_score} \n The Sentiment value is: {sentiment}")

print("")

# Compare the similarity of two reviews in the dataframe

# Retrieve the two reviews using indexing

my_review_choice_1 = 3

my_review_choice_2 = 4


reviews = df['reviews.text']

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform reviews into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(reviews)

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Calculate similarity of the two reviews
similarity_score = cosine_sim[my_review_choice_1, my_review_choice_2]

print(f"The Similarity score between the reviews in row {my_review_choice_1} and row {my_review_choice_2} of the reviews.text column is: {similarity_score:.2f}")