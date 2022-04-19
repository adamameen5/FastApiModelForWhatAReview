from re import S
import string
from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
spacy.load('en_core_web_sm')
from heapq import nlargest
import json
from fastapi.middleware.cors import CORSMiddleware

#for sarcasm detection
import numpy as np
import pandas as pd 
import os
import re
import matplotlib.pyplot as plt
import keras
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import sys
from wordcloud import WordCloud, ImageColorGenerator

app = FastAPI()

students = {
    1: {
        "name":"John",
        "age":17,
        "year":"Year 12"
    }
}

#this basemodel is used to create a student object. It will define the required values to create a student object
class Student(BaseModel):
    name: str
    age: int
    year: str


class UpdateStudent(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    year: Optional[str] = None
    
    
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# creates an end point using the home url
@app.get("/")
def index():
    return {"name":"First Data"}


# creates an end point using the home url followed by /get-student/{student_id}
# Path parameter
@app.get("/get-student/{student_id}")
def get_student(student_id: int = Path(None,description = "The ID of the student you want to view", gt=0)):
    return students[student_id]

#Query parameter
@app.get("/get-by-name")
def get_student(*, name: Optional[str] = None, test : int):
    for student_id in students:
        if students[student_id]["name"] == name:
            return students[student_id]
    return {"Data": "Not found!"}


#Query parameter with a path parameter
@app.get("/get-by-name/{student_id}")
def get_student(*, student_id: int , name: Optional[str] = None, test : int):
    for student_id in students:
        if students[student_id]["name"] == name:
            return students[student_id]
    return {"Data": "Not found!"}


#taking a path parameter with an id to create a student object
@app.post("/create-student/{student_id}")
def create_student(student_id: int, student: Student):
    if student_id in students:
        return {"Error": "Student exists"}
    students[student_id] = student
    return students[student_id]


#to update a record
@app.put("/update_student/{student_id}")
def update_student(student_id : int, student : UpdateStudent):
    if student_id not in students:
        return {"Error" : "Student doesn't exist"}

    if student.name != None:
        students[student_id].name = student.name
    if student.age != None:
        students[student_id].age = student.age
    if student.year != None:
        students[student_id].year = student.year    
    return students[student_id]

#to delete a record
@app.delete("/delete_student/{student_id}")
def delete_student(student_id: int):
    if student_id not in students:
        return {"Error": "Record not found in database"}

    del students[student_id]
    return {"Message": "Student deleted succesfully"}


@app.get("/summarize/{review}")
def summarize(review: str):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(review)
    #tokenization
    tokens = [token.text for token in doc]
    #remove stop words and punctuations -- part of text cleaning
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    #if the word is introduced for the first time, then the occurance of the word will be 1
                    word_frequencies[word.text] = 1
                else:
                    #if the word is already introduced, then the occurance of the word will be increased by 1
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency
    #Sentence tokenization
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    #Get 30 percent of the sentence with the maximum score. 
    #Result of it will be: the number of sentences that will be in the summary
    select_length = int(len(sentence_tokens)*0.3)
    # To get the summary of the text
    summary = nlargest(select_length,sentence_scores,key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    #To combine the finaly summary statements to one paragraph
    summary = ' '.join(final_summary)
    print(summary)
    return summary
    

@app.get("/detectSarcasm/{review}")
def detectSarcasm(review: str):
    
    return trainModel(review)
    
    
def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    print("called clean_text")
    return text

def CleanTokenize(df):
    head_lines = list()
    lines = df["headline"].values.tolist()
    print("called CleanTokenize")
    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

def trainModel(s):
    data_1 = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
    data_2 = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
    data =  pd.concat([data_1, data_2])
    head_lines = CleanTokenize(data)

    # pos_data = data.loc[data['is_sarcastic'] == 1]
    # pos_head_lines = CleanTokenize(pos_data)
    # pos_lines = [j for sub in pos_head_lines for j in sub] 
    # word_could_dict=Counter(pos_lines)

    # wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)
    # plt.figure(figsize=(15,8))
    # plt.imshow(wordcloud)
    # plt.axis("off")

    validation_split = 0.2
    max_length = 25

    tokenizer_obj = Tokenizer()
    
    tokenizer_obj.fit_on_texts(head_lines)
    sequences = tokenizer_obj.texts_to_sequences(head_lines)

    word_index = tokenizer_obj.word_index
    print("unique tokens - ",len(word_index))
    vocab_size = len(tokenizer_obj.word_index) + 1
    print('vocab size -', vocab_size)

    lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
    sentiment =  data['is_sarcastic'].values

    indices = np.arange(lines_pad.shape[0])
    np.random.shuffle(indices)
    lines_pad = lines_pad[indices]
    sentiment = sentiment[indices]

    num_validation_samples = int(validation_split * lines_pad.shape[0])

    X_train_pad = lines_pad[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    X_test_pad = lines_pad[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]

    print('Shape of X_train_pad:', X_train_pad.shape)
    print('Shape of y_train:', y_train.shape)

    print('Shape of X_test_pad:', X_test_pad.shape)
    print('Shape of y_test:', y_test.shape)

    embeddings_index = {}
    embedding_dim = 100
    GLOVE_DIR = '/Users/adamameen/Documents/Masters/2nd Year/Msc Project/Dataset/glove.twitter.27B'
    f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))


    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    c = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            c+=1
            embedding_matrix[i] = embedding_vector
    print(c)

    embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    print('Summary of the built model...')
    print(model.summary())

    # history = model.fit(X_train_pad, y_train, batch_size=32, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

    x_final = pd.DataFrame({"headline":[s]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    pred*=100
    if pred[0][0]>=50: return "It's a sarcasm!" 
    else: return "It's not a sarcasm."

    # return "success"

# trainModel()


""" Original Review """
# Our solicitor Umar was fantastic he was introduced to us 10 weeks into the process after quite a lot of delays and quickly got everything resolved. We were really impressed with how fast and responsive he was and I'd highly recommend him. Overall I think Muve is a good option with some very competitive pricing but unfortunately we did experience some delays at the beginning of the process which I think it has to do with the amount of cases they had due to the stamp duty holiday but once our solicitor was assigned things moved really quickly. That's the only reason why I wouldn't give them 5 starts overall and I'd highly recommend getting a solicitor assigned as soon as you start the process. Would not have made the stamp duty deadline without them!  I was selling my existing property and purchasing a new one. It was all smooth sailing until the last few weeks where issues initiated by my buyers side, started to pop up out of nowhere. Umar and Ashleigh, really did everything they could to help me navigate these obstacles and were in constant contact. Keeping me updated and keeping the application on track. Completed on the final day of the stamp duty holiday and saved lots of £££. It was stressful at the end, but we made it.  Thank you both so much!

""" Summarized Review"""
#Overall I think Muve is a good option with some very competitive pricing but unfortunately we did experience some delays at the beginning of the process which I think it has to do with the amount of cases they had due to the stamp duty holiday but once our solicitor was assigned things moved really quickly. Completed on the final day of the stamp duty holiday and saved lots of £££. Our solicitor Umar was fantastic he was introduced to us 10 weeks into the process after quite a lot of delays and quickly got everything resolved.