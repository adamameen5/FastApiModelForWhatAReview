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
    


""" Original Review """
# Our solicitor Umar was fantastic he was introduced to us 10 weeks into the process after quite a lot of delays and quickly got everything resolved. We were really impressed with how fast and responsive he was and I'd highly recommend him. Overall I think Muve is a good option with some very competitive pricing but unfortunately we did experience some delays at the beginning of the process which I think it has to do with the amount of cases they had due to the stamp duty holiday but once our solicitor was assigned things moved really quickly. That's the only reason why I wouldn't give them 5 starts overall and I'd highly recommend getting a solicitor assigned as soon as you start the process. Would not have made the stamp duty deadline without them!  I was selling my existing property and purchasing a new one. It was all smooth sailing until the last few weeks where issues initiated by my buyers side, started to pop up out of nowhere. Umar and Ashleigh, really did everything they could to help me navigate these obstacles and were in constant contact. Keeping me updated and keeping the application on track. Completed on the final day of the stamp duty holiday and saved lots of £££. It was stressful at the end, but we made it.  Thank you both so much!

""" Summarized Review"""
#Overall I think Muve is a good option with some very competitive pricing but unfortunately we did experience some delays at the beginning of the process which I think it has to do with the amount of cases they had due to the stamp duty holiday but once our solicitor was assigned things moved really quickly. Completed on the final day of the stamp duty holiday and saved lots of £££. Our solicitor Umar was fantastic he was introduced to us 10 weeks into the process after quite a lot of delays and quickly got everything resolved.