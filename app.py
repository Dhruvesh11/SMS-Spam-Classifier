import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(text):
    #1.LOWERCASE
    text=text.lower()
    #2.TOKENIZATION
    text=nltk.word_tokenize(text)
    #3.REMOVING SPECIAL CHARACTERS
    y=[]
    for i in text:
        if( i.isalnum()):
            y.append(i)
    #4.REMOVING STOP WORDS AND PUNCTUATION
    text=y[:]
    y.clear()
    for i in text:
        if(i not in stopwords.words('english') and i not in string.punctuation ):
            y.append(i)
    #5.STEMMING
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS-Spam-Classifier")
input_sms=st.text_area("Enter the message")

#BUTTON
if st.button('Predict'):


    #1.Preprocess
    transformed_sms=transform_text(input_sms)
    #2.Vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3.Predict
    result=model.predict(vector_input)
    print(result)
    #4.Display
    if(result==1):
        st.header("Spam")
    else:
        st.header("Ham")