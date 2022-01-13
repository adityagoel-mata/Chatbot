"""ChatbotProject_Preprocessing"""

import pandas as pd
import string

import nltk
import nltk.corpus
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

#Pre-Processing 1: Convert all data to string**
def toString(text):
  text = str(text)
  return text

#Pre-Processing 2: Lower-case data
def lower_case(text):
  text = text.lower()
  return text

#Pre-Processing 3: Remove Punctuations from Data
def remove_punctuations(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text

#Pre-Processing 4: Tokenize each sentence
def tokenise(text):
  tokens = word_tokenize(text)
  return tokens

#Pre-Processing 5: Remove white spacing (taken care by tokenisation)
def white_space_removal(text):
  text.strip()

#Pre-Processing 6: POS + Lemmatisation
word_lem = WordNetLemmatizer()

def pos_lemma(tokens):
  lemma_array = []
  for word, tag in pos_tag(tokens):
    wntag = tag[0].lower()
    # a=adjectives, r=adverbs, n=nouns, v=verbs (POS are valid only for these tags)
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None   
    if not wntag:
      lemma = word
    else:
      lemma = word_lem.lemmatize(word, wntag)
    lemma_array.append(lemma)
  return lemma_array

#Pre-Processing 7: Remove Stop Words
def remove_stopwords(tokens):
  stop_words = set(stopwords.words("english"))
  filtered_text = [word for word in tokens if word not in stop_words]
  return filtered_text

#Convert Pre-Processed tokens back to a Sentence
def toSentence(tokens):
  str = ''
  for word in tokens:
    str = str + word + " "
  return str

#Pre-proccessing Phase

def pre_process(dataset):
  for i in range(len(dataset)):

      sentence = dataset['Question'][i]                      
      sentence = toString(sentence)                     #Covert data-type to string
      sentence = lower_case(sentence)                   #Lower-case the sentence
      sentence = remove_punctuations(sentence)          #Remove punctuations from sentence
      sent_tokens = tokenise(sentence)                  #Tokenize the sentence
      lemma_tokens = pos_lemma(sent_tokens)             #Lemmatise the tokens
      sentence = toSentence(lemma_tokens)               #Convert tokens back to sentence

      dataset['Question'][i] = sentence
  return dataset

def pre_process_question(question):
  
  sentence = toString(question)                     #Covert data-type to string
  sentence = lower_case(sentence)                   #Lower-case the sentence
  sentence = remove_punctuations(sentence)          #Remove punctuations from sentence
  sent_tokens = tokenise(sentence)                  #Tokenize the sentence
  lemma_tokens = pos_lemma(sent_tokens)             #Lemmatise the tokens
  sentence = toSentence(lemma_tokens)               #Convert tokens back to sentence
  
  return sentence