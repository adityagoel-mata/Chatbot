"""ChatbotProject_TopicSelection"""

import pandas as pd
import numpy as np

def read_dataset(filename, user_input):

  QAdata = pd.read_csv(filename, encoding='latin1')
  topics = pd.DataFrame(QAdata['ArticleTitle'])

  #Remove Duplicates
  topics = topics['ArticleTitle'].unique()
  
  print("You have selected the topic: ", topics[user_input])
  
  #Extract only the required data
  dataset = pd.DataFrame(QAdata[QAdata['ArticleTitle']==topics[user_input]])

  #Re-assign index values from 0
  dataset = dataset.reset_index()
  
  return dataset