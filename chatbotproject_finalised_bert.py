"""ChatbotProject_Finalised_BERT"""

import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

def toSentence(tokens):
  str = ''
  for word in tokens:
    str = str + word + " "
  return str

#  Importing Bert For Question Answering class from transformers library.
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def question_answer(question, paragraph):

    max_tokens_length = 512
    sep_tokens = 2   #Separator token. One between Q and A. Other at the end of A.

    total_input_ids = tokenizer.encode(question, paragraph)      #Encodes words to numbers. 
    sep_1_idx = total_input_ids.index(tokenizer.sep_token_id)    #index of the first [SEP] token

    sep = [total_input_ids[sep_1_idx]]
    question_idx = total_input_ids[0:sep_1_idx]                  #Array of numbers representing the question.
    
    #Number of Words from the paragraph to be sent to the transformer in one loop.
    step_size_for_loop = max_tokens_length-len(question_idx)-sep_tokens-50   
    #Eg. array = [CLS:0, .., SEP:12, ..,SEP:9999], Therefore step size = 512-12-2-50(buffer)= 448
    
    #i is the first word of the paragraph to be processed for answering the input question.
    for i in range(sep_1_idx+1, len(total_input_ids)-1, step_size_for_loop):

      #j is the last word of the snippet of the paragraph
      j = min(i +step_size_for_loop +50, len(total_input_ids))

      #Tokens of the paragraph
      paragraph_idx = total_input_ids[i : j]

      #Make an input_id array which consits of the question and the paragraph.
      input_ids = []
      input_ids.extend(question_idx)
      input_ids.extend(sep)
      input_ids.extend(paragraph_idx)
      input_ids.extend(sep)
      
      #string version of tokenized ids
      tokens = tokenizer.convert_ids_to_tokens(input_ids)

      #segment IDs
      sep_idx = input_ids.index(tokenizer.sep_token_id)
      num_seg_a = sep_idx+1
      num_seg_b = len(input_ids) - num_seg_a
      
      #list of 0s and 1s for segment embeddings
      segment_ids = [0]*num_seg_a + [1]*num_seg_b
      assert len(segment_ids) == len(input_ids)
      
      #model output using input_ids and segment_ids
      output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
      #reconstructing the answer
      answer_start = torch.argmax(output.start_logits)
      answer_end = torch.argmax(output.end_logits)
      answer = tokens[answer_start]

      if answer_end >= answer_start:
          answer = tokens[answer_start]
          for i in range(answer_start+1, answer_end+1):
              if tokens[i][0:2] == "##":
                  answer += tokens[i][2:]
              else:
                  answer += " " + tokens[i]

      
      if(answer.split()[0] != '[CLS]' and answer.split()[0] != '[SEP]'):
        break
                     
      if (answer.startswith("[CLS]") or answer.startswith("[SEP]")):
        if(i >= len(total_input_ids)- step_size_for_loop):
          answer = "Unable to find the answer to your question."
        else:
          continue
      
    print("\nPredicted answer:\n{}".format(answer.capitalize()))

def bert_computation(question, paragraph):
    question_answer(question, paragraph)