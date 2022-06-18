from transformers import BertModel, BertTokenizer
import torch
from progressbar import *
use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

def get_bert_embedding(sentences, similarity_threshold = 0.5):
  pbar = ProgressBar(widgets=['Getting wordwise similarity ', Percentage(), ' ', Bar(),
        ' ', ETA(), ' '], maxval=len(sentences)).start()
  total_set = {}
  valid_set={}
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model = BertModel.from_pretrained("bert-base-uncased").to(device)
  count = 0
  for s in sentences:
    pbar.update(count)
    encoded_input = tokenizer(s, return_tensors='pt').to(device)
    output = model(**encoded_input)
    aa = torch.squeeze(output.last_hidden_state,dim=0)
    each = aa[1:-1,:]
    split_s = s.split()
    for word in range(0,len(split_s)):
      for another in range(0,len(split_s)):
        w1 = split_s[word]
        w2 = split_s[another]
        score = torch.cosine_similarity(each[word].view(1,-1),each[another].view(1,-1),dim=1)
        key = ''+w1+','+w2
        if key not in total_set.keys():
          total_set[key] = 1
        else:
          total_set[key]+=1
        if score.data>similarity_threshold:
          if key not in valid_set.keys():
            valid_set[key] = 1
          else:
            valid_set[key] += 1
    count +=1
  pbar.finish()
  return total_set,valid_set

# DO NOT TOKENIZE YOUR SENTENCE!!!!!! [[sentence0],[sentence1],[sentence2],[sentence3],.......]
#total_set,valid_set = get_bert_embedding(sentences)
#import pickle

#f = open('bert_semantic.pkl','wb')
#pickle.dump(valid_set,f)
#f.close()