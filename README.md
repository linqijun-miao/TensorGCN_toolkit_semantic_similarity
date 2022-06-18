# TensorGCN_toolkit_semantic_similarity
This is a toolkit to calculate semantic similarity using LSTM or BERT for the implementation of [TensorGCN](https://arxiv.org/pdf/2001.05313.pdf) in paper:

Liu X, You X, Zhang X, et al. Tensor graph convolutional networks for text classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 8409-8416.

# Requirement

nltk

pytorch

transformers >= 4.11.3


# Usage

Strongly recommend you to use BERT version, makes your life a little bit easier!

## BERT based similarity

```python
# DO NOT TOKENIZE YOUR SENTENCE!!!!!! [[sentence0],[sentence1],[sentence2],[sentence3],.......]
total_set,valid_set = get_bert_embedding(sentences)
import pickle

f = open('bert_semantic.pkl','wb')
pickle.dump(valid_set,f)
f.close()
```

## LSTM based similarity
You will need to train your LSTM model first using train_model(train_data,test_data), therefore you will split your data into two part!

```python
# DO NOT TOKENIZE YOUR SENTENCE!!!!!! [[sentence0],[sentence1],[sentence2],[sentence3],.......]
train_model(train_data,train_label,test_data,test_label)
get_similarity(train_data,train_label,test_data,test_label)
```
