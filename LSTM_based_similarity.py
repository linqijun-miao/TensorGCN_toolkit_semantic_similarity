from itertools import count
import pickle
from progressbar import ProgressBar
from torch import nn, optim
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

save_model = True
model_path = './new_lstm.pth'
use_cuda = torch.cuda.is_available()
#device = torch.device('cuda' if use_cuda else 'cpu')
device = torch.device('cpu')

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, word2id):
        super(TextClassifier, self).__init__()
        self.embedding_size = embedding_size
        self.word2id = word2id
        self.embeddingLayer = nn.Embedding(vocab_size+2, embedding_size, padding_idx=vocab_size+1)
        #self.sentenceEncoder = nn.GRU(input_size=embedding_size, hidden_size=embedding_size//2, num_layers=1, bidirectional=True, batch_first=True)
        self.sentenceEncoder2 = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size // 2, num_layers=1,
                                       bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )
    def forward(self, sentences):
        text = sentences
        max_length = max(len(s) for s in text)
        embedding, sorted_len, reversed_indices = self.embedding_lookup(text, max_length)
        packed_embed = pack_padded_sequence(embedding, sorted_len, batch_first=True)

        each, (h,c) = self.sentenceEncoder2(packed_embed)
        h = h[-2:]
        h = torch.cat([h[0], h[1]], dim=-1)
        feat = h[reversed_indices]
        user_feat = torch.mean(feat, dim=0, keepdim=True)
        prediction = self.classifier(user_feat)
        return prediction,each
    def embedding_lookup(self, sentences, max_length):
        ids = []
        lengths = []
        for sentence in sentences:
            id = []
            lengths.append(len(sentence))
            for word in sentence:
                if word in self.word2id:
                    id.append(self.word2id[word])
                else:
                    id.append(self.embeddingLayer.padding_idx - 1)
            id += [self.embeddingLayer.padding_idx for _ in range(max_length - len(id))]
            ids.append(id)
        ids = torch.LongTensor(ids).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        sorted_len, indices = torch.sort(lengths, 0, descending=True)
        _, reversed_indices = torch.sort(indices, 0)
        ids = ids[indices]
        return self.embeddingLayer(ids), sorted_len.tolist(), reversed_indices.to(device)

def cal_metrics(prediction, ground_truth):
    eps = 1e-15
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    total_num = len(ground_truth)

    for (p, l) in zip(prediction, ground_truth):
        if p == l:
            correct += 1
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    acc = correct / total_num
    precision = tp / (tp + fp + eps)
    n_precision = tn / (tn + fn + eps)
    recall = tp / (tp + fn + eps)
    n_recall = tn / (tn + fp + eps)
    F1 = 2 * precision * recall / (precision + recall + eps)
    n_F1 = 2 * n_precision * n_recall / (n_precision + n_recall +eps)
    m_precision = (precision + n_precision) / 2
    m_recall = (recall + n_recall) / 2
    m_F1 = (F1 + n_F1) / 2
    return acc, m_precision, m_recall, m_F1


class UserDataset(Dataset):
    def __init__(self, data,label):
        self.ds = []
        self.vocab = set()
        self.word2id = {}
        self.labels = label

        for sen in data:
            cleaned_sen = [w.lower() for w in sen.split()]
            self.vocab.update(cleaned_sen)
            self.ds.append(cleaned_sen)
        self.word2id = {word: id for id, word in enumerate(self.vocab)}

    def __getitem__(self, index):

        return self.ds[index], self.labels[index]

    def __len__(self):
        return len(self.ds)


def train_model(train_data,train_label,test_data,test_label):
    train_dataset = UserDataset(train_data,train_label)
    train_loader = DataLoader(train_dataset, batch_size=1,  shuffle=True)
    dev_loader = DataLoader(UserDataset(test_data,test_label), batch_size=1)
    model = TextClassifier(vocab_size=len(train_dataset.vocab), embedding_size=128, word2id=train_dataset.word2id)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    print(len(train_data), len(test_data))
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        # Train step
        bar = ProgressBar(maxval=len(train_data))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        bar.start()
        for sentence, label in train_loader:
            i += 1

            ground_truth += label
            output,_ = model([sentence])
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for sentence, label in dev_loader:

                ground_truth += label
                output,_ = model([sentence])
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')
            if save_model:
                if F1 > max_F1:
                    max_F1 = F1
                    torch.save(model, model_path)
        break




def get_similarity(train_data,train_label,test_data,test_label,similarity_threshold = 0.5):

    dev_loader = DataLoader(UserDataset(train_data+test_data,train_label+test_label), batch_size=1)

    model = torch.load(model_path)

    model.to(device)

    total_set = {}
    valid_set = {}

    model.eval()
    with torch.no_grad():
        for sentence, label in dev_loader:

                embedding, sorted_len, reversed_indices = model.embedding_lookup([sentence], len(sentence))
                packed_embed = pack_padded_sequence(embedding, sorted_len, batch_first=True)

                each, (h, c) = model.sentenceEncoder2(packed_embed)

                for word in range(0, len(sentence)):
                    for another in range(0, len(sentence)):
                        w1 = sentence[word]
                        w2 = sentence[another]
                        s = torch.cosine_similarity(each.data[word].view(1, -1), each.data[another].view(1, -1), dim=1)
                        key = '' + w1 + ',' + w2
                        if key not in total_set.keys():
                            total_set[key] = 1
                        else:
                            total_set[key] += 1
                        if s.data > similarity_threshold:
                            if key not in valid_set.keys():
                                valid_set[key] = 1
                            else:
                                valid_set[key] += 1


    f = open('lstm_semantic.pkl', 'wb')
    pickle.dump(valid_set, f)
    f.close()

"""
if __name__ == '__main__':
    train_model(train_data, test_data)
    get_similarity(train_data, test_data)
"""