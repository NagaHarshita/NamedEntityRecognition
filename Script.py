import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from collections import Counter
import csv
from sklearn.metrics import accuracy_score

from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import torchtext.vocab as vocab

import numpy as np
import warnings
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')


print('----Reading the data files---')


with open('./data/train', 'r') as f:
    train_file = f.readlines()
train_list = [line.split() for line in train_file if len(line) > 0]
train = pd.DataFrame(train_list, columns=['idx', 'word', 'tag'])
train.dropna(how='all', inplace=True)


with open('./data/dev', 'r') as f:
    dev_file = f.readlines()
dev_list = [line.split() for line in dev_file if len(line) > 0]
dev = pd.DataFrame(dev_list, columns=['idx', 'word', 'tag'])
dev.dropna(how='all', inplace=True)

with open('./data/test', 'r') as f:
    test_file = f.readlines()
test_list = [line.split() for line in test_file if len(line) > 0]
test = pd.DataFrame( test_list, columns=['idx', 'word'])
test = test.dropna().reset_index(drop=True)




if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print('---Creating the vocabulary---')
word2idx = {word: i+1 for i, word in enumerate(set(train['word']))}
label2idx = {label: i+1 for i, label in enumerate(set(train['tag']))}

train['labelidx'] = train['tag'].apply(lambda x: label2idx[x])
dev['labelidx'] = dev['tag'].apply(lambda x: label2idx[x])

def createDataset(df):
    df['word2idx'] = df['word'].apply(lambda x: word2idx[x] if x in word2idx else 0)
    df['label2idx'] = df['tag'].apply(lambda x: label2idx[x])

    row_numbers = (df[df['idx']=='1'].index.tolist())
    row_numbers.append(row_numbers[-1]+1)
    new_df = []

    for i in range(len(row_numbers)-1):
        sentence = list(df.loc[row_numbers[i]:row_numbers[i+1]-1, 'word'])
        tags = list(df.loc[row_numbers[i]:row_numbers[i+1]-1, 'tag'])
        word_idxs = list(df.loc[row_numbers[i]:row_numbers[i+1]-1, 'word2idx'])
        label_idxs = list(df.loc[row_numbers[i]:row_numbers[i+1]-1, 'label2idx'])
        idx = list(df.loc[row_numbers[i]:row_numbers[i+1]-1, 'idx'])
        row = [idx, sentence, tags, word_idxs, label_idxs]
        new_df.append(row)

    new_df = pd.DataFrame(new_df, columns=['idx', 'sentence', 'tags', 'word_idxs', 'label_idxs'])
    return new_df

print('---Creating the sentences dataframe---')
ner_train = createDataset(train)
ner_dev = createDataset(dev)

class NerDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get word and label at the current index
        words = self.df.loc[idx, 'word_idxs']
        labels = self.df.loc[idx, 'label_idxs']
        
        # Return the word index and label index as PyTorch tensors
        return  torch.tensor(words), torch.tensor(labels), len(words)
    
    def collate_fn(self, batch):
        # Pad the variable-length sequences in batch with 0s
        words = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        lengths = [b[2] for b in batch]
        padded_words = pad_sequence(words, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return padded_words, padded_labels, lengths
    


ner_dev['predict_1'] = None
ner_dev['predict_2'] = None


class BLSTM(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim):
        super(BLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.linear_output_dim = linear_output_dim

        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=self.embedding_dim,padding_idx=23625)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers, bidirectional=True, dropout=self.lstm_dropout)
        self.linear = nn.Linear(in_features=self.lstm_hidden_dim * 2, out_features=self.linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(in_features=self.linear_output_dim, out_features=NUM_CLASSES)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        lstm_output, _ = self.lstm(embedded)
        linear_output = self.linear(lstm_output)
        elu_output = self.elu(linear_output)
        output = self.classifier(elu_output)
        return output
    

class BLSTM_Glove(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim):
        super(BLSTM_Glove, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.linear_output_dim = linear_output_dim

        self.glove = vocab.GloVe(name='6B', dim=100)
        
        # Create the embedding layer using the GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(self.glove.vectors)
        self.embedding.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers, bidirectional=True, dropout=self.lstm_dropout)
        self.linear = nn.Linear(in_features=self.lstm_hidden_dim * 2, out_features=self.linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(in_features=self.linear_output_dim, out_features=10)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        lstm_output, _ = self.lstm(embedded)
        linear_output = self.linear(lstm_output)
        elu_output = self.elu(linear_output)
        output = self.classifier(elu_output)
        return output
    
print("--------------TASK 1-------------------")  

NUM_EPOCHS = 150
BATCH_SIZE = 32
VOCAB_SIZE = 23700
NUM_CLASSES = 10

print('---Creating the dataloaders---')
train_ner_dataset = NerDataset(ner_train)
dev_ner_dataset = NerDataset(ner_dev)
train_loader = DataLoader(train_ner_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=train_ner_dataset.collate_fn)
dev_loader = DataLoader(dev_ner_dataset, batch_size=BATCH_SIZE, collate_fn=dev_ner_dataset.collate_fn)

# Initialize the model
BiLSTM_model = BLSTM(
    embedding_dim=100,
    lstm_hidden_dim=256,
    lstm_layers=1,
    lstm_dropout=0.33,
    linear_output_dim=128
).to(device)

print('Started training model 1')

class_weights = compute_class_weight('balanced', classes=np.unique(train['labelidx']), y=train['labelidx'])
class_weights_tensor = torch.FloatTensor(class_weights)
class_weights_tensor = class_weights_tensor.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(BiLSTM_model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)

for epoch in range(NUM_EPOCHS):
    train_f1, valid_f1 = 0.0, 0.0
    train_acc, valid_acc = 0.0, 0.0
    
    BiLSTM_model.train() 
    for ind, (data, target, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = BiLSTM_model(data)
        loss = criterion(output.permute(0, 2, 1), target)
        
        correct = 0
        for k in range(len(lengths)):
            correct += np.count_nonzero(output[k].argmax(dim=1)[:lengths[k]] == target[k][:lengths[k]])
        
        train_acc += correct
        loss.backward()
        optimizer.step()
    
    # Update the learning rate
    scheduler.step()
    
    BiLSTM_model.eval()
    for idx, (data, target, lengths) in enumerate(dev_loader):
        data = data.to(device)
        target = target.to(device)
        output = BiLSTM_model(data)

        correct = 0
        for k in range(len(lengths)):
            correct += np.count_nonzero(output[k].argmax(dim=1)[:lengths[k]] == target[k][:lengths[k]])

        valid_acc += correct 
        
        for k in range(len(lengths)):
            ner_dev.at[BATCH_SIZE * idx + k, 'predict_1'] = list(output[k].argmax(dim=1)[:lengths[k]].cpu().numpy().flatten())
        

    train_acc = train_acc / len(train)
    valid_acc = valid_acc / len(dev)

    print('Epoch: {} Training Acc: {:.6f} Validation Acc: {:.6f}'.format(epoch + 1, train_acc, valid_acc))
    print('----------------------------------')

print('Saving the model 1')
torch.save(BiLSTM_model, 'bilstm1.pt')


idx2label = {v: k for k, v in label2idx.items()}
ner_dev['predict_labels_1'] = ner_dev['predict_1'].apply(lambda x: [idx2label[i] if i in idx2label else 'unk' for i in x  ])

print('Saving the output in dev1.out')
with open('dev1.out', 'w') as f:
    for index, row in ner_dev.iterrows():
        for i in row['idx']:
            f.write(i+" "+str(row['sentence'][int(i)-1])  + " " + str(row['predict_labels_1'][int(i)-1]) + "\n")
        f.write("\n")
        

model1 = torch.load('bilstm1.pt')
test['wordidx'] = test['word'].apply(lambda x: word2idx[x] if x in word2idx else 0)
test['predict_1'] = None

for i, row in test.iterrows():
    output = model1(torch.tensor([row['wordidx']]))
    ind = int(output.argmax(dim=1)[0])
    label = idx2label[ind] if ind in idx2label else 'unk'
    test.at[i, 'predict_1'] = label
    
print('Saving the output in test1.out')
with open('test1.out', 'w') as f:
    for index, row in test.iterrows():
        if(index!=0 and row['idx']=='1'):
            f.write("\n"+row['idx']+" "+str(row['word'])+ " " + str(row['predict_1'])+"\n")
        else:
            f.write(row['idx']+" "+str(row['word'])+ " " + str(row['predict_1'])+"\n")
            

print("--------------TASK 2-------------------")        
NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 10

print('---Creating the dataloaders---')
train_ner_dataset = NerDataset(ner_train)
dev_ner_dataset = NerDataset(ner_dev)
train_loader = DataLoader(train_ner_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=train_ner_dataset.collate_fn)
dev_loader = DataLoader(dev_ner_dataset, batch_size=BATCH_SIZE, collate_fn=dev_ner_dataset.collate_fn)

# Initialize the model
BiLSTM_Glove_model = BLSTM_Glove(
    embedding_dim=100,
    lstm_hidden_dim=256,
    lstm_layers=1,
    lstm_dropout=0.33,
    linear_output_dim=128
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(BiLSTM_Glove_model.parameters(), lr=1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
print('Started training model 2')

for epoch in range(NUM_EPOCHS):
    train_f1, valid_f1 = 0.0, 0.0
    train_acc, valid_acc = 0.0, 0.0
    
    BiLSTM_Glove_model.train() 
    for ind, (data, target, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = BiLSTM_Glove_model(data)
        # print(output.shape, target.shape)
        loss = criterion(output.permute(0, 2, 1), target)
        
        correct = 0
        for k in range(len(lengths)):
            correct += np.count_nonzero(output[k].argmax(dim=1)[:lengths[k]] == target[k][:lengths[k]])
        
        train_acc += correct
        loss.backward()
        optimizer.step()
    
    # Update the learning rate
    # scheduler.step()
    BiLSTM_Glove_model.eval()
    for idx, (data, target, lengths) in enumerate(dev_loader):
        data = data.to(device)
        target = target.to(device)
        output = BiLSTM_Glove_model(data)

        correct = 0
        for k in range(len(lengths)):
            correct += np.count_nonzero(output[k].argmax(dim=1)[:lengths[k]] == target[k][:lengths[k]])

        valid_acc += correct 
        
        for k in range(len(lengths)):
            ner_dev.at[BATCH_SIZE * idx + k, 'predict_2'] = list(output[k].argmax(dim=1)[:lengths[k]].cpu().numpy().flatten())
        

    train_acc = train_acc / len(train)
    valid_acc = valid_acc / len(dev)

    print('Epoch: {} Training Acc: {:.6f} Validation Acc: {:.6f}'.format(epoch + 1, train_acc, valid_acc))
    print('----------------------------------')
    
print('Saving the model 2')
torch.save(BiLSTM_Glove_model, 'bilstm2.pt')

ner_dev['predict_labels_2'] = ner_dev['predict_2'].apply(lambda x: [idx2label[i] if i in idx2label else 'unk' for i in x  ])

print('Saving the output in dev2.out')

with open('dev2.out', 'w') as f:
    for index, row in ner_dev.iterrows():
        for i in row['idx']:
            f.write(i+" "+str(row['sentence'][int(i)-1]) + " " + str(row['predict_labels_2'][int(i)-1]) + "\n")
        f.write("\n")
        

        
model2 = torch.load('bilstm2.pt')

test['wordidx'] = test['word'].apply(lambda x: word2idx[x] if x in word2idx else 0)
test['predict_2'] = None

for i, row in test.iterrows():
    output = model2(torch.tensor([row['wordidx']]))
    ind = int(output.argmax(dim=1)[0])
    label = idx2label[ind] if ind in idx2label else 'unk'
    test.at[i, 'predict_2'] = label
    
print('Saving the output in test2.out')
with open('test2.out', 'w') as f:
    for index, row in test.iterrows():
        if(index!=0 and row['idx']=='1'):
            f.write("\n"+row['idx']+" "+str(row['word'])+ " " + str(row['predict_2'])+"\n")
        else:
            f.write(row['idx']+" "+str(row['word'])+ " " + str(row['predict_2'])+"\n")