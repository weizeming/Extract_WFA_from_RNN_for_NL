import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import time
from utils import dev
from dataset import dataset
from path import Path

class RNN(nn.Module):
    def __init__(self, vocab_size, output_dim, embedding_dim=100, hidden_dim=64, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # maybe set no grad, if pretrained
        self.rnn = nn.LSTM(embedding_dim, hidden_dim) #num_layers=1, bidirectional=False
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = dev()

    def forward(self, x):
        '''
        Embedding => Dropout => LSTM => Dropout => Linear
        x torch.Size([25, 50])
        50 instance, 25 length each, embedded into a 100-dim vector
        input  torch.Size([25, 50, 100])
        |out torch.Size([25, 50, 32]) 
        |hid torch.Size([1, 50, 32])
        |cell torch.Size([1, 50, 32])
        hid' torch.Size([50, 32])
        output_sequence[0]  torch.Size([50, 7])
        return  torch.Size([50, 7])
        '''
        self.input = self.dropout(self.embedding(x)) #Embedded and Dropout. x:[max_length, instance_num]
        self.rnn_out, (self.rnn_hid, self.rnn_cell) = self.rnn(self.input)
        self.rnn_hid = self.dropout(self.rnn_hid[0, :, :]) #3-dim to 2-dim, then dropout
        self.output_sequence = []
        for i in range(len(self.rnn_out)):  #foreach input, append a output
            self.output_sequence.append(self.fc(self.rnn_out[i]))
        return self.fc(self.rnn_hid)

    def clear_output_sequence(self):
        self.output_sequence = []
    
    def runtime_predict(self):
        return self.output_sequence #[steps, batch_size, internal_data]


    def runtime_predict_size(self):
        length = len(self.output_sequence)
        template_shape = self.output_sequence[0].shape
        return length, template_shape


def process_text(batch, model):
    text = batch[0].clone()
    text = torch.transpose(text, 0, 1)
    text = text.to(dev())
    return text

def process_label(batch, model):
    label = np.array(batch[1])
    label = torch.from_numpy(label)        
    label = label.to(model.device)
    return label

def process_batch(batch, model):
    return process_text(batch, model), process_label(batch, model)

def train_epoch(model, data, optimizer, scheduler):
    epoch_loss, epoch_correct, epoch_total_num = 0, 0, 0

    model.train()

    for idx, batch in enumerate(data):
        text, label = process_batch(batch, model)

        output = model(text)
        prediction = output.argmax(dim=1)

        loss = F.cross_entropy(output, label, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * label.shape[0]
        epoch_correct += (prediction == label).float().sum()
        epoch_total_num += label.shape[0]
    
    scheduler.step()

    return epoch_loss/epoch_total_num, float(epoch_correct/epoch_total_num)


def eval(model, data):
    epoch_loss, epoch_correct, epoch_total_num = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch in data:
            text, label = process_batch(batch, model)

            output = model(text)
            prediction = output.argmax(dim=1)

            loss = F.cross_entropy(output, label, reduction='mean')

            epoch_loss += loss.item() * label.shape[0]
            epoch_correct += (prediction == label).float().sum()
            epoch_total_num += label.shape[0]
    return epoch_loss/epoch_total_num, float(epoch_correct/epoch_total_num)


def train(model, train_data, test_data, optimizer, scheduler, num_epochs):
    start_time = time.time()
    for idx in range(num_epochs):
        tmp_time = time.time()
        train_loss, train_accuracy = train_epoch(model, train_data, optimizer, scheduler)
        test_loss, test_accuracy = eval(model, test_data)
        used_time = time.time() - tmp_time
        print(f'epoch{idx+1}:')
        print(f'train loss:{train_loss:.3f} train accuracy:{train_accuracy*100:.2f}%')
        print(f'test loss:{test_loss:.3f}  test accuracy:{test_accuracy*100:.2f}%')
        print(f'use time:{used_time:.1f}')
        print('-'*100)
    total_time = time.time() - start_time
    print(f'Train finished. Time consuming:{total_time:.1f}')

if __name__=='__main__':
    DATASET = 'toxic'
    BATCH_SIZE = 1024
    train_dataset = dataset(DATASET,True)
    test_dataset = dataset(DATASET,False)

    model = RNN(len(train_dataset.vocab), train_dataset.classes)
    model.to(model.device)

    train_data = train_dataset.batch(BATCH_SIZE)
    test_data = test_dataset.batch(BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    train(model, train_data, test_data, optimizer, scheduler, 100)

    torch.save(model, Path+DATASET+'_model.pth')