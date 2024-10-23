import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from keras.preprocessing.sequence import pad_sequences


def letter2tensor(letter): # 원 핫 인코딩 -> tensor 변환 #letter = 이름 1개
  res = [0]

  for char in all_letters:
    if char == letter:
      res.append(1)
    else:
      res.append(0)
  assert len(res) == len(all_letters) + 1
  assert res[0] == 0
  return torch.tensor(res)

# y.append(idx2tensor(all_categories.index(lang),n_categories))
# all_categories : 언어
# n_categories : len(all_categories)
def idx2tensor(idx, N): # 인덱스 -> tensor 변환 # (언어의 인덱스, 언어의 총 갯수)
  res = []

  for i in range(N):
    if i == idx:
      res.append(1)
    else:
      res.append(0)
  return torch.tensor(res)

def word2tensor(word, max_length = 10): # word -> tensor 변환
  res = torch.zeros(max_length, len(all_letters) + 1, dtype = torch.float32)

  for idx, char in enumerate(word): # word = name
    res[idx] = letter2tensor(char)
  # res: len(word), len(all_letters)
  for idx in range(max_length - len(word)):
    res[len(word) + idx] = torch.tensor([1] + [0 for _ in range(len(all_letters))])
  return res

# step 1. 데이터 준비
def prepare_data(batch_size = 32):
  files = glob.glob('names/*.txt')
  assert len(files) == 18

  category_names = {}
  all_categories = []

  n_letters = len(all_letters)
  assert n_letters == 26

  for file in files:
    with open(file) as f:
      names = f.read().strip().split('\n')

    lang = file.split('/')[-1].split('.')[0]

    # if lang in ['Korean', 'Czech']:
    all_categories.append(lang)

    names = [n.lower() for n in names[:10]]
    names = [''.join([c for c in n if c in all_letters]) for n in names]
    category_names[lang] = names # {언어 : 이름}

  print("all_categories :", all_categories)
  n_categories = len(all_categories)
  print("n_categories : ", n_categories) # 18
  # for lang, names in category_names.items():
  #     # print(len(names))
  x = []
  y = []
  max_length = 0

  for lang, names in category_names.items():
    for name in names:
      if len(name) > max_length:
        max_length = len(name)

  for lang, names in category_names.items():
    for name in names:
      # x.append(list(name))
      x.append(word2tensor(name, max_length = max_length)) # word2tensor(name): len(word), len(all_letters)
      y.append(all_categories.index(lang))

  # print('len(x) : ', len(x)) # == 20074
  # print('len(y) : ', len(y)) # == 20074

  x_array =  np.array(x)
  y_array =  np.array(y)
  # x_tensor: len(names), max_name_len, len(all_letters)
  X_tensor = torch.tensor(x_array, dtype = torch.float32)
  y_tensor = torch.tensor(y_array, dtype = torch.long)

  print("X_tensor.shape() : ", X_tensor.size())
  print("y_tensor.shape() : ", y_tensor.size())
  dataset = TensorDataset(X_tensor,y_tensor)
  dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
  return dataloader, all_categories

def plot_loss_history(loss_history):
  plt.plot(range(1, len(loss_history)+1), loss_history)
  plt.show()

# rnn = RNN(n_letters, n_hidden, n_categories) # 26, 32, 18
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size): # 26, 32, 18
    super(RNN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.i2h = nn.Linear(input_size, hidden_size)
    self.h2h = nn.Linear(hidden_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim = 1)
    self.optimizer = optim.Adam
    self.loss = torch.nn.NLLLoss()

  def forward(self, input, hidden):
    # input: (batch_size, input_size)
    # hidden: (hidden_size, )

    # self.i2h(input): (batch_size, hidden_size)
    # self.h2h(hidden): (1, hidden, )
    # hidden: F.tanh( (batch_size, hidden_size) + (1, hidden_size,)): (batch_size, hidden_size)
    batch_size, input_size = input.size()
    assert input_size == self.input_size
    assert hidden.size(1) == self.hidden_size

    hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
    output = self.h2o(hidden) # output: (batch_size, output_size)
    output = self.softmax(output) # output: (batch_size, output_size)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)

  def train_model(self, train_data, learning_rate = 0.01, epochs = 100):
    optimizer = self.optimizer(self.parameters(), lr = learning_rate)
    loss_history = []

    for epoch in range(epochs):
      for idx, (x, y) in enumerate(train_data):
        hidden = self.initHidden()
        # for char in x:
        #   output, hidden = self(char, hidden)
        #   #print(output)

        for i in range(x.size(1)):
          output, hidden = self(x[:,i], hidden)

        # loss = self.loss(output.view(-1, n_categories), y)
        loss = self.loss(output, y)

        if idx % 100 == 0:
          loss_history.append(loss.item())
        #print(torch.mean(loss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


      print(f'Epoch {epoch}, Loss: {loss.item()}')
    plot_loss_history(loss_history)
    return loss_history

def predict_nationality(model, word, all_categories): # predict_nationality(rnn, 'ang')
  hidden = model.initHidden()

  word = word2tensor(word) # res
  output, hidden = model(word, hidden)
  # print("output :", output)
  # print('output.shape : ', output.shape)
  # print('output.argmax : ', torch.argmax(output[1]))
  # print("lang : ", all_categories[torch.argmax(output[1]).item()])
  return all_categories[torch.argmax(output[1]).item()]

all_letters = 'abcdefghijklmnopqrstuvwxyz'
n_letters = len(all_letters) + 1
n_hidden = 32
n_categories = 18

# 데이터 준비 및 모델 학습
dataset, all_categories  = prepare_data()

rnn = RNN(n_letters, n_hidden, n_categories) # 26, 32, 18
rnn.train_model(dataset)
# learning_rate_lst = [0.001, 0.0001, 0.0001] # 0.001이 loss가 가장 작음
# for i in (learning_rate_lst):
#   print(f"learning_rate : {i}\n")
#   rnn.train_model(dataset,learning_rate = i, epochs = 20)

predict_nationality(rnn, 'ang', all_categories)
# plot_loss_history(loss_history)