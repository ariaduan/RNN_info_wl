import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np

import re
value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = open('test_sentences.txt','rb')
outf = open('log_20','a',encoding = 'utf-8')

#vocabulary
data_set = []
vocab = dict()
for line in data:
	line = line.decode('utf-8').split()
	tmp = ["BOS"]
	for i in range(len(line)):
		if line[i].isdigit() or value.match(line[i]):
			line[i] = 'NUM'
		if line[i] in vocab:
			vocab[line[i]] += 1
		else:
			vocab[line[i]] = 1
		tmp.append(line[i])
	data_set.append(tmp)

#frequency
outf.write('freq\nword\tlength\tfrequency\n')
for i in vocab:
	outf.write(i + '\t' + str(len(i)) + '\t' + str(vocab[i]) + '\n')

#RNN for information calculation
class RNN(nn.Module):
	def __init__(self, hidden_size, input_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim = 1)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1,1,-1)
		output, hidden = self.gru(embedded, hidden)
		tmp = self.out(output[0])
		output = self.softmax(tmp)#logsoftmax
		return output, hidden, tmp

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device = device)

#word2index & index2word
word2index = {"BOS" : 0}
index2word = ["BOS"]
tmp = 1
for i in vocab:
	word2index[i] = tmp
	index2word.append(i)
	tmp += 1

#train, predict the next word using all the previous words(hidden_layer)
def train(model, epoch):
	criterion = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr = 20)
	for i in range(epoch):
		for line in data_set:
			hidden = model.initHidden()
			optimizer.zero_grad()
			loss = 0
			ix = word2index[line[0]]
			input = torch.tensor([[ix]], device = device)
			output, hidden, tmp = model(input, hidden)
			#print(input)
			#print(output)
			#print(out)
			for j in range(1, len(line)):
				ix = word2index[line[j]]
				input = torch.tensor([[ix]], device = device)
				tmp = criterion(output, input[0])
				loss += tmp
				output, hidden, tmp = model(input, hidden)
			#print(loss.item() / len(line))
			outf.write('epoch' + str(epoch) + '\t' + 'line' + str(j) + '\t' + str(loss.item() / len(line)) + '\n')
			loss.backward()
			optimizer.step()

input_size = len(index2word)
output_size = input_size
model = RNN(650, input_size, output_size)
outf.write('loss\n')
train(model, 1)

#calculate the information volume perword
info = {}
for line in data_set:
	hidden = model.initHidden()
	ix = word2index[line[0]]
	input = torch.tensor([[ix]], device = device)
	output, hidden, out = model(input, hidden)
	for j in range(1, len(line)):
		ix = word2index[line[j]]
		input = torch.tensor([[ix]], device = device)
		output, hidden, tmp = model(input, hidden)
		#print(tmp)
		tmp = tmp.squeeze().exp()
		print(tmp)
		total = sum(tmp)
		p = tmp[ix] / total
		#print(p)
		'''
		out = output.squeeze().exp()#softmax
		'''
		'''
		total = tmp.squeeze().exp()#softmax
		print('total',total)
		out = [float(total[i]) / float(sum(total)) for i in range(total.size()[0])]
		'''
		'''
		print('output',output)
		print('out',out)
		print('sum ',sum(out))
		print('sum2',sum(output[0]))
		'''
		'''
		if line[j] in info:
			info[line[j]].append(float(out[ix]) / float(sum(out)))
		else:
			info[line[j]] = [float(out[ix]) / float(sum(out))]
		'''
		'''
		print(output.squeeze()[ix])
		if line[j] in info:
			info[line[j]].append(float(output.squeeze()[ix].exp()))
		else:
			info[line[j]] = [float(output.squeeze()[ix].exp())]
		#print(output.squeeze()[ix].exp())
		'''
		if line[j] in info:
			info[line[j]].append(float(p))
		else:
			info[line[j]] = [float(p)]
outf.write('info\nword\tlength\tinfomation_volume\n')
for i in info:
	outf.write(i + '\t' + str(len(i)) + '\t' + str(-float(sum(np.log(np.array(info[i])))) / len(info[i])) + '\n')

data.close()
outf.close()