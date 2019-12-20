import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import sys


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        #self.grus = []
        #for i in range(2, 9):
        #    self.grus.append(nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True))
        self.gru2 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru3 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru4 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru5 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru6 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru7 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru8 = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)



    def forward(self, src, gruIdx, hidden=None):
        embedded = self.embed(src)
        if gruIdx == 2:
            outputs, hidden = self.gru2(embedded, hidden)
        elif gruIdx == 3:
            outputs, hidden = self.gru3(embedded, hidden)
        elif gruIdx == 4:
            outputs, hidden = self.gru4(embedded, hidden)
        elif gruIdx == 5:
            outputs, hidden = self.gru5(embedded, hidden)
        elif gruIdx == 6:
            outputs, hidden = self.gru6(embedded, hidden)
        elif gruIdx == 7:
            outputs, hidden = self.gru7(embedded, hidden)
        elif gruIdx == 8:
            outputs, hidden = self.gru8(embedded, hidden)
        else:
            print("??????")
            print(gruIdx)
            sys.exit(1)
        #self.gru = self.grus[gruIdx]
        #outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)

        #self.grus = []
        #for i in range(2, 9):
        #    self.grus.append(nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout))
        self.gru2 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru3 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru4 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru5 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru6 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru7 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.gru8 = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)

        
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs, gruIdx):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)

        #self.gru = self.grus[gruIdx]
        #output, hidden = self.gru(rnn_input, last_hidden)
        if gruIdx == 2:
            output, hidden = self.gru2(rnn_input, last_hidden)
        elif gruIdx == 3:
            output, hidden = self.gru3(rnn_input, last_hidden)
        elif gruIdx == 4:
            output, hidden = self.gru4(rnn_input, last_hidden)
        elif gruIdx == 5:
            output, hidden = self.gru5(rnn_input, last_hidden)
        elif gruIdx == 6:
            output, hidden = self.gru6(rnn_input, last_hidden)
        elif gruIdx == 7:
            output, hidden = self.gru7(rnn_input, last_hidden)
        elif gruIdx == 8:
            output, hidden = self.gru8(rnn_input, last_hidden)
        else:
            print("??????")
            print(gruIdx)
            sys.exit(1)

        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.lang = lang

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        srcLen = src.shape[0]
        trgLen = trg.shape[0]

        encoder_output, hidden = self.encoder(src,srcLen-1)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output, trgLen-1)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
            #print_tensor(output)
        return outputs

