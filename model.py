import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfMatchingLayer(nn.Module):

    def __init__(self,  seq_length, embed_dim, **kwargs):

      super(SelfMatchingLayer, self).__init__()

      self.seq_length = seq_length
      self.embed_dim  = embed_dim

      self.P = torch.nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))

      self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))

      
    def forward(self, x):  
      
      # input shape: [batch, seq_len, embed_dim]


      #---------------------------------------------#
      # calculate weight vector a = {e_i . P.Q . e_j}
      #---------------------------------------------#

      out = torch.matmul(x,  self.P)   #out shape: [batch, seq_len, embed_dim]

      out = torch.matmul(out, torch.transpose(x, 1, 2))   #out shape: [batch, seq_len, seq_len]

      # return out

      out = F.gelu(out)         # apply non linear activation

      #------------------------------------#
      # take row wise mean and apply softmax
      #------------------------------------#
      out = torch.mean(out, 2)  #out shape: [batch, seq_len, seq_len]

      out = torch.softmax(out, 0)     #out shape: [batch, seq_len, seq_len]

      out = out.unsqueeze(1)          #out shape: [batch, 1, seq_len]

      #-------------------------------------------#
      # calculate weighted embedding of every word
      #-------------------------------------------#
      out = torch.matmul(out, x)

      out = out.squeeze(1)

      return out      #out shape: [batch, seq_len]


class SelfNet(nn.Module):

    def __init__(self, vocab_size, embed_dim, embedding_matrix, word2idx, hidden_size_lstm, hidden_size_linear, num_layer, seq_len, bidirectional, num_class, dropout):
        super(SelfNet, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = word2idx['<PAD>'])
        self.embedding.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})
        self.embedding.weight.requires_grad = True

        self.selfnet_layer = SelfMatchingLayer(seq_len, embed_dim)

        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size_lstm, num_layers = num_layer, dropout = dropout, bidirectional = bidirectional, batch_first = True )

        if bidirectional:
            self.fc1 = nn.Linear(2* hidden_size_lstm + embed_dim , hidden_size_linear)
        else:
            self.fc1 = nn.Linear(hidden_size_lstm + embed_dim , hidden_size_linear)
            
        self.fc2 = nn.Linear(hidden_size_linear, num_class)
        
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)



    def forward(self, input):
        
        input = input[:, :self.seq_len]

        embedded = self.embedding(input)  #out shape = [batch, seq_len, embed_dim] 

        selfmatch_output = self.selfnet_layer(embedded)  #out shape = [batch, seq_len] 

        lstm_out, _ = self.lstm(embedded)     

        lstm_out = lstm_out[:, -1, :]      #out shape = [batch, 2 * hidden_size]      

        concat = torch.cat((selfmatch_output, lstm_out), 1)     #out shape = [batch, 2 * hidden_size_lstm + embed_dim ]      

        linear_out = self.dropout(F.relu(self.fc1(concat)))     #out shape = [batch, hidden_size_linear]      

        final_out = self.fc2(linear_out)     #out shape = [batch, 2]      

        return final_out