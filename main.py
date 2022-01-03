import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import argparse

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from torch.nn.utils.rnn import pad_sequence


from helper import init_embedding, get_data_sequences, create_embedding
from dataset import Dataset, get_data_loaders
from model import SelfNet


parser = argparse.ArgumentParser(description='SelfNet')

parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')

parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lr', default=5.16e-05, type=float,
                    metavar='LR', help='learning rate for the model')

parser.add_argument('--hidden_size_lstm', default=1024, type=int,
                    help='hidden size of lstm')

parser.add_argument('--hidden_size_linear', default=128, type=int,
                    help='hidden size of linear layer')

parser.add_argument('--seq_len', default=56, type=int,
                    help='sequence lenght of input text')

parser.add_argument('--clip', default=0.218, type=float,
                    help='gradient clipping')

parser.add_argument('--dropout', default=0.198, type=float,
                    help='dropout value')

parser.add_argument('--num_layers', default=1, type=int,
                    help='number of lstm layers')

parser.add_argument('--lstm_bidirectional', default=False, type=bool,
                    help='bidirectional lstm')

parser.add_argument('--fasttext_embed_file', type=str, default='new_hing_emb',
                    help='path to fasttext embedding file')

parser.add_argument('--train_dir', type=str, default='train.csv',
                    help='path to train file')

parser.add_argument('--valid_dir', type=str, default='valid.csv',
                    help='path to validation file')

parser.add_argument('--test_dir', type=str, default='test.csv',
                    help='path to test file')

parser.add_argument('--checkpoint_dir', type=str, default='selfnet.pt',
                    help='path to the saved model')

parser.add_argument('--test', type=bool, default=False,
                    help='testing the model')

args = parser.parse_args()


def get_accuracy(preds, labels):
    
    total_acc = 0.0
    
    for i in range(len(labels)):
        
        if labels[i] == preds[i]:
            total_acc+=1.0
            
    return total_acc / len(labels)


def train(config, train_dir, test_dir, valid_dir, fasttext_embed_file, test_mode = False, checkpoint_dir=None):
    
  if not os.path.exists(train_dir):
    print("Train file not found")
            
  if not os.path.exists(test_dir):
    print("Test file not found")
    
  if not os.path.exists(valid_dir):
    print("Valid file not found")
    
  if not os.path.exists(fasttext_embed_file):
    print("Fastext embedding file not found")            

  #initialize embedding matrix
  embedding_matrix, vocab_size, vocab_dim, word2idx = init_embedding(fasttext_embed_file)
  
  # load sequences
  train_sequences, train_labels, test_sequences, test_labels, val_sequences, val_labels, word2idx_updated = get_data_sequences(train_dir, valid_dir, test_dir, word2idx)

  # get final embedding matrix
  final_embedding_matrix, final_word2idx = create_embedding(embedding_matrix, word2idx_updated, vocab_size, vocab_dim)

  # padded sequences
  train_padded_sequences = pad_sequence(train_sequences, batch_first= True, padding_value=final_word2idx['<PAD>'])

  val_padded_sequences = pad_sequence(val_sequences, batch_first= True, padding_value=final_word2idx['<PAD>'])

  test_padded_sequences = pad_sequence(test_sequences, batch_first= True, padding_value=final_word2idx['<PAD>'])

  # get dataset
  train_dataset = Dataset(train_padded_sequences, train_labels)

  val_dataset = Dataset(val_padded_sequences, val_labels)

  test_dataset = Dataset(test_padded_sequences, test_labels)

  # get dataloader
  dataloaders = get_data_loaders(train_dataset, val_dataset, test_dataset, config)


  # final vocab_size
  vocab_size = len(final_word2idx)

  # embedding dimension of words
  embed_dim = vocab_dim

  num_class = 2
    
  EPOCHS = args.epochs


  #get model
  model = SelfNet(vocab_size, embed_dim, final_embedding_matrix, final_word2idx, config["hidden_size_lstm"], config["hidden_size_linear"], config["num_layers"], config["seq_len"], config["lstm_bidirectional"], num_class, config["dropout"])


  #check gpu
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model.to(device)

  #optimizer
  optimizer = Adam(model.parameters(), lr=config["lr"], eps=1e-8)

  #Loss function
  criterion = nn.CrossEntropyLoss()


  if not test_mode:
    
      best_valid_f1 = 0.00

      for epoch in range(0, EPOCHS):
        print('-'*50)
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))

        for phase in ['Train', 'Val']:

          batch_loss = 0.0000   #live loss
          batch_acc = 0.0000   #live accuracy

          y_true = []
          y_pred = []

          if phase == 'Train':
              model.train()
          else:
              model.eval()

          with tqdm(dataloaders[phase], unit="batch", desc=phase) as tepoch:

            for idx, batch in enumerate(tepoch):

              labels = batch["label"].to(device)
              text = batch["token"].to(device)

              output = model(text)

              loss = criterion(output, labels)

              if phase == 'Train':

                  #zero gradients
                  optimizer.zero_grad() 

                  # Backward pass  (calculates the gradients)
                  loss.backward()   

                  # gradient clipping
                  nn.utils.clip_grad_norm_(model.parameters(), config["clip"])    

                  optimizer.step()             # Updates the weights    

              batch_loss += loss.item()

              _, preds = output.data.max(1)
              y_pred.extend(preds.tolist())
              y_true.extend(labels.tolist())

              batch_acc = get_accuracy(y_pred, y_true)

              tepoch.set_postfix(loss = batch_loss/(idx+1), accuracy = batch_acc )

            pre = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            print("F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}.".format(f1, pre, recall))

            if phase == 'Val':
              if f1 > best_valid_f1:
                  best_valid_f1 = f1
                  torch.save(model.state_dict(), checkpoint_dir)
                  print('Best model Saved!')

            print()
            
  else:
    
      if not os.path.exists(checkpoint_dir):
            print("Model file not found")
        
      model.load_state_dict(torch.load(checkpoint_dir))
    
      print("Model loaded successfully")

      print("-"*50)

      print("Inference:")


      test(model, dataloaders, device, optimizer, criterion)
  
  
def test(model, dataloaders, device, optimizer, criterion):

  batch_loss = 0.0   #batch loss
  batch_acc = 0.0   #batch accuracy

  y_true = []
  y_pred = []

  # set the model to evaluation mode            
  model.eval()

  phase = 'Test'
          
  with tqdm(dataloaders[phase], unit="batch", desc=phase) as tepoch:
    for idx, batch in enumerate(tepoch):
      labels = batch["label"].to(device)
      text = batch["token"].to(device)
      
      with torch.no_grad():
        output = model(text)
        loss = criterion(output, labels)
      
      _, preds = output.data.max(1)
      y_pred.extend(preds.tolist())
      y_true.extend(labels.tolist())

      batch_acc = get_accuracy(y_pred, y_true)
      batch_loss += loss.item()

      tepoch.set_postfix(loss = batch_loss/(idx+1), accuracy = batch_acc )
                
  pre = precision_score(y_true, y_pred, average='macro')
  recall = recall_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  print("")

  print("F1: {:.6f}, Precision: {:.6f}, Recall : {:.6f}".format(f1, pre, recall))


def main():

  fasttext_embed_file = args.fasttext_embed_file
  train_dir = args.train_dir 
  valid_dir = args.valid_dir
  test_dir = args.test_dir
  checkpoint_dir = args.checkpoint_dir

  test_mode = args.test
    
  #hyper parameters 
  config =  {'hidden_size_lstm': args.hidden_size_lstm, 
             'hidden_size_linear': args.hidden_size_linear, 
             'seq_len': args.seq_len, 
             'lr': args.lr, 
             'clip': args.clip,
             'dropout': args.dropout,
             'num_layers': args.num_layers, 
             'batch_size': args.batch_size, 
             'lstm_bidirectional': args.lstm_bidirectional}

  train(config, train_dir=train_dir, test_dir = test_dir, valid_dir = valid_dir, fasttext_embed_file = fasttext_embed_file,  test_mode = test_mode, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":

  main()
