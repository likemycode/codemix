This repo contains codes for the following paper:

How Effective is Incongruity? Implications for Code-mix Sarcasm Detection. <br>
Aditya Shah, Chandresh Kumar Maurya, <i>In Proceedings of the 18th International Conference on Natural Language Processing -  (ACL 2021)</i>.

### Requirements
  Python 3.6 or higher <br>
  Pytorch >= 1.3.0 <br>
  Pytorch_transformers (also known as transformers) <br>
  Pandas, Numpy, Pickle <br>
  Fasttext  <br>

### Download the fasttext embed file:

The fasttext embedding file can be obtained [here](https://drive.google.com/file/d/1L7NBq58JYdXlGjSTwaJrVfZCLllXd0sh/view?usp=sharing)


### Arguments:

```
--epochs:  number of total epochs to run, default=10

--batch-size: train batchsize, default=2

--lr: learning rate for the model, default=5.16e-05

--hidden_size_lstm: hidden size of lstm, default=1024

--hidden_size_linear: hidden size of linear layer, default=128

--seq_len: sequence lenght of input text, default=56

--clip: gradient clipping, default=0.218

--dropout: dropout value, default=0.198

--num_layers: number of lstm layers, default=1

--lstm_bidirectional: bidirectional lstm, default=False

--fasttext_embed_file: path to fasttext embedding file, default='new_hing_emb'

--train_dir: path to train file, default='train.csv'

--valid_dir: path to validation file, default='valid.csv'

--test_dir: path to test file, default='test.csv'

--checkpoint_dir: path to the saved, default='selfnet.pt'

--test: testing the model, default=False
```
### Train

```python main.py```

### Test

```python main.py --test True```
