import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    def __init__(self, sequences, labels):
        self.labels = labels
        self.sequences = sequences

    def __getitem__(self, idx):
        sample = {}
        sequence = self.sequences[idx]
        label = torch.tensor(self.labels[idx])

        try:
            sample["label"] = label
            sample["token"] = sequence
        except Exception as e:
            print(e)
        
        return sample
    
    def __len__(self):
        return len(self.labels)
        


def get_data_loaders(train_dataset, val_dataset, test_dataset, config):
    
    """
    Creates Pytorch Data loaders
    """

    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size= config['batch_size'],
      pin_memory=True,
      num_workers=2,
      shuffle=True,
      drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size= config['batch_size'],
      pin_memory=True,
      num_workers=2,
      shuffle=True,
      drop_last=True
    )

    ## For testing
    test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=config['batch_size'],
      pin_memory=True,
      num_workers=2,
      shuffle=True,
      drop_last=True
    )

    dataloaders = {'Train': train_loader, 'Val': val_loader, 'Test': test_loader}

    return dataloaders 