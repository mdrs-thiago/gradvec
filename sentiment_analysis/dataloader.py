from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class DatasetBERT(Dataset):
    def __init__(self, data, tokenizer, max_size):

        self.data = data
        self.tokenizer = tokenizer
        self.max_size = max_size 

    def __len__(self):
        return len(self.data)

    def __getitem__(self):
        pass


def transform_dataset(name, tokenizer):  
  if name != 'sst':
    dataset = load_dataset(name)
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)

    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
  else:
    dataset = load_dataset('sst','default')
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], truncation=True, padding='max_length'), batched=True)

    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


  train_dataset = encoded_dataset['train']
  test_dataset = encoded_dataset['test']

  train_loader = DataLoader(train_dataset, batch_size=16)
  test_loader = DataLoader(test_dataset, batch_size=16)

  return train_loader, test_loader