from tkinter import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random 
import numpy as np 

from transformers import ViTFeatureExtractor, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, AutoAugment, AutoAugmentPolicy

import torch
from torchvision.datasets import CIFAR10, SVHN, INaturalist, Places365, LSUN, StanfordCars, DTD, CIFAR100, ImageFolder

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, data_augmentation = False):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        transform = []

        if data_augmentation:
            transform.append(AutoAugment(AutoAugmentPolicy.CIFAR10))

        if feature_extractor.do_resize:
            transform.append(Resize((feature_extractor.size, feature_extractor.size)))

        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))


        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)



def create_CIFAR10_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      CIFAR10('./', download=True, train=False, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      CIFAR10('./', download=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_SVHN_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
    SVHN('./', download=True, split='test', transform=ViTFeatureExtractorTransforms(model_name)),
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=my_collate
  )

  if ID: 
    train_loader = DataLoader( 
      SVHN('./', download=True, split='train', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader 

  return val_loader

def create_INaturalist_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      INaturalist('./', '2021_valid', download=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      INaturalist('./', '2021_train', download=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_LSUN_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      LSUN('./', download=True, classes='test', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      LSUN('./', download=True, classes='train', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_Places365_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      Places365('./', download=True, split='val', small=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      Places365('./', download=True, split='train-standard', small=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_StanfordCars_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      StanfordCars('./', download=True, split='test', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      StanfordCars('./', download=True, split='train', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_textures_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      DTD('./', download=True, split='test', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      DTD('./', download=True, split='train', transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_CIFAR100_loader(model_name, batch_size = 24, num_workers = 2, ID=False, data_augmentation = False):
  val_loader = DataLoader( 
      CIFAR100('./', download=True, train=False, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      CIFAR100('./', download=True, transform=ViTFeatureExtractorTransforms(model_name, data_augmentation = data_augmentation)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_custom_OOD_loader(model_name, path = '', batch_size = 24, num_workers = 2):
  val_loader = DataLoader(
      ImageFolder(path, transform = ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  return val_loader


def create_loader(model_name, dataset, ID=False, data_augmentation = False):
  if dataset == 'cifar10':
    return create_CIFAR10_loader(model_name, ID=ID)
  elif dataset == 'svhn':
    return create_SVHN_loader(model_name, ID=ID)
  elif dataset == 'inaturalist':
    return create_INaturalist_loader(model_name, ID=ID)
  elif dataset == 'lsun':
    return create_LSUN_loader(model_name, ID=ID)
  elif dataset == 'places365':
    return create_Places365_loader(model_name, ID=ID)
  elif dataset == 'textures':
    return create_textures_loader(model_name, ID=ID)
  elif dataset == 'cifar100':
    return create_CIFAR100_loader(model_name, ID=ID, data_augmentation = data_augmentation)
  else:
    return create_custom_OOD_loader(model_name, path=dataset)

















#-------------------------OLD


class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)





def transform_dataset(name, extractor, subsample=True,prop=0.10, ID=True, only_ID = True):  

  if name == 'svhn':
    train_dataset, test_dataset = load_dataset(name, 'cropped_digits', split=['train','test'])
    n = 'image'

  elif subsample:
    _dataset = load_dataset(name, split=['train[:15%]'])
    train_dataset = _dataset[0]
    n = 'img'
    
  else:
    train_dataset, test_dataset = load_dataset(name, split=['train','test'])
    print(type(train_dataset))
    n = 'img'

  if subsample:

    if only_ID:
      encoded_dataset = train_dataset.map(lambda examples: extractor(examples['img']), batched=True)
      encoded_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
      
      train_loader = DataLoader(encoded_dataset, batch_size=16)
      return train_loader

    list_index = np.arange(test_dataset.shape[0])
    random.shuffle(list_index)
    limit = int(test_dataset.shape[0]*prop)
    ranges = list_index[:limit]
    test_subsample = test_dataset[ranges]
    test_subsample[n] = list(map(extractor, test_subsample[n]))
    _test_dataset = CustomDataset(test_subsample[n], test_subsample['label'])
    test_loader = DataLoader(_test_dataset, batch_size=16)

    if ID:
      prop = 0.5
      list_index = np.arange(train_dataset.shape[0])
      random.shuffle(list_index)
      limit = int(train_dataset.shape[0]*prop)
      ranges = list_index[:limit]
      train_subsample = train_dataset[ranges]
      train_subsample[n] = list(map(extractor, train_subsample[n]))
      _train_dataset = CustomDataset(train_subsample[n], train_subsample['label'])
      train_loader = DataLoader(_train_dataset, batch_size=16)
      
      #train_dataset[n] = list(map(extractor, train_dataset[n]))
      #_train_dataset = CustomDataset(train_dataset[n], train_dataset['label'])
      #train_loader = DataLoader(_train_dataset, batch_size=16)
      
      del train_dataset, _train_dataset, test_dataset

      return train_loader, test_loader

  else:

    test_dataset[n] = list(map(extractor, test_dataset[n]))
    _test_dataset = CustomDataset(test_subsample[n], test_subsample['label'])
    test_loader = DataLoader(_test_dataset, batch_size=16)

    if ID:
      train_dataset[n] = list(map(extractor, train_dataset[n]))
      _train_dataset = CustomDataset(train_dataset[n], train_dataset['label'])
      train_loader = DataLoader(_train_dataset, batch_size=16)
      
      del train_dataset, _train_dataset, test_dataset
      return train_loader, test_loader

  del train_dataset, test_dataset
  return test_loader