import torch.nn as nn 
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig


class AutoBERT:
    def __init__(self, model_name, device, show_hidden_states = False):

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=show_hidden_states)

        self.model = AutoModelForImageClassification.from_pretrained(model_name, config=config)
        self.model.to(device)
        
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False 
        
        self.model.classifier.weight.requires_grad = True 
        self.model.classifier.bias.requires_grad = True 