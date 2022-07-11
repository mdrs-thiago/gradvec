import transformers 
import torch.nn as nn 

class AutoBERT:
    def __init__(self, model_name, device):

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
    def freeze_layers(self, stop = 102):
        i = 0
        for param in self.model.parameters():
            i += 1
            if i == stop:
                break
            param.requires_grad = False




class BERT_MODEL(nn.Module):
    '''
    BERT Model using HuggingFaces implementation. 
    '''


    def __init__(self, cfg):
        super().__init__()

        self.auto_config = transformers.AutoConfig.from_pretrained(cfg.MODEL_NAME)
        config = transformers.BertConfig.from_pretrained(cfg.MODEL_NAME, output_hidden_states=cfg.SHOW_HIDDEN_STATES, output_attentions = cfg.SHOW_ATTENTION)

        self.bert = transformers.BertModel.from_pretrained(cfg.MODEL_NAME, config = config)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(self.auto_config.hidden_size, cfg.n_classes)


    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output

    def freeze_layers(self, layer):
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:layer]]
        for module in modules:
          for param in module.parameters():
              param.requires_grad = False


class BERT_HIDDEN(BERT_MODEL):

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        






class DBERT_BASE_UNCASED(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.auto_config = transformers.AutoConfig.from_pretrained(cfg.MODEL_NAME)


        self.dbert = transformers.DistilBertModel.from_pretrained(cfg.MODEL_NAME)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(self.auto_config, cfg.n_classes)
    
    def forward(self, ids, mask):
        output = self.dbert(ids, attention_mask=mask, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output