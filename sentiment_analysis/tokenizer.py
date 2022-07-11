import transformers 


class TokenizerBERT:
    def __init__(self, cfg):
        
        self.tokenizer = transformers.BertTokenizer.from_pretrained(cfg.MODEL_NAME)
        self.max_len = cfg.max_len 

    def convert(self, dataset):

        encoded_dataset = dataset.map(lambda examples: self.tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)

        encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

        return encoded_dataset