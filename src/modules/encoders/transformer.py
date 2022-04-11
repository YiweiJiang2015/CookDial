import torch, logging
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, BigBirdModel
logger = logging.getLogger(__name__)


class TransModel(nn.Module):

    def __init__(self, bert_name, device, is_split_into_words=False, fine_tune=False,
                 padding=False, model_path=None, use_finetuned_model=False,
                 attention_type=None):
        super(TransModel, self).__init__()
        self.bert_name = bert_name
        logger.info(f"BERT name: {bert_name}")
        self.is_split_into_words = is_split_into_words
        self.device = device
        add_prefix_space = False
        if 'longformer' in bert_name or 'bigbird' in bert_name:
            add_prefix_space = True # for RoBerta-based model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, add_prefix_space=add_prefix_space)
        if model_path is not None and use_finetuned_model:
            self.trans_model = AutoModel.from_pretrained(model_path)
            logger.info(f'Successfully load pre-trained bert from {model_path}.')
        else:
            if 'bigbird' in bert_name:
                self.trans_model = AutoModel.from_pretrained("google/bigbird-roberta-base",
                                                            attention_type=attention_type, #original_full
                                                            add_cross_attention=False)
            else:
                self.trans_model = AutoModel.from_pretrained(bert_name)
        self.padding=padding
        # for param in model.base_model.parameters(): # for task specific models
        if fine_tune:
            # if we fine tune bert, only freeze the last pooler layer.
            try:
                modules = [self.trans_model.pooler]

            except AttributeError as e:
                logger.warning(e)
            else:
                for mod in modules:
                    for param in mod.parameters():
                        param.requires_grad = False
        else:
            # freeze all of bert weights
            for param in self.trans_model.parameters():
                param.requires_grad = False

    def forward(self, texts):
        # due to huggingface's limitation, `return_offsets_mapping`=True has no benefits for now. See issue below:
        # https://github.com/huggingface/transformers/issues/11828
        if isinstance(texts, list):
            encoded_inputs = self.tokenizer(texts, is_split_into_words=self.is_split_into_words, padding=self.padding,
                                            return_tensors='pt', truncation=True, return_offsets_mapping=False).to(self.device)
        if isinstance(texts, tuple):
            encoded_inputs = self.tokenizer(*texts, is_split_into_words=self.is_split_into_words, padding=self.padding,
                                            return_tensors='pt', truncation=True, return_offsets_mapping=False).to(self.device)
        # offset = encoded_inputs.pop('offset_mapping')
        output = self.trans_model(**encoded_inputs, output_hidden_states=False, output_attentions=True)
        return output, encoded_inputs

    def get_output_size(self):
        if self.bert_name in ['bert-base-cased', 'bert-base-uncased', 'allenai/longformer-base-4096', 'google/bigbird-roberta-base']:
            return 768
        if self.bert_name in ['bert-large-cased', 'bert-large-uncased']:
            return 1024
    
    def save_pretrained(self, path: str):
        self.trans_model.save_pretrained(path)
    
    @classmethod
    def from_config(cls, config: dict):
        bert_name = config['bert_name']
        device = config['device']
        is_split_into_words = config['is_split_into_words']
        fine_tune = config.get('fine_tune', False)
        padding = config.get('padding', False)
        model_path = config.get('model_path', None)
        use_finetuned_model = config.get('use_finetuned_model', False)
        attention_type = config.get('attention_type', "original_full")
        return cls(bert_name, device, is_split_into_words=is_split_into_words, fine_tune=fine_tune,
                   padding=padding, model_path=model_path, use_finetuned_model=use_finetuned_model,
                   attention_type=attention_type)