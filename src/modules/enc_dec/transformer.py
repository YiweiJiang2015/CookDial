import torch, logging
from torch import nn
from transformers import T5ForConditionalGeneration, BigBirdPegasusForConditionalGeneration

try:
    from modules.encoders import TransModel
except ImportError:
    from src.modules.encoders import TransModel
logger = logging.getLogger(__name__)


class TransEncDec(TransModel):
    def __init__(self, bert_name, device, is_split_into_words=False, fine_tune=False,
                 padding=False, model_path=None, use_finetuned_model=False):
        super(TransEncDec, self).__init__(bert_name, device, is_split_into_words=is_split_into_words,
                                          fine_tune=fine_tune,
                                          padding=padding, model_path=model_path,
                                          use_finetuned_model=use_finetuned_model)
        self.trans_model = T5ForConditionalGeneration.from_pretrained(bert_name)

    def forward(self, input_text, target_text):
        if isinstance(input_text, tuple):
            input_toknized = self.tokenizer(*input_text, is_split_into_words=self.is_split_into_words, max_length=1024,
                                            return_tensors='pt', padding=self.padding).to(self.device)
        if isinstance(input_text, list):
            input_toknized = self.tokenizer(input_text, is_split_into_words=self.is_split_into_words, max_length=1024,
                                            return_tensors='pt', padding=self.padding).to(self.device)
        input_ids = input_toknized.input_ids
        mask = input_toknized.attention_mask
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            decoder_tokenized = self.tokenizer(target_text, is_split_into_words=self.is_split_into_words,
                                               return_tensors='pt', padding=self.padding).to(self.device)
            y = decoder_tokenized.input_ids
            decoder_input_ids = y[:, :-1]  # .to(self.device, dtype = torch.long)
            target_mask = decoder_tokenized.attention_mask
            labels = y[:, 1:].clone().detach()
            labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
        # the forward function automatically creates the correct decoder_input_ids
        outputs = self.trans_model(input_ids=input_ids, attention_mask=mask, labels=y,  # labels,
                                   decoder_attention_mask=target_mask)  # decoder_input_ids=decoder_input_ids,
        predictions = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
        return outputs, predictions, labels

    def get_output_size(self):
        return

    def generate(self, input_text, num_return_sequences=3):
        """
        Used for valid phase.
        :param input_text:
        :param target_text:
        :return:
        """
        if isinstance(input_text, tuple):
            input_toknized = self.tokenizer(*input_text, is_split_into_words=self.is_split_into_words,
                                            return_tensors='pt', padding=self.padding).to(self.device)
        if isinstance(input_text, list):
            input_toknized = self.tokenizer(input_text, is_split_into_words=self.is_split_into_words,
                                            return_tensors='pt', padding=self.padding).to(self.device)
        input_ids = input_toknized.input_ids
        mask = input_toknized.attention_mask
        generated_ids = self.trans_model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_length=64,
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=0.8,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2
        )
        bsz = input_ids.size(0)
        preds = []
        top_preds = []
        for i in range(bsz):
            preds.append([self.tokenizer.decode(generated_ids[i * num_return_sequences + j],
                                                skip_special_tokens=True, clean_up_tokenization_spaces=True)
                          for j in range(num_return_sequences)])
            top_preds.append(self.tokenizer.decode(generated_ids[i * num_return_sequences],
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=True))
        return preds, top_preds

    @classmethod
    def from_config(cls, config):
        bert_name = config['bert_name']
        device = config['device']
        is_split_into_words = config['is_split_into_words']
        fine_tune = config.get('fine_tune', False)
        padding = config.get('padding', False)
        model_path = config.get('model_path', None)
        use_finetuned_model = config.get('use_finetuned_model', False)
        return cls(bert_name, device, is_split_into_words=is_split_into_words, fine_tune=fine_tune,
                   padding=padding, model_path=model_path, use_finetuned_model=use_finetuned_model)
