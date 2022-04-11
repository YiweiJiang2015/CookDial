import torch
from torch import nn
from torchcrf import CRF
from transformers import BatchEncoding

try:
    from models.base_model import BaseModel
except ImportError:
    from src.models.base_model import BaseModel


def align_span_indices(encodings: BatchEncoding, spans: torch.Tensor, device: torch.device, sequence_index=0):
    """
    BertTokenizer likes to splits a word into pieces making span extraction nontrivial.
    This util function addresses this problem.
    :param encodings: BatchEncoding output by TokenizerFast.
    :param spans: (bsz, num_spans, 2)
    :param sequence_index: 0 or 1, the index in the pair input to BertTokenizer
    :return:
    """

    def add_batch_id(spans_id: torch.Tensor):
        bsz = spans_id.size(0)
        batch_pt = torch.arange(bsz).view(bsz, 1, 1).expand_as(spans_id).to(device=spans_id.device)
        spans_id = torch.stack([batch_pt, spans_id], dim=-1).squeeze(2)
        return spans_id.view(-1, 2)

    spans_start, spans_end = torch.chunk(spans, chunks=2, dim=-1)
    spans_start = add_batch_id(spans_start)
    spans_end = add_batch_id(spans_end)

    aligned_spans_start = [encodings.word_to_tokens(*batch_word_tuple, sequence_index=sequence_index).start
                           for batch_word_tuple in spans_start.tolist()]
    aligned_spans_end = [encodings.word_to_tokens(*batch_word_tuple, sequence_index=sequence_index).end - 1
                         for batch_word_tuple in spans_end.tolist()]

    aligned_spans = [(s, e) for s, e in zip(aligned_spans_start, aligned_spans_end)]

    if device.type == 'cpu':
        aligned_spans = torch.LongTensor(aligned_spans).view(spans.shape)
    else:
        aligned_spans = torch.cuda.LongTensor(aligned_spans).view(spans.shape)
    return aligned_spans


class ModelUserTask(BaseModel):
    def __init__(self, encoder_dialog, span_embedder_step=None, vocabs=None, criterions=None,
                 dropout=0.0, loss_alpha=0.5, loss_beta=1.0, loss_gamma=0.5):
        super(ModelUserTask, self).__init__()
        self.encoder_dialog = encoder_dialog
        self.span_embedder_step = span_embedder_step
        self.vocabs = vocabs
        self.criterions = criterions
        self.dropout = dropout
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_gamma = loss_gamma

        # FFNN layers
        self.completed_step_fc = nn.Sequential(
            nn.Linear(self.span_embedder_step.get_output_size(), 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.requested_step_fc = nn.Sequential(
            nn.Linear(self.span_embedder_step.get_output_size(), 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.intent_fc = nn.Sequential(
            nn.Linear(self.encoder_dialog.get_output_size(), 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, self.vocabs['intent'].size),
        )

    def forward(self, inputs, targets=None, meta=None, masks=None):
        output = {}
        h_utt, step_span_vecs = self._encode(inputs, masks)
        logits_intent = self.intent_fc(h_utt)
        logits_tracker_requested_step = self.requested_step_fc(step_span_vecs).squeeze(-1)
        logits_tracker_completed_step = self.completed_step_fc(step_span_vecs).squeeze(-1)
        output['logits_intent'] = logits_intent
        output['logits_tracker_requested_step'] = logits_tracker_requested_step.masked_fill(~masks['step_spans'],
                                                                                            value=-1e7)
        output['logits_tracker_completed_step'] = logits_tracker_completed_step.masked_fill(~masks['step_spans'],
                                                                                            value=-1e7)

        if targets is not None:
            loss_intent = self.get_loss_intent(logits_intent, targets['intent'])
            loss_tracker_requested_step = self.get_loss_tracker_requested_step(logits_tracker_requested_step,
                                                                               targets['tracker_requested_step'])
            loss_tracker_completed_step = self.get_loss_tracker_completed_step(logits_tracker_completed_step,
                                                                               targets['tracker_completed_step'])

            output['loss_intent'] = loss_intent
            output['loss_tracker_requested_step'] = loss_tracker_requested_step
            output['loss_tracker_completed_step'] = loss_tracker_completed_step
            output['loss'] = self.loss_alpha * loss_intent + self.loss_beta * loss_tracker_requested_step + \
                             self.loss_gamma * loss_tracker_completed_step
        return output

    def _encode(self, inputs, masks):
        hiddens_utt_rec, encoded_utt_rec = self.encoder_dialog((inputs['utterances'], inputs['recipe']))
        ## use [CLS] as h_utt
        h_utt = hiddens_utt_rec['last_hidden_state'][:, 0]
        step_spans = inputs['step_spans']
        step_spans_mask = masks['step_spans']
        step_spans_aligned = align_span_indices(encoded_utt_rec, step_spans, self.device, sequence_index=1)
        step_span_vecs = self.span_embedder_step(hiddens_utt_rec['last_hidden_state'], step_spans_aligned)
        return h_utt, step_span_vecs

    def get_loss_tracker_requested_step(self, logits_tracker, target):
        return self.criterions['tracker_requested_step'](logits_tracker, target)

    def get_loss_tracker_completed_step(self, logits_tracker, target):
        return self.criterions['tracker_completed_step'](logits_tracker, target)

    def get_loss_intent(self, logits_intent, target):
        return self.criterions['intent'](logits_intent, target)


class ModelAgentTask(BaseModel):
    def __init__(self, encoder_dialog, span_embedder_full_set_ptr=None, vocabs=None, criterions=None,
                 dropout=0.0, loss_alpha=0.5, loss_beta=1.0):
        super(ModelAgentTask, self).__init__()
        self.encoder_dialog = encoder_dialog
        self.span_embedder_full_set_ptr = span_embedder_full_set_ptr
        self.vocabs = vocabs
        self.criterions = criterions
        self.dropout = dropout
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self._start_index = vocabs['agent_acts'].lookup_token('<START>')
        self._end_index = vocabs['agent_acts'].lookup_token('<END>')
        self._act_seq_max_len = 5
        # agent_act modules
        self.agent_acts_emitters = nn.Sequential(
            nn.Linear(self.encoder_dialog.get_output_size(), 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, self.vocabs['agent_acts'].size * self._act_seq_max_len),
        )
        self.crf_layer = CRF(self.vocabs['agent_acts'].size, batch_first=True)
        # full_set pointer modules
        full_set_fc_input_size = self.span_embedder_full_set_ptr.get_output_size()
        self.full_set_ptr_fc = nn.Sequential(
            nn.Linear(full_set_fc_input_size, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 4),
        )

    def forward(self, inputs, targets=None, meta=None, masks=None):
        output = {}
        h_utt, full_set_span_vecs = self._encode(inputs, masks)
        emission_scores = self.agent_acts_emitters(h_utt)
        emission_scores = torch.chunk(emission_scores, self._act_seq_max_len, dim=-1)
        emission_scores = torch.stack(emission_scores, dim=1)
        agent_acts_preds = torch.LongTensor(
            self.pad(input=self.crf_layer.decode(emission_scores, mask=masks['agent_acts_pad']),
                     mask=masks['agent_acts_pad'],
                     pad_id=-1)).to(device=self.device)

        logits_full_set_ptr = self.full_set_ptr_fc(full_set_span_vecs)  # .transpose(1,2).contiguous()
        logits_full_set_ptr = logits_full_set_ptr.masked_fill(~masks['node_spans'].unsqueeze(-1),
                                                              value=-1e7).transpose(1, 2).contiguous()

        output['agent_acts_preds'] = agent_acts_preds
        output['logits_full_set_ptr'] = logits_full_set_ptr
        #
        if targets is not None:
            loss_agent_acts = -self.crf_layer(emission_scores, targets['agent_acts'],
                                              mask=masks['agent_acts_pad'], reduction='mean')
            loss_full_set_ptr = self.get_loss_full_set_ptr(logits_full_set_ptr, targets['full_set_ptr'],
                                                           mask=masks['full_set_ptr'])
            output['loss_agent_acts'] = loss_agent_acts
            output['loss_full_set_ptr'] = loss_full_set_ptr
            output['loss'] = self.loss_alpha * loss_agent_acts + self.loss_beta * loss_full_set_ptr
        return output

    def _encode(self, inputs, masks):
        hiddens_utt_rec, encoded_utt_rec = self.encoder_dialog((inputs['utterances'], inputs['recipe']))
        ## use [CLS] as h_utt
        h_utt = hiddens_utt_rec['last_hidden_state'][:, 0]
        full_set_spans = inputs['node_spans']
        full_set_spans_mask = masks['node_spans']
        full_set_spans_aligned = align_span_indices(encoded_utt_rec, full_set_spans, self.device, sequence_index=1)
        full_set_span_vecs = self.span_embedder_full_set_ptr(hiddens_utt_rec['last_hidden_state'], full_set_spans_aligned)
        return h_utt, full_set_span_vecs

    def get_loss_full_set_ptr(self, logits_full_set_ptr, target, mask):
        return self.criterions['full_set_ptr'](logits_full_set_ptr, target, mask)

    def pad(self, input, mask, pad_id=-1):
        'Pad the crf decode output sequence'
        max_len = mask.size(-1)
        padded_input = []
        for l in input:
            l.extend([pad_id] * (max_len - len(l)))
            padded_input.append(l)
        return padded_input


class ModelGenerationTask(BaseModel):
    def __init__(self, enc_dec=None, vocabs=None, criterions=None, device=None, dropout=0.2):
        super(ModelGenerationTask, self).__init__()
        self.enc_dec = enc_dec
        self.dropout = dropout

    def forward(self, inputs, targets=None, meta=None, masks=None, debug=True):
        output = {}
        if not self.training:
            decoded_predictions, top_predictions = self.enc_dec.generate(
                (inputs['query_utterance'], inputs['response_ptr_text']))
            output['decoded_response_gene'] = decoded_predictions
            output['decoded_top_response_gene'] = top_predictions
        if targets is not None:
            trans_outputs, predictions, labels = self.enc_dec((inputs['query_utterance'], inputs['response_ptr_text']),
                                                              targets['gold_response'])
            loss_response_gene = trans_outputs.loss
            output['loss_response_gene'] = loss_response_gene
            output['loss'] = loss_response_gene
            output['preds_response_gene'] = predictions
            output['labels_response_gene'] = labels

        return output
