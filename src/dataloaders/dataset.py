r"""
Author: Yiwei Jiang
Date:
Description:
    The philosophy is to read processed data from disk files as more as possible and do processing here as less as possible.
    This requires that lots of dirty work has to be done in preprocess.py.
    Such arrangement takes advantage of the flexibility of objects in the preprocess script
    as everything here is awkwardly de-serialized from dicts which clutters data processing.

"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from typing import Tuple, List, Union, Any

try:
    from utils import read_json, Vocabulary, is_empty_list, to_device
except ImportError:
    from src.utils import read_json, Vocabulary, is_empty_list, to_device


def pad_node_span(batch):
    """
    :return:
        node_spans: padded span tuple indices, including indices for dummy null span
        node_spans_mask: stores the actual number of spans for each instance within minibatch
    """
    node_spans = [b['node_spans'] for b in batch]
    max_len = torch.IntTensor([len(x['node_span_dict']) for x in batch]).max().item()
    node_spans_mask = torch.zeros((len(batch), max_len + 1)).bool()
    new_node_spans = []
    for i, x in enumerate(node_spans):
        old_len = len(x)
        y = [t for t in x]
        if old_len < max_len:
            y.extend([(0, 0)] * (max_len - old_len))  # todo (0,0) may be not a good padding token
        node_spans_mask[i, :old_len + 1] = True
        new_node_spans.append(y)
    new_node_spans = torch.LongTensor(new_node_spans)
    return new_node_spans, node_spans_mask


def pad_piece_spans(piece_spans):
    """
    :return:
    """
    max_len = torch.IntTensor([len(x) for x in piece_spans]).max().item()
    piece_spans_mask = torch.zeros((len(piece_spans), max_len)).bool()
    new_piece_spans = []
    for i, x in enumerate(piece_spans):
        old_len = len(x)
        y = [t for t in x]
        if old_len < max_len:
            y.extend([(0, 0)] * (max_len - old_len))
        piece_spans_mask[i, :old_len] = True
        new_piece_spans.append(y)

    new_piece_spans = torch.LongTensor(new_piece_spans)
    return new_piece_spans, piece_spans_mask


def get_utt_span_mask(utterance_spans):
    """
    This only works when we did not pad the history with [PAD] tokens.
    """
    return ~utterance_spans.sum(-1).eq(0)


r"""
Below is used for paper. Moving it outside of this file needs extra engineering work...
Author: Yiwei Jiang
Date: 2021.07.12
Description:
    Refactor the messy codes in dataset.py.
    There will be 3 dataset classes corresponding to the 3 tasks in the paper.
    Most methods are copied from dataset.py and are simplified.
"""


class CookDatasetBase(Dataset):
    def __init__(self, filename, vocabs, history_prepend=True, history_window=1, seed=None):
        self.dialogues = read_json(filename)["dialogues"]
        self.vocabs = vocabs
        self.history_prepend = history_prepend
        self.history_window = history_window
        self.seed = seed
        self.instances = []

    def get_user_turns(self):
        """
        Each turn is a training instance. Batch size can be larger than 1.
        """
        raise NotImplementedError

    def convert(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def remove_bot_marker(his_list):
        """
        meta['history_tokens'] has a nested structure [[token_list: list[str], bot_marker: bool], ...]
        This util removes the bot_marker
        """
        return [l[0] if isinstance(l, list) else l for l in his_list]

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)


def collate_user_task(batch):
    batch.sort(key=lambda x: len(x['utterances']), reverse=True)
    utterances = [x['utterances'] for x in batch]

    utterance_length = torch.LongTensor([x['utterance_length'] for x in batch])

    tracker_requested_step = torch.LongTensor([x['tracker_requested_step'] for x in batch])
    tracker_completed_step = torch.LongTensor([x['tracker_completed_step'] for x in batch])
    recipe = [x['recipe'] for x in batch]
    intent = torch.stack([x['intent'] for x in batch], dim=0)
    # spans of recipe pieces: title, individual ingredients, individual instruction
    step_spans, step_spans_mask = pad_piece_spans([x['step_spans'] for x in batch])

    inputs = {'utterances': utterances,
              'utterance_length': utterance_length,
              # 'history': history,
              'recipe': recipe,
              'step_spans': step_spans,
              }
    targets = {
        'tracker_requested_step': tracker_requested_step,
        'tracker_completed_step': tracker_completed_step,
        'intent': intent
    }
    masks = {
        'step_spans': step_spans_mask,
    }
    meta = {
        'user_utts': [x['user_utt_meta'] for x in batch],
        'bot_utts': [x['bot_utt_meta'] for x in batch],
        'gold_intent': [x['gold_intent'] for x in batch],
        'dialog_id': [x['dialog_id'] for x in batch],
        'utt_id': [x['utt_id'] for x in batch],
        'ing_list_length': [x['ing_list_length'] for x in batch]
    }
    return {
        'inputs': inputs,
        'targets': targets,
        'masks': masks,
        "meta": meta
    }


class DatasetUserTask(CookDatasetBase):
    def __init__(self, filename, vocabs, seed=None, history_prepend=True, history_window=1):
        super(DatasetUserTask, self).__init__(filename, vocabs, history_prepend=history_prepend,
                                              history_window=history_window, seed=seed)
        self.get_user_turns()

    def get_user_turns(self):
        for dialog in self.dialogues:
            for utt_user, utt_bot in itertools.zip_longest(
                    dialog['utterances'][:], dialog['utterances'][1:]):
                if not utt_user['bot'] and 'agent_acts_shift' in utt_user['meta']:
                    self.instances.append(self.convert(utt_user, kb=dialog['knowledge_base'], utt_bot=utt_bot))

    def convert(self, utt_user, kb=None, utt_bot=None):

        utterance_pretokens = []
        recipe_pretokens = []

        if self.history_window:
            utterance_pretokens = self.add_history_as_whole_beta(utt_user['meta'])
        else:
            utterance_pretokens = utt_user['meta']['utt_tokens']
        utterance_length = (0, len(utterance_pretokens) - 1)
        # if 'history_tokens' in utt_user['meta']:
        #     history_pretokens=utt_user['meta']['history_tokens']

        # tracker task
        tracker_requested_step = utt_user['meta']['tracker_requested_step_num']

        tracker_completed_step = utt_user['meta']['tracker_completed_step_num']

        # user intent task
        intent_indices = torch.LongTensor(
            [self.vocabs['intent'].lookup_token(intent) for intent in utt_user['meta']['parsed_intents']])
        intent = torch.zeros(size=[self.vocabs['intent'].size]).float()  # float type is required by torch.F.bce_loss()
        intent.scatter_(0, intent_indices, 1.)  # scatter 1.0 on the right positions to create multi labels
        gold_intent = utt_user['meta']['intent']

        # utterance for meta
        user_utt_meta = utt_user['utterance']
        if utt_bot is not None:
            bot_utt_meta = utt_bot['utterance']
        else:  # zip_longest may produce None trailing item
            bot_utt_meta = '__EOF__'

        # recipe input
        recipe_pretokens.extend([token for token in kb['title']['meta']['tokens']])
        for ingredient in kb['ingredients']:
            recipe_pretokens.extend(
                [token for token in ingredient['meta']['tokens']])
        for instruction in kb['instructions']:
            recipe_pretokens.extend(
                [token for token in instruction['meta']['tokens']])
            # recipe_boundaries.append(instruction['meta']['boundary'])
        node_spans = [(n[-1]) for n in kb['nodes']]
        step_spans = self.get_step_spans(kb)

        return {'utterances': utterance_pretokens,
                'utterance_length': utterance_length,
                'intent': intent,
                'recipe': recipe_pretokens,
                'tracker_requested_step': tracker_requested_step,
                'tracker_completed_step': tracker_completed_step,
                'step_spans': step_spans,
                'user_utt_meta': user_utt_meta,
                'bot_utt_meta': bot_utt_meta,
                'gold_intent': gold_intent,
                # gold joint pointer will be generated in collate_fn, not here!!
                'dialog_id': utt_user['meta']['dialog_id'],
                'utt_id': utt_user['utt_id'].split('-')[1],
                'ing_list_length': len(kb['ingredients'])
                }

    def add_history_as_whole(self, meta):
        """
        Simplified version of add_history_as_whole() in MyDatasetBERT class
        Concat history tokens with utt tokens in one list
        """
        utt_tokens = []
        utterance_length = []

        if self.history_prepend:
            if not is_empty_list(self.remove_bot_marker(meta['history_tokens'])):
                for hist in meta['history_tokens'][-self.history_window:]:
                    utt_tokens.extend(hist[0])
                    utterance_length.append(len(utt_tokens))
            utt_tokens.extend(meta['utt_tokens'])
            utterance_length.append(len(utt_tokens))
        else:
            utt_tokens.extend(meta['utt_tokens'])
            utterance_length.append(len(utt_tokens))
            if not is_empty_list(self.remove_bot_marker(meta['history_tokens'])):
                for hist in meta['history_tokens'][-self.history_window:]:
                    utt_tokens.extend(hist[0])
                    utterance_length.append(len(utt_tokens))

        return utt_tokens

    def add_history_as_whole_beta(self, meta):
        cxt_token_list = []
        utt_tokens = []
        if not is_empty_list(self.remove_bot_marker(meta['history_tokens'])):
            for hist in meta['history_tokens'][-self.history_window:]:
                cxt_token_list.append(hist[0])

        cxt_token_list.append(meta['utt_tokens'])
        for cxt_token in cxt_token_list:
            utt_tokens.extend(cxt_token)
        return utt_tokens

    def remove_bot_marker(self, his_list):
        """
        meta['history_tokens'] has a nested structure [[token_list: list[str], bot_marker: bool], ...]
        This util removes the bot_marker
        """
        return [l[0] if isinstance(l, list) else l for l in his_list]

    @staticmethod
    def get_step_spans(kb):
        """
        Copied from get_micro_piece_spans()
        Get step span indices from recipes: title, ing-x, inst-x
        """
        step_spans = []
        step_spans.append(kb['title']['meta']['boundary'])
        step_spans.extend([ing['meta']['boundary'] for ing in kb['ingredients']])
        step_spans.extend([inst['meta']['boundary'] for inst in kb['instructions']])
        return step_spans


def collate_agent_task(batch):
    batch.sort(key=lambda x: len(x['utterances']), reverse=True)
    # utterance = rnn_utils.pad_sequence([x['utterance'] for x in batch], batch_first=True)
    # seq_lengths = torch.IntTensor([x['utterance'].shape[0] for x in batch])
    utterances = [x['utterances'] for x in batch]

    # utterance_length = torch.LongTensor([x['utterance_length'] for x in batch])
    # agent act seq
    agent_acts = torch.stack([x['agent_acts'] for x in batch], dim=0)
    # full_set ptr seq
    full_set_ptr = torch.stack([x['full_set_ptr'] for x in batch], dim=0)
    # completed step
    completed_step = torch.LongTensor([x['completed_step'] for x in batch])
    node_spans, node_spans_mask = pad_node_span(batch)
    recipe = [x['recipe'] for x in batch]

    inputs = {'utterances': utterances,
              # 'utterance_length': utterance_length, # deprecated
              'completed_step': completed_step,
              'recipe': recipe,
              'node_spans': node_spans,
              }
    targets = {
        'full_set_ptr': full_set_ptr,
        'agent_acts': agent_acts
    }
    masks = {
        'agent_acts_pad': torch.stack([x['agent_act_mask_pad'] for x in batch], dim=0),
        'agent_acts_wrap': torch.stack([x['agent_act_mask_wrap'] for x in batch], dim=0),
        'full_set_ptr': torch.stack([x['full_set_ptr_mask'] for x in batch], dim=0),
        'node_spans': node_spans_mask,
    }
    meta = {
        'user_utts': [x['user_utt_meta'] for x in batch],
        'bot_utts': [x['bot_utt_meta'] for x in batch],
        'agent_action_frames': [x['agent_action_frames'] for x in batch],
        'dialog_id': [x['dialog_id'] for x in batch],
        'utt_id': [x['utt_id'] for x in batch],
        'ing_list_length': [x['ing_list_length'] for x in batch],
        'node_span_dict': [x['node_span_dict'] for x in batch]
    }
    return {
        'inputs': inputs,
        'targets': targets,
        'masks': masks,
        "meta": meta
    }


class DatasetAgentTask(DatasetUserTask):
    def __init__(self, filename, vocabs, seed=None, history_prepend=True,
                 history_window=1):
        self._max_fname_len = 5
        super(DatasetAgentTask, self).__init__(filename, vocabs, history_prepend=history_prepend,
                                               history_window=history_window, seed=seed)

    def convert(self, utt_user, kb=None, utt_bot=None):
        utterance_pretokens = []
        recipe_pretokens = []

        if self.history_window:
            utterance_pretokens = self.add_history_as_whole_beta(utt_user['meta'])
        else:
            utterance_pretokens = utt_user['meta']['utt_tokens']
        utterance_length = (0, len(utterance_pretokens) - 1)

        # agent act seq task
        # first we wrap <START> and <END>
        unpadded_agent_act_vector = torch.LongTensor(
            [self.vocabs['agent_acts'].lookup_token(fname)
             for fname in self.wrap_start_end(utt_user['meta']['agent_acts_shift'], add_start=False)])

        agent_act_mask_pad = self.get_pad_mask(unpadded_agent_act_vector, maxlen=self._max_fname_len)

        agent_act_vector = self.padding_to_maxlen(unpadded_agent_act_vector,
                                                  maxlen=self._max_fname_len,
                                                  pad_id=-1 
                                                  )
        agent_act_mask_wrap = self.get_wrap_mask(agent_act_vector, vocab=self.vocabs['agent_acts'])

        # full_set pointer seq task
        full_set_ptr = torch.LongTensor(utt_user['meta']['full_set_pointer_seq_shift'])
        full_set_ptr_mask = agent_act_mask_pad & agent_act_mask_wrap
        full_set_ptr_mask = full_set_ptr_mask[:-1]  # due to the preprocessing, we only need the middle 4 positions
        # extra input
        ## user intents
        user_intents = utt_user['meta']['parsed_intents']

        ## completed step
        completed_step = utt_user['meta']['tracker_completed_step_num'] + 1

        # utterance for meta
        user_utt_meta = utt_user['utterance']
        if utt_bot is not None:
            bot_utt_meta = utt_bot['utterance']
        else:  # zip_longest may produce None trailing item
            bot_utt_meta = '__EOF__'
        # agent_action_frame meta
        agent_action_frames_meta = utt_user['meta']['agent_action_frames_shift']

        # recipe input
        recipe_pretokens.extend([token for token in kb['title']['meta']['tokens']])
        for ingredient in kb['ingredients']:
            recipe_pretokens.extend(
                [token for token in ingredient['meta']['tokens']])
        for instruction in kb['instructions']:
            recipe_pretokens.extend(
                [token for token in instruction['meta']['tokens']])
            # recipe_boundaries.append(instruction['meta']['boundary'])
        node_spans = [(n[-1]) for n in kb['nodes']]
        node_span_dict_meta = {i + 1: tuple(x) for i, x in enumerate(kb['nodes'])}
        # step_spans = self.get_step_spans(kb)

        return {
            'utterances': utterance_pretokens,
            # 'utterance_length': utterance_length,
            'recipe': recipe_pretokens,
            'completed_step': completed_step,
            'agent_acts': agent_act_vector,
            'agent_act_mask_pad': agent_act_mask_pad,
            'agent_act_mask_wrap': agent_act_mask_wrap,
            'node_spans': node_spans,
            'node_span_dict': node_span_dict_meta,
            'full_set_ptr': full_set_ptr,
            'full_set_ptr_mask': full_set_ptr_mask,
            # 'step_spans': step_spans,
            'user_utt_meta': user_utt_meta,
            'bot_utt_meta': bot_utt_meta,
            'agent_action_frames': agent_action_frames_meta,
            'dialog_id': utt_user['meta']['dialog_id'],
            'utt_id': utt_user['utt_id'].split('-')[1],
            'ing_list_length': len(kb['ingredients'])
        }

    @staticmethod
    def padding_to_maxlen(input: torch.Tensor, maxlen: int, pad_id: int = 0):
        'Pad the last dim of input to maxlen'
        padded = input.new_ones((*input.shape[:-1], maxlen)) * pad_id
        padded[..., :input.shape[-1]] = input  # avoid in-place op
        return padded

    @staticmethod
    def wrap_start_end(tokens: list, add_start=True):
        'Wrap a sentence with START and END tokens'
        wrapped = ['<START>'] if add_start else []
        wrapped.extend(tokens)
        wrapped.append('<END>')
        return wrapped

    @staticmethod
    def get_pad_mask(input: torch.Tensor, maxlen: int):
        """
        Get the mask of padded input
        :param input: shall not be padded
        :param maxlen:
        :return:
        """
        # print(input.size(-1), maxlen)
        # if input.size(-1) > maxlen:
        #     print()
        assert input.size(-1) <= maxlen, ""
        mask = torch.zeros((*input.shape[:-1], maxlen), dtype=torch.bool)
        mask_ = torch.ones(input.shape, dtype=torch.bool)
        mask[..., :input.shape[-1]] = mask_
        return mask

    @staticmethod
    def get_wrap_mask(input: torch.Tensor, vocab: Vocabulary):
        """
        Get a mask of <START> and <END>
        """
        mask_start = input.eq(vocab.lookup_token('<START>'))
        mask_end = input.eq(vocab.lookup_token('<END>'))

        return ~(mask_start | mask_end)


def collate_generation_task(batch):
    query_utterance = [x['prompt'] + x['query_utterance'] for x in batch]
    recipe_text = [x['recipe_text'] for x in batch]
    response_ptr_text = [x['response_ptr_text'] for x in batch]
    gold_response = [x['gold_response'] for x in batch]
    inputs = {
        'query_utterance': query_utterance,
        'response_ptr_text': response_ptr_text,
        'recipe_text': recipe_text
    }
    targets = {
        'gold_response': gold_response,
    }
    meta = {
        'user_utts': [x['user_utt_meta'] for x in batch],
        'bot_utts': [x['bot_utt_meta'] for x in batch],
        'agent_action_frames': [x['agent_action_frames'] for x in batch],
        'gold_intent': [x['gold_intent'] for x in batch],
        'dialog_id': [x['dialog_id'] for x in batch],
        'utt_id': [x['utt_id'] for x in batch],
    }
    return {
        'inputs': inputs,
        'targets': targets,
        'meta': meta
    }


class DatasetGenerationTask(CookDatasetBase):
    def __init__(self, filename, vocabs: dict[str: Vocabulary], history_prepend=True,
                 history_window=1, use_act_hint=True, use_full_set_ptr_hint=True, seed=None):
        self.use_act_hint = use_act_hint
        self.use_full_set_ptr_hint = use_full_set_ptr_hint
        super(DatasetGenerationTask, self).__init__(filename, vocabs, history_prepend=history_prepend,
                                                    history_window=history_window, seed=seed)
        self.get_user_turns()

    def get_user_turns(self):
        for dialog in self.dialogues:
            for utt_user, utt_bot in itertools.zip_longest(
                    dialog['utterances'][:], dialog['utterances'][1:]):
                if not utt_user['bot'] and 'agent_acts_shift' in utt_user['meta']:
                    self.instances.append(self.convert(utt_user, kb=dialog['knowledge_base'],
                                                  utt_bot=utt_bot))

    def get_response_ptr_text(self, pointer, kb):
        if pointer == 0:
            return kb['title']['text']
        if pointer == 1:
            return kb['ing_list']['text']
        else:
            return kb['instructions'][pointer - 2]['text']

    def get_step_text(self, step_ptr, kb):
        ing_list_len = len(kb['ingredients'])
        if step_ptr == 0:
            return kb['title']['text']
        elif step_ptr <= ing_list_len:
            return kb['ingredients'][step_ptr - 1]['text']
        else:
            return kb['instructions'][step_ptr - ing_list_len - 1]['text']

    def get_entire_recipe_text(self, kb):
        entire_recipe_txt = (kb['title']['text'] + ' ')
        entire_recipe_txt += (kb['ing_list']['text'] + ' ')
        for inst in kb['instructions']:
            entire_recipe_txt += (inst['text'] + ' ')
        return entire_recipe_txt.strip()

    def get_response_full_set_text(self, full_set_ptr_seq, requested_step, kb):
        """This is for the paper:: Generation:: section"""
        grounding_text = [kb['nodes'][full_set_ptr - 1][1] if full_set_ptr != 0 else '' for full_set_ptr in
                          full_set_ptr_seq]
        grounding_string = ' '.join(grounding_text).strip()
        if len(grounding_string) == 0:
            grounding_string = self.get_step_text(requested_step, kb)
        return grounding_string

    def add_history_using_string(self, utt_string, meta):
        if self.history_prepend:  # deprecated
            utt_string = ' ' + utt_string
            if not is_empty_list(self.remove_bot_marker(meta['history'])):
                utt_string = ' '.join([hist if isinstance(hist, str) else hist[0]
                                       for hist in meta['history'][-self.history_window:]] + [utt_string])
                return utt_string.strip()
            else:
                return utt_string.strip()
        else:
            utt_string += ' '
            if not is_empty_list(self.remove_bot_marker(meta['history'])):
                utt_string += ' '.join([hist if isinstance(hist, str) else hist[0]
                                        for hist in meta['history'][-self.history_window:]])
                return utt_string.strip()
            else:
                return utt_string.strip()

    def add_history_using_string_beta(self, utt_string, meta):
        """
        Add a reverse option
        """
        if not is_empty_list(self.remove_bot_marker(meta['history'])):
            cxt_string_list = [hist if isinstance(hist, str) else hist[0]
                               for hist in meta['history'][-self.history_window:]] + [utt_string]
        else:
            cxt_string_list = [utt_string]
        return ' '.join(cxt_string_list)

    def convert(self, utt_user, kb=None, utt_bot=None):
        recipe_text = ''
        # prompt = 'paraphrase: '
        if self.use_act_hint:
            prompt = 'answer question with hints [' + ', '.join(utt_bot['meta']['agent_acts']) + ']: '
        else:
            prompt = 'answer question: '
        if self.history_window:
            query_utterance = self.add_history_using_string_beta(utt_user['utterance'].lower(), utt_user['meta'])  #
        else:
            query_utterance = utt_user['utterance'].lower()
        if self.use_full_set_ptr_hint:
            response_ptr_text = self.get_response_full_set_text(utt_user['meta']['full_set_pointer_seq_shift'],
                                                                    utt_user['meta']['tracker_requested_step_num'],
                                                                    kb)
        else:
            response_ptr_text = self.get_entire_recipe_text(kb)

        gold_response = utt_bot['utterance']
        gold_intent = utt_user['meta']['intent']
        # recipe text
        recipe_text += (kb['title']['text'] + kb['ing_list']['text'])
        for inst in kb['instructions']:
            recipe_text += inst['text']

        # utterance for meta
        user_utt_meta = utt_user['utterance']
        if utt_bot is not None:
            bot_utt_meta = utt_bot['utterance']
        else:  # zip_longest may produce None trailing item
            bot_utt_meta = '__EOF__'
        # agent_action_frames meta
        agent_action_frames_meta = utt_user['meta']['agent_action_frames_shift']
        return {
            'prompt': prompt,
            'query_utterance': query_utterance,
            'recipe_text': recipe_text,
            'response_ptr_text': response_ptr_text,
            'gold_response': gold_response,
            'user_utt_meta': user_utt_meta,
            'bot_utt_meta': bot_utt_meta,
            'agent_action_frames': agent_action_frames_meta,
            'gold_intent': gold_intent,
            'dialog_id': utt_user['meta']['dialog_id'],
            'utt_id': utt_user['utt_id'].split('-')[1]
        }
