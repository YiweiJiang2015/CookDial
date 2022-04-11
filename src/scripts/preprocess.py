import json, re, glob, spacy, sys, time
import logging, argparse, random
from pathlib import Path
from overrides import overrides
from collections import deque, namedtuple
import numpy as np
from typing import Tuple, Union

# add subproject dir to sys path
pwd = Path(__file__).resolve().__str__()
sys.path.append(re.sub(r'scripts.*', '', pwd))
sys.path.append(re.sub(r'src.*', '', pwd))

from utils.recipe_util import UtteranceBase, SessionBase, RecipeBase, EAMRParser, EAMRParseError
from utils.vocabulary import Vocabulary
from utils.util import write_json, read_json, padding
from utils.fparser import FuncStringParser, FstringParseError, IllegalArgError

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

SpanTuple = namedtuple('SpanTuple', 'name, mention, span_start, span_end')
SLOT_MATRIX_ROWS = 4  # set to 5 because in seq2seq, the START symbol won't emit scores
SLOT_MATRIX_COLS = 4


class Tokenizer:
    """A simple spacy tokenizer."""

    def __init__(self, lowercase=False):
        self.lowercase = lowercase
        self.nlp = spacy.load('en_core_web_sm')  # en_core_web_sm.load()

    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [tok.text.lower() if self.lowercase else tok.text for tok in doc]
        return tokens


class Meta:
    r"""
    Meta entries for utterances and instructions. Fields can be added or deleted dynamically.
    Fields:
        tokens:
        history:
        agent_acts_shift:
        arg_matrix:
        section:
    """

    def add_item(self, name, value):
        """Dynamically add meta field"""
        setattr(self, name, value)

    def del_item(self, name):
        """Dynamically delete meta field"""
        delattr(self, name)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Utterance(UtteranceBase):
    def __init__(self, utterance, from_serialized=False):
        super(Utterance, self).__init__(utterance)
        # Execute additional functions
        self.meta = Meta()
        if from_serialized:
            self.load_serialized_content_to_meta(utterance['meta'])
        self.preprocess_annotations()

    def load_serialized_content_to_meta(self, serialized_meta):
        for k, v in serialized_meta.items():
            self.meta.add_item(k, v)

    def add_meta(self):
        """
        Here we create gold labels for utterances. Copy attributes outside utt.meta to utt.meta
        """
        if self.bot:
            self.meta.add_item('agent_acts', self.get_agent_acts())
            full_set_arg_seq = self.get_simplified_arg_by_row()
            self.meta.add_item('full_set_arg_seq', full_set_arg_seq)
        else:
            items = ['section', 'intent',
                     'tracker_completed_step', 'tracker_completed_step_num',
                     'tracker_requested_step', 'tracker_requested_step_num']
            for item in items:
                try:
                    value = getattr(self, item)
                    if not isinstance(value, int) and (value is None or len(value) == 0):
                        logger.warning(
                            f'Find empty field % Dialog {self.dialog_id} {self.utt_id} `{item}` has empty value')
                    self.meta.add_item(item, getattr(self, item))
                except AttributeError:
                    logger.error(f'{item} is not in {self.dialog_id} {self}')

    def __getattr__(self, item):
        """Get attributes hidden in self.annotations_dict or self.meta"""
        try:
            return getattr(self.meta, item)
        except AttributeError:
            try:
                return self.annotations_dict[item]
            except KeyError:
                raise AttributeError(f'No attribute `{item}` in utterance')

    @overrides
    def to_dict(self):
        """Serialize for final datasets"""
        out_key = ['utt_id', 'utterance', 'bot', 'annotations']
        out_dict = {k: getattr(self, k) for k in out_key}
        out_dict['meta'] = self.meta.to_dict()
        return out_dict

    @classmethod
    def from_json(cls, serialized_content):
        return cls(serialized_content, from_serialized=True)

    def to_file_dict(self):
        """Serialize for file rewriting"""
        out_key = ['utt_id', 'utterance', 'bot', 'annotations']
        out_dict = {k: getattr(self, k) for k in out_key}
        return out_dict

    def set_label(self, label_name):
        """Copy the gold label from self.meta"""
        try:
            setattr(self, 'label', getattr(self.meta, label_name))
        except AttributeError:
            setattr(self, 'label', None)

    def split_raw_anno(self, raw_anno: str):
        fstrings = []
        for fstring in raw_anno.split('\r\n'):
            if fstring.strip() != '':
                fstrings.append(fstring)

        return fstrings

    def preprocess_annotations(self):
        """
        Remove some functions
        :return:
        """
        deserted_act_list = [r'start_ing_list', r'humor',
                     r'\bstart_instruction',
                     ]
        pattern_list = [f + r'\(\);(\r\n|)' for f in deserted_act_list]
        for p in pattern_list:
            self.annotations = re.sub(p, '', self.annotations, count=0)  # count may affect

    def get_parsed_funcs(self):
        if not self.bot:
            return
        # Parse func string
        fstrings = self.split_raw_anno(self.annotations)
        parsed_funcs = []
        for f in fstrings:
            try:
                parsed_funcs.append(FuncStringParser(f).parse())
            except (FstringParseError, IllegalArgError) as e:
                msg = str(e) + '\n'
                msg += ' ' * 4 + re.sub(r'\r\n', ' ', self.annotations) + '\n'
                msg += ' ' * 6 + self.meta.dialog_id + ' ' + self.utt_id
                logger.warning(msg)
        self.parsed_funcs = parsed_funcs

    def get_agent_acts(self):
        agent_acts = [f.name for f in self.parsed_funcs]
        if len(agent_acts) > 4:
            logger.warning(f"Dialog {self.meta.dialog_id} {self.utt_id} has {len(agent_acts)} agent action frames")
        return agent_acts

    def get_simplified_arg_by_row(self):
        """Each row only has one arg"""
        arg_sequence = [None] * SLOT_MATRIX_ROWS
        for i, f in enumerate(self.parsed_funcs):
            arg_sequence[i] = f.simplified_args
        return arg_sequence

    def get_args_by_row(self):
        """Args are factored into row vectors"""
        arg_matrix = [None] * SLOT_MATRIX_ROWS
        for i, f in enumerate(self.parsed_funcs):
            if len(self.parsed_funcs) > SLOT_MATRIX_ROWS:
                print()
            arg_matrix[i] = f.args_slots

        return arg_matrix

    def pad_list(self, l: list, max_len=3, pad_token=None):
        """Pad a list to max_len"""
        len_list = len(l)
        for j in range(max_len - len_list):
            l.append(pad_token)
        return l

    def annotation_append(self, key, value):
        """Append string to annotations and annotations_dict field"""
        assert not self.bot, 'Only user utterance can append strings to annotations fields.'
        self.annotations_dict[key] = value
        self.annotations = json.dumps(self.annotations_dict, indent=4)

    def max_num(self, section_type):
        """
        Find the max num of ing/inst in the bot utterance, e.g.
            inform(inst-9);
            inform(inst-10);
        the max num is 10
        :param section_type: ing or inst
        :return:
        """
        assert self.bot, 'Only bot utterance can invoke max_num().'
        if section_type == 'inst':
            pattern = re.compile(r'(inst|ac)-(\d+)')  # grouping will be captured separately
        if section_type == 'ing':
            pattern = re.compile(r'(ing)-(\d+)')
        matches = pattern.findall(self.annotations)
        if len(matches) == 0:
            return -1
        else:
            num = [int(m[1]) for m in matches]
            num.sort()
            return num[-1]

    def min_step(self, section_type) -> (Union[str, None], int):
        """
        Find the minimum num and step id of ing/inst in the bot utterance
        :param section_type: ing or inst
        :return:
        """
        assert self.bot, 'Only bot utterance can invoke min_step().'
        if section_type == 'inst':
            pattern = re.compile(r'(inst|ac)-(\d+)')  # grouping will be captured separately
        if section_type == 'ing':
            pattern = re.compile(r'(ing)-(\d+)')
        matches = pattern.findall(self.annotations)
        if len(matches) == 0:
            return None, -1
        else:
            num = [int(m[1]) for m in matches]
            num.sort()
            return section_type + '-' + str(num[0]), num[0]

    def parse_user_intents(self):
        intents = [intent.rstrip(';') for intent in self.intent.strip().split(' ')]
        return intents


class Session(SessionBase):
    def __init__(self, utt_cls,
                 file=None,
                 tokenizer=None,
                 from_serialized=False,
                 serialized_content=None
                 ):
        super(Session, self).__init__(utt_cls,
                                      file=file,
                                      from_serialized=from_serialized,
                                      serialized_content=serialized_content)
        self.tokenizer = tokenizer
        self.add_meta()
        self.knowledge_base = {}
        if from_serialized:
            self.knowledge_base = serialized_content['knowledge_base']

    @overrides
    def dump_file(self):
        'Rewrite the original json file'
        content = {"messages": [utt.to_file_dict() for utt in self.utterances.values()]}
        write_json(content, self.dialog_json_path)

    @overrides
    def to_dict(self):
        keys = ['dialog_id', 'utterances']
        return {'dialog_id': self.dialog_id,
                'utterances': [utt.to_dict() for utt in self.utterances.values()],
                'knowledge_base': self.knowledge_base}

    @classmethod
    def from_json(cls, tokenizer, serialized_content):
        return cls(Utterance,
                   tokenizer=tokenizer,
                   from_serialized=True,
                   serialized_content=serialized_content)

    def add_meta(self):
        """Create gold labels for utterances."""
        for utt in self.utterances.values():
            utt.meta.add_item('dialog_id', self.dialog_id)
            utt.get_parsed_funcs()
            utt.add_meta()  # all magical label creation happens here

    def shift_agent_action_frames(self):
        utt_list = list(self.utterances.values())
        utt_len = len(self)
        for utt_c, utt_n in zip(utt_list[:utt_len - 1], utt_list[1:]):
            if not utt_c.bot and utt_n.bot:
                utt_c.meta.add_item('agent_action_frames_shift', utt_n.annotations)

    def shift_agent_acts(self):
        utt_list = list(self.utterances.values())
        utt_len = len(self)
        for utt_c, utt_n in zip(utt_list[:utt_len - 1], utt_list[1:]):
            if not utt_c.bot and utt_n.bot:
                if len(utt_n.meta.agent_acts) == 0:
                    logger.warning('Found no func names in this utterance: either filtered or wrongly annotated.')
                    logger.warning(f'\t{self.dialog_id}\t{utt_n}\t{utt_n.annotations}')
                utt_c.meta.add_item('agent_acts_shift', utt_n.meta.agent_acts)

    def shift_agent_action_frame_args(self):
        utt_list = list(self.utterances.values())
        utt_len = len(self)
        for utt_c, utt_n in zip(utt_list[:utt_len - 1], utt_list[1:]):
            if not utt_c.bot and utt_n.bot:
                utt_c.meta.add_item('full_set_pointer_seq_shift', utt_n.meta.full_set_pointer_seq)

    def parse_user_intents(self):
        utt_list = list(self.utterances.values())
        utt_len = len(self)
        for utt_c, utt_n in zip(utt_list[:utt_len - 1], utt_list[1:]):
            if not utt_c.bot and utt_n.bot:
                utt_c.meta.add_item('parsed_intents', utt_c.parse_user_intents())

    def set_label(self, label_name):
        for utt in self.utterances.values():
            utt.set_label(label_name)

    def add_history(self, window=1):
        """
        Add conversation history to meta info
        :param window: number of previous utterances, defaulted as 1
        :return:
        """
        history = deque(maxlen=window)
        for i, utt in self.utterances.items():
            utt.meta.add_item('history_window', window)
            if i == '0':
                utt.meta.add_item('history', [''])
            else:
                utt.meta.add_item('history', list(history))
            history.append((utt.utterance.lower(), utt.bot) if self.tokenizer.lowercase
                           else (utt.utterance, utt.bot))

    def tokenize_utterance(self):
        """Tokenize current utterance not including the history"""
        for utt in self.utterances.values():
            utt_tokens = self.tokenizer.tokenize(utt.utterance)
            utt.meta.add_item('utt_tokens', utt_tokens)

    def tokenize_history(self):
        """Tokenize all sentences in utt.meta.history and concatenate them together"""
        for utt in self.utterances.values():
            if hasattr(utt, 'history'):
                his_tokens = []
                if len(utt.history[0]) > 0:  # check if there exists only an empty string
                    for his_sent, bot_mark in utt.history:
                        his_tokens.append((self.tokenizer.tokenize(his_sent), bot_mark))
                else:
                    his_tokens.append([[], False])
                utt.meta.add_item("history_tokens", his_tokens)
            else:
                raise AttributeError(f"{self.dialog_id}-{utt.utt_id} has no history in meta info")

    def token_len_check(self):
        """
        Check if len(history_token+utt_token) > 512
        :return:
        """

        # for window_size in history_window:
        for utt in self.utterances.values():
            total_len = 0
            if not utt.bot:
                total_len += len(utt.meta.utt_tokens)
                his_token_copy = [l for l in utt.meta.history_tokens]
                his_token_copy.reverse()
                for j in range(len(his_token_copy)):
                    total_len += len(his_token_copy[j])
                    if j > 1:  # only checks the last two history utterances.
                        break

    def get_response_ptr_text(self, pointer):
        if pointer == 0:
            return self.knowledge_base['title']['text']
        if pointer == 1:
            return self.knowledge_base['ing_list']['text']
        else:
            return self.knowledge_base['instructions'][pointer - 2]['text']


class Collection:
    """A base class for dialogue and recipe collections"""
    def __init__(self,
                 collection_path: str,
                 white_list: list = None,
                 tokenizer: Tokenizer = None,
                 from_serialized: bool = False,
                 serialized_content=None):
        self.tokenizer = tokenizer
        if not from_serialized:
            self.file_list = self.get_file_list(collection_path, white_list)
            self.collection: dict = self.load_all_files()  # The source of iterator
        else:
            self.collection = self.load_serialized_content(serialized_content)
        self.__len = len(self)

    def get_file_list(self, collection_path, white_list):
        all_files = glob.glob(collection_path + '/*.json')
        file_list = []
        pattern = re.compile(r'\d{3}')
        for file in all_files:
            m = pattern.match(file.split('/')[-1])
            if m[0] in white_list:
                file_list.append(file)
        file_list.sort(key=lambda x: int(pattern.match(x.split('/')[-1])[0]))  # sort by recipe_id
        return file_list

    def load_all_files(self) -> dict:
        raise NotImplementedError

    def load_serialized_content(self, serialized_content):
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, key):
        return self.collection[key]

    def __contains__(self, key):
        return True if key in self.collection else False

    def __iter__(self):
        self.iter_keys = iter(self.collection.keys())  # initialize iterator key
        return self

    def __next__(self):
        """
        Main implementation for iter(self). The iterator source is self.collection
        Example:
            foo_collection = Collection(path)
            for entry in foo_collection:
                func(entry)
        :return:
        """
        return self[next(self.iter_keys)]

    def get_keys(self):
        return list(self.collection.keys())

    def __sub__(self, other) -> dict:
        if not isinstance(other, Collection):
            raise TypeError("Both operands must be instances of Collection")
        diff = {}
        subtrahend_keys = other.get_keys()
        for key, item in self.collection.items():
            if key not in subtrahend_keys:
                diff[key] = item
        return diff

    @property
    def lowercase(self):
        """
        In some methods where tokenizes are not used, we shall still be consistent with the tokenizer lowercase setting.
        """
        return self.tokenizer.lowercase


class DialogCollection(Collection):
    def __init__(self,
                 collection_path: str = None,
                 white_list: list = None,
                 tokenizer: Tokenizer = None,
                 from_serialized: bool = False,
                 serialized_content=None):
        super(DialogCollection, self).__init__(collection_path, white_list,
                                               tokenizer=tokenizer,
                                               from_serialized=from_serialized,
                                               serialized_content=serialized_content)

    @overrides
    def load_all_files(self):
        collection = {}
        if len(self.file_list) > 0:
            collection = {Path(file).name.split('.')[0]: Session(Utterance, file=file,
                                                                 tokenizer=self.tokenizer)
                          for file in self.file_list}
        return collection

    @overrides
    def load_serialized_content(self, serialized_content: list):

        collection = {dialog['dialog_id']: Session.from_json(tokenizer=self.tokenizer,
                                                             serialized_content=dialog)
                      for dialog in serialized_content}
        return collection

    def set_label(self, label_name):
        for session in self:
            session.set_label(label_name)

    def shift_agent_action_frames(self):
        logger.info("Start to shift agent action frames to user utterances...")
        for session in self:
            session.shift_agent_action_frames()
        logger.info("\tFinish shifting agent action frames")

    def shift_agent_acts(self):
        logger.info("Start to shift agent acts...")
        for session in self:
            session.shift_agent_acts()
        logger.info("\tFinish shifting agent acts to user utterances.")

    def shift_agent_action_frame_args(self):
        logger.info("Start to shift agent action frame args to user utterances...")
        for session in self:
            session.shift_agent_action_frame_args()
        logger.info("\tFinish shifting agent action frame args")

    def parse_user_intents(self):
        logger.info("Start to parse intents of user utterances...")
        for session in self:
            session.parse_user_intents()
        logger.info("\tFinish parsing intents")

    def add_history(self, window=1):
        """
        Add conversation history to meta info
        :param window: number of previous utterances, defaulted as 1
        :return:
        """
        logger.info(f"Start to add history (window={window}) to each utterance...")
        for session in self:
            session.add_history(window)
        logger.info("\tFinish adding history.")

    def tokenize(self):
        logger.info("Start to tokenize utterance (and history)...")
        for session in self:
            session.tokenize_utterance()
            try:
                session.tokenize_history()
            except AttributeError as e:
                logger.warning(e)
        logger.info("\tFinish tokenization.")

    def create_vocabulary(self, vocabs):
        """Create a vocabulary from scratch and store it into json file"""
        logger.info("Build vocabularies from dialogues...")
        for session in self:
            for utt in session.utterances.values():
                vocabs['words'].add_many(utt.meta.utt_tokens)
                if not utt.bot:
                    if hasattr(utt.meta, 'agent_acts_shift'):
                        vocabs['agent_acts'].add_many(utt.meta.agent_acts_shift)
                    if hasattr(utt.meta, 'section'):
                        vocabs['section'].add_token(utt.meta.section)
                    if hasattr(utt.meta, 'parsed_intents'):
                        vocabs['intent'].add_many(utt.meta.parsed_intents)
        logger.info(f"\tFinish building vocabs from dialogues: {list(vocabs.keys())}")

    def to_dict(self):
        return {'dialogues': [dialog.to_dict() for dialog in self]}

    @classmethod
    def from_json(cls, serialized_path):
        dialogues = read_json(serialized_path)['dialogues']
        return cls(from_serialized=True, serialized_content=dialogues)

    def rewrite_files(self):
        logger.info("Start to rewrite the original json files...")
        for dialog in self:
            dialog.dump_file()
        logger.info("\tFinish rewriting json files.")

    def token_len_check(self):
        for dialog in self:
            dialog.token_len_check()


class SubDialogCollection(DialogCollection):
    def __init__(self, collection: dict):
        self.collection = collection


class NodeMap:
    """
    Map a valid node in eamr graph to its corresponding position in the flattened array of all valid nodes.
    """

    def __init__(self, pos_in_array, loc_id, span: SpanTuple):
        self.pos_in_array = pos_in_array
        self.loc_id = loc_id
        self.name = span.name
        self.mention = span.mention
        self.span = (span.span_start, span.span_end)

    @classmethod
    def from_span(cls, global_pos, loc_id, span: SpanTuple):
        return cls(global_pos, loc_id, span)

    def to_array_entry(self):
        return (self.name, self.mention, self.span)

    def __repr__(self):
        return f"'{self.name}' '{self.mention}' pos: {self.pos_in_array}, {self.span}"


class NodeBook:
    """
    A keeping-book that stores a dict of NodeMap instances.
    """

    def __init__(self, valid_nodes: dict, num_nodes: int):
        self._valid_nodes = valid_nodes
        self._num_nodes = num_nodes

    def get_gold_array_on_spans(self, arg_array):
        """Label each span with correct slot position."""
        gold_array = np.zeros((self.num_nodes), dtype=np.int)
        if all([arg is None for arg in arg_array]):
            return gold_array
        # first arg
        if 'title' in arg_array[0]:
            pos = self._valid_nodes['title'].pos_in_array
            gold_array[pos] = 1
            return gold_array
        # first arg
        if 'ac' in arg_array[0]:
            inst_id = 'inst-' + arg_array[0].split('-')[1]
            pos = self.get_pos_in_array(inst_id, arg_array[0])
            gold_array[pos] = 1
        elif 'inst' in arg_array[0]:
            inst_id = arg_array[0]
            pos = self.get_pos_in_array(inst_id, arg_array[0])
            gold_array[pos] = 1
        # second arg
        if arg_array[1] is not None:
            pos = self.get_pos_in_array(inst_id, arg_array[1])
            gold_array[pos] = 2
        # third arg
        if arg_array[2] is not None:
            pos = self.get_pos_in_array(inst_id, arg_array[2])
            gold_array[pos] = 3
        # fourth arg
        if arg_array[3] is not None:
            pos = self.get_pos_in_array(inst_id, arg_array[1])
            gold_array[pos] = 4
        return gold_array

    def get_gold_array_on_slots(self, arg_array):
        """ """
        gold_array = np.zeros((4), dtype=np.int)
        inst_id = None
        if all([arg is None for arg in arg_array]):
            return gold_array
        # first arg
        if 'title' in arg_array[0]:
            pos = self._valid_nodes['title'].pos_in_array
            gold_array[0] = pos
            return gold_array
        # first arg
        if 'ac' in arg_array[0]:
            inst_id = 'inst-' + arg_array[0].split('-')[1]
            pos = self.get_pos_in_array(inst_id, arg_array[0])
            gold_array[0] = pos
        elif 'inst' in arg_array[0]:
            inst_id = arg_array[0]
            pos = self.get_pos_in_array(inst_id, arg_array[0])
            gold_array[0] = pos
        # second arg
        if arg_array[1] is not None:
            if inst_id is None:
                print()
            pos = self.get_pos_in_array(inst_id, arg_array[1])
            gold_array[1] = pos
        # third arg
        if arg_array[2] is not None:
            pos = self.get_pos_in_array(inst_id, arg_array[2])
            gold_array[2] = pos
        # fourth arg
        if arg_array[3] is not None:
            pos = self.get_pos_in_array(inst_id, arg_array[1])
            gold_array[3] = pos
        return gold_array

    def to_array(self):
        node_array = [self._valid_nodes[key].to_array_entry() for key in self._valid_nodes.keys() if 'inst' not in key]
        for key in self._valid_nodes.keys():
            if 'inst' in key:
                t_array = [item.to_array_entry() for item in self._valid_nodes[key]]
                node_array.extend(t_array)

        return node_array

    def get_gold_full_set_pointer_pos(self, arg_full_set_ptr):
        """Used for the paper: agent action frame task"""
        inst_id = None
        if arg_full_set_ptr is None:
            return 0
        if 'title' in arg_full_set_ptr:
            pos = self._valid_nodes['title'].pos_in_array
        elif 'ac' in arg_full_set_ptr:
            inst_id = 'inst-' + arg_full_set_ptr.split('-')[1]
            pos = self.get_pos_in_array(inst_id, inst_id)
        elif 'inst' in arg_full_set_ptr:
            inst_id = arg_full_set_ptr
            pos = self.get_pos_in_array(inst_id, arg_full_set_ptr)
        elif 'ing' in arg_full_set_ptr:
            inst_id = None
            pos = self.get_pos_in_array(inst_id, arg_full_set_ptr)
        else:
            try:
                inst_id = 'inst-' + arg_full_set_ptr.split('-')[1]
            except IndexError:
                print()
            pos = self.get_pos_in_array(inst_id, arg_full_set_ptr)
        return pos

    def get_pos_in_array(self, inst_id, node_name):
        """Get position in the flattened node array"""
        pos = None
        if inst_id is not None:
            try:
                for node in self._valid_nodes[inst_id]:
                    if node.name == node_name:
                        pos = node.pos_in_array
            except KeyError:
                print()
        else:
            # if pos is None:
            if 'ing' in node_name:
                try:
                    pos = self._valid_nodes[node_name].pos_in_array
                except KeyError:
                    print()

        if pos is None:
            raise RuntimeError('Cannot find the node in this NodeBook.')
        return pos

    def to_dict(self):
        return

    @classmethod
    def from_nodes(cls, valid_nodes, num_nodes):
        return cls(valid_nodes, num_nodes)

    @property
    def num_nodes(self):
        return self._num_nodes


class Ingredient(RecipeBase):
    def __init__(self, item, tokenizer: Tokenizer):
        super().__init__(item)
        self.meta = Meta()
        self.tokenizer = tokenizer
        self.quantity = None
        self.ingredient = None
        try:
            self.graph = EAMRParser(self.eamr).parse()
            # self.valid_nodes = self.get_all_valid_nodes()
        except EAMRParseError as e:
            self.graph = None
            logger.debug("ERROR", e)
            logger.debug("->", self.eamr)

    def get_all_valid_nodes(self):
        nodes = self.graph.get_nodes(filter=lambda x: x.span is not None and x.label != 'R')
        assert len(nodes) == 1, f'Ingredient {self.id} {self.text} has more than 1 span'
        valid_nodes = []
        for node in nodes:
            # global_s, global_e = self.get_global_start_end(node.span.mention)
            valid_nodes.append(SpanTuple(node.name, node.span.mention, self.meta.boundary[0], self.meta.boundary[1]))
            # valid_nodes[node.name] = (node.span.mention, global_s, global_e)
        return valid_nodes

    def to_dict(self):
        return {"id": self.id, "text": self.text, "eamr": self.eamr, "meta": self.meta.to_dict()}


def find_span(target, sentence, tokenizer: Tokenizer):
    """
    Find entity span in corresponding sentence and return (start_id, end_id)
    :param target: string
    :param sentence: list of tokens
    :return: (start_id, end_id)
    """
    target_tokens = [token.lower() for token in tokenizer.tokenize(target)]
    target_length = len(target_tokens)
    span_results = []
    for i in range(0, len(sentence)):
        if sentence[i:(i + target_length)] == target_tokens:
            span_results.append(
                (i, i + target_length - 1))  # minus 1 because we use end point span representation in the model
    if len(span_results) > 1:
        logger.warning(f'  There are multiple matches for span "{target}"')
    if len(span_results) == 0:
        logger.warning(f'  There are no match for span "{target}"')
    return span_results[0]


class Instruction(RecipeBase):
    def __init__(self, item, tokenizer: Tokenizer):
        super().__init__(item)
        self.meta = Meta()
        self.tokenizer = tokenizer
        try:
            self.graph = EAMRParser(self.eamr).parse()
            # self.valid_nodes = self.get_all_valid_nodes()
        except EAMRParseError as e:
            self.graph = None
            logger.debug("ERROR", e)
            logger.debug("->", self.eamr)

    def get_global_start_end(self, mention):
        """Get global token (start, end) for a span mention"""
        local_start, local_end = find_span(mention, self.meta.tokens, self.tokenizer)
        global_s, global_e = local_start + self.meta.boundary[0], local_end + self.meta.boundary[0]
        return global_s, global_e

    def get_all_valid_nodes(self):
        """
        Valid nodes: parent edge label does not start with "_"
        """
        valid_nodes = []
        valid_nodes.append(SpanTuple(self.id, self.text, self.meta.boundary[0], self.meta.boundary[1]))
        valid_labels = ['TEMPERATURE', 'TOOL', 'DUR', 'CONDITION_CLAUSE']
        # in recipe 000-010, some intermediate result nodes (:_result) are fasely taken as positive.
        # for now, I have no better method to exclude them.
        nodes = self.graph.get_nodes(filter=lambda x: x.span is not None and x.name is not None
                                                      and not x.name.startswith('ing-') and x.label in valid_labels)
        for node in nodes:
            global_s, global_e = self.get_global_start_end(node.span.mention)
            valid_nodes.append(SpanTuple(node.name, node.span.mention, global_s, global_e))
        return valid_nodes

    def to_dict(self):
        return {"id": self.id, "text": self.text, "eamr": self.eamr, "meta": self.meta.to_dict()}


class RecipeTitle:
    def __init__(self, title_string=None):
        self.text = title_string
        self.meta = Meta()

    def get_all_valid_nodes(self):
        valid_nodes = []
        valid_nodes.append(SpanTuple('title', self.text, self.meta.boundary[0], self.meta.boundary[1]))
        return valid_nodes

    @classmethod
    def from_title_string(cls, title):
        return cls(title)

    def to_dict(self):
        return {"text": self.text, "meta": self.meta.to_dict()}


class RecipeTxt:
    """Store one recipe json."""
    def __init__(self, file: str, tokenizer: Tokenizer):
        self.recipe_json_path = Path(file)
        self.recipe_id = ''
        self.tokenizer = tokenizer
        self.title = RecipeTitle()
        self.ingredients: dict[str: Ingredient] = {}
        self.instructions: dict[str: Instruction] = {}
        self.load_json_file()
        self.graph_nodes = None

    def load_json_file(self):
        with open(self.recipe_json_path, 'r') as fp:
            dict = json.load(fp)
            self.recipe_id = dict["id"]
            self.title = RecipeTitle.from_title_string(dict["title"])
            self.ingredients = {item["id"]: Ingredient(item, self.tokenizer) for item in dict["content"]
                                if item["type"] == "ingredient"}
            self.instructions = {item["id"]: Instruction(item, self.tokenizer) for item in dict["content"]
                                 if item["type"] == "instruction"}

    def tokenize(self):
        """Tokenize title, instructions, ingredients"""
        recipe_token_len = 0
        tokens = self.tokenizer.tokenize(self.title.text)
        self.title.meta.add_item('tokens', tokens)
        recipe_token_len += len(tokens)
        for inst in self.instructions.values():
            tokens = self.tokenizer.tokenize(inst.text)
            inst.meta.add_item('tokens', tokens)
            recipe_token_len += len(tokens)
        for ing in self.ingredients.values():
            tokens = self.tokenizer.tokenize(ing.text)
            ing.meta.add_item('tokens', tokens)
            recipe_token_len += len(tokens)
        if recipe_token_len > 512:
            logger.warning(f'Recipe {self.recipe_id} has more than 512 tokens.')

    def get_sent_boundary(self):

        end = len(
            self.title.meta.tokens) - 1  # first inst is appended after the ingredient list in the input to the model
        self.title.meta.add_item('boundary', [0, end])
        # ing token boundary
        for ing in self.ingredients.values():
            if not hasattr(ing.meta, 'tokens'):
                raise RuntimeError(f'{self.recipe_id} inst-{ing.id} is not tokenized')
            else:
                start = end + 1
                end += len(ing.meta.tokens)
                ing.meta.add_item('boundary', [start, end])
        # inst token boundary
        for inst in self.instructions.values():
            if not hasattr(inst.meta, 'tokens'):
                raise RuntimeError(f'{self.recipe_id} inst-{inst.id} is not tokenized')
            else:
                start = end + 1
                end += len(inst.meta.tokens)
                inst.meta.add_item('boundary', [start, end])

    def get_all_valid_nodes(self):
        """
        Find valid nodes including ingredients, title, instruction entities
        :return: valid_nodes:
                 array_pos+1: the num of valid nodes
        """
        valid_nodes = {}
        array_pos = 1  # The first node shall be null node which does not appear here but will be added in the model.
        valid_nodes['title'] = NodeMap(array_pos, 'title', self.title.get_all_valid_nodes()[0])

        for ing_id, ing in self.ingredients.items():
            array_pos += 1
            valid_nodes[ing_id] = NodeMap(array_pos, ing_id, ing.get_all_valid_nodes()[0])
        for inst_id, inst in self.instructions.items():
            valid_nodes[inst_id] = []
            for span_t in inst.get_all_valid_nodes():
                array_pos += 1
                valid_nodes[inst_id].append(NodeMap(array_pos, inst_id, span_t))
        self.graph_nodes = NodeBook.from_nodes(valid_nodes, array_pos + 1)

    def get_node_span(self, node_id=None, loc_id=None):

        def assemble_mention_span(inst_id, node_id):
            span_l = self.instructions[inst_id].graph.get_spans(lambda x: x.name == node_id)
            # assert len(span_l) == 1, f'{node_id} matches more than one span.'
            if len(span_l) > 1:
                logger.warning(f'\nRecipe{self.recipe_id} {inst_id} {node_id} matches more than one span.')
                print(span_l)
            elif len(span_l) == 0:
                logger.warning(f'\nRecipe{self.recipe_id} {inst_id} {node_id} matches nothing.')
                return None
            span_mention = span_l[0].mention
            # get global span start, end
            span_s, span_e = self.instructions[inst_id].get_global_start_end(span_mention)
            return [span_mention, span_s, span_e]

        if 'inst' in node_id:
            return [node_id] + self.instructions[node_id].meta.boundary

        if 'ac' in node_id:
            inst_id = 'inst-' + node_id.split('-')[1]
            return assemble_mention_span(inst_id, node_id)
        ## Normalize loc_id
        if loc_id == None:
            print()
        if 'inst' in loc_id:
            inst_id = loc_id
        else:  # ac in loc_id
            inst_id = 'inst-' + loc_id.split('-')[1]
        ## Normalize ends
        if 'ing' in node_id or 'tool' in node_id:
            return assemble_mention_span(inst_id, node_id)

    def __repr__(self):
        return f"Recipe {self.recipe_id}: {len(self.ingredients)} ingredients, {len(self)} instructions"

    def __len__(self):
        """Number of instruction steps"""
        return len(self.instructions)

    def to_dict(self):
        return {'recipe_id': self.recipe_id,
                'title': self.title.to_dict(),
                'ing_list': {'text': ', '.join([ing.text for ing in self.ingredients.values()])},
                'ingredients': [ing.to_dict() for ing in self.ingredients.values()],
                'instructions': [inst.to_dict() for inst in self.instructions.values()],
                'nodes': self.graph_nodes.to_array() if self.graph_nodes is not None else None}


class RecipeCollection(Collection):
    def __init__(self,
                 collection_path: str,
                 white_list: list,
                 tokenizer: Tokenizer = None,
                 from_serialized: bool = False):
        super(RecipeCollection, self).__init__(collection_path,
                                               white_list,
                                               tokenizer=tokenizer,
                                               from_serialized=from_serialized)

    def load_all_files(self):
        collection = {}
        if len(self.file_list) > 0:
            collection = {Path(file).name.split('_')[0]: RecipeTxt(file, tokenizer=self.tokenizer) for file in
                          self.file_list}
        return collection

    def tokenize(self):
        for recipe_txt in self:
            recipe_txt.tokenize()

    def create_vocabulary(self, vocabs):
        logger.info("Build vocabularies from recipes...")
        for recipe_txt in self:
            vocabs['words'].add_many(recipe_txt.title.meta.tokens)
            for inst in recipe_txt.instructions.values():
                vocabs['words'].add_many(inst.meta.tokens)
            for ing in recipe_txt.ingredients.values():
                vocabs['words'].add_many(ing.meta.tokens)
        logger.info("\tFinish building vocabs from recipes.")
        logger.info(f"\tVocab size: {len(vocabs['words'])}")

    def get_sent_boundary(self):
        """Compute the global token boundary of each instruction"""
        for recipe_txt in self:
            recipe_txt.get_sent_boundary()

    def get_all_valid_nodes(self):
        for recipe_txt in self:
            recipe_txt.get_all_valid_nodes()

    def to_dict(self):
        return {"recipes": [recipe.to_dict() for recipe in self]}


class Sampler:
    def __init__(self, ratio: tuple = (0.2, 0.2), seed=1111):
        self.ratio = ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.randgen = random.randrange

    def split(self, collection: Collection) \
            -> Tuple[SubDialogCollection, SubDialogCollection, SubDialogCollection]:
        spill = {}
        len_collection = len(collection)
        memo = -1
        test_id_list = []
        dev_id_list = []
        for i in range(int(self.ratio[1] * len_collection)):
            rand = self.randgen(0, len_collection)
            idx = padding(rand)
            if rand == memo or idx not in collection:
                continue
            else:
                spill[idx] = collection[idx]
                memo = rand
                test_id_list.append(idx)

        test_collection = SubDialogCollection(spill)
        train_dev_collection = SubDialogCollection(collection - test_collection)

        # clear spill
        spill = {}
        for i in range(int(self.ratio[0] * len_collection)):
            rand = self.randgen(0, len_collection)
            idx = padding(rand)
            if rand == memo or idx in test_id_list or idx not in collection:
                continue
            else:
                spill[idx] = collection[idx]
                memo = rand
                dev_id_list.append(idx)
        dev_collection = SubDialogCollection(spill)
        train_collection = SubDialogCollection(train_dev_collection - dev_collection)
        return train_collection, dev_collection, test_collection

    def percentile_split(self, collection: Collection):
        """
        Split train for the learning curve.
        """
        start_percentile = 0.2
        percentile_step = 0.1
        sub_collections = {}
        idx_full = np.asarray(collection.get_keys())
        np.random.shuffle(idx_full)
        for percentile in np.arange(start_percentile, 1.0, step=percentile_step):
            sub_collection_idx = idx_full[:int(percentile * len(collection))]

            spill = {idx: collection[idx] for idx in sub_collection_idx}
            sub_collections[int(percentile * 100)] = SubDialogCollection(spill)
        return sub_collections

    def __repr__(self):
        return f"split_ratio: {self.ratio}, seed: {self.seed}"


class PreProcess:
    def __init__(self, dialog_collection_path: str, recipe_collection_path: str, white_list: list,
                 dump_path: str, prediction_path: str, vocab_config, args):
        self._dump_path = dump_path
        self.check_dump_path()
        self._dialog_merged_target = dump_path + '/dialog/cookdial_dialog_merged.json'
        self._recipe_target = dump_path + '/recipe/recipe_fix.json'
        self._vocab_path = dump_path + '/vocab'
        self._prediction_path = prediction_path
        self.tokenizer = Tokenizer(lowercase=args.lowercase)
        self.dialogs = DialogCollection(dialog_collection_path, white_list, tokenizer=self.tokenizer)
        self.recipes = RecipeCollection(recipe_collection_path, white_list, tokenizer=self.tokenizer)
        self.vocabs = Vocabulary.from_config(vocab_config)
        self.history_window = args.history_window

        # init data sampler
        self.split = args.split
        self.percentile_split = args.percentile_split
        self.sampler = Sampler(ratio=args.ratio, seed=args.seed)

        if self.split:
            self.train_set, self.dev_set, self.test_set = None, None, None
            self._dialog_train_target = dump_path + '/dialog/cookdial_dialog.train.json'
            self._dialog_dev_target = dump_path + '/dialog/cookdial_dialog.dev.json'
            self._dialog_test_target = dump_path + '/dialog/cookdial_dialog.test.json'
        if self.percentile_split:
            self.train_subsets = None

    def check_dump_path(self):
        dp = Path(self._dump_path)
        dp_dialog = dp / 'dialog'
        dp_vocab = dp / 'vocab'
        if not dp.exists():
            logger.warning(f"{self._dump_path} does not exist. Creat it now.")
            dp.mkdir()
        if not dp_dialog.exists():
            dp_dialog.mkdir()
        if not dp_vocab.exists():
            dp_vocab.mkdir()

    def integrate_recipes(self):
        """Add each recipe to its corresponding dialogue"""
        logger.info('Start to integrate recipes into dialogues...')
        for session in self.dialogs:
            for recipe_txt in self.recipes:
                if session.dialog_id == recipe_txt.recipe_id:
                    session.knowledge_base = recipe_txt.to_dict()
        logger.info('\tIntegrating recipes is done.')

    def dialog_args_to_gold_full_set_pointer_sequence(self):
        logger.info('Start converting func-args to full_set pointer sequence...')
        for session in self.dialogs:
            recipe_txt = self.recipes[session.dialog_id]
            for utt in session.utterances.values():
                if utt.bot and hasattr(utt.meta, 'full_set_arg_seq'):
                    gold_full_set_pointer_seq = np.zeros((SLOT_MATRIX_ROWS), dtype=np.int)
                    frame_arg_seq = utt.meta.full_set_arg_seq
                    for i, arg_full_set_pointer in enumerate(frame_arg_seq):
                        if arg_full_set_pointer is None:
                            continue
                        else:
                            gold_full_set_pointer = recipe_txt.graph_nodes.get_gold_full_set_pointer_pos(
                                arg_full_set_pointer)
                            try:
                                gold_full_set_pointer_seq[i] = gold_full_set_pointer
                            except TypeError:
                                print()
                    utt.meta.add_item('full_set_pointer_seq', gold_full_set_pointer_seq.tolist())

        logger.info('\tFinish full_set pointer sequence conversion.')

    def dump_dialogs(self):
        if self.split:
            write_json(self.train_set.to_dict(), self._dialog_train_target)
            write_json(self.dev_set.to_dict(), self._dialog_dev_target)
            write_json(self.test_set.to_dict(), self._dialog_test_target)
        else:
            write_json(self.dialogs.to_dict(), self._dialog_merged_target)
        if self.percentile_split:
            for percentile, subset in self.train_subsets.items():
                subset_target = self._dump_path + f'/dialog/percentiles/dialog_fix.train.{percentile}.json'
                write_json(subset.to_dict(), subset_target)

    def dump_recipes(self):
        write_json(self.recipes.to_dict(), self._recipe_target)

    def dump_vocabs(self):
        for name, vocab in self.vocabs.items():
            vocab.to_file(parent=self._vocab_path)

    def split_data(self):
        logger.info('Start to split data into train and dev...')
        self.train_set, self.dev_set, self.test_set = self.sampler.split(self.dialogs)
        logger.info(f'\tTrain instances: {len(self.train_set)}')
        logger.info(f'\tDev instances: {len(self.dev_set)}')
        logger.info(f'\tTest instances: {len(self.test_set)}')
        logger.info('\tData splitting is done.')

    def split_train_data(self):
        logger.info('Start to split train data into multiple sets...')
        self.train_subsets = self.sampler.percentile_split(self.train_set)
        logger.info('\tTrain data splitting is done.')

    def refactor(self):
        """Collate dialogue and recipe files"""
        logger.info('Collate dialogue and recipe files')
        self.dialogs.shift_agent_action_frames()
        self.dialogs.shift_agent_acts()
        self.dialogs.parse_user_intents()
        self.dialogs.add_history(window=self.history_window)
        self.dialogs.tokenize()
        self.dialogs.create_vocabulary(self.vocabs)

        self.recipes.tokenize()
        self.recipes.get_sent_boundary()
        self.recipes.create_vocabulary(self.vocabs)
        self.recipes.get_all_valid_nodes()
        self.integrate_recipes()

        self.dialog_args_to_gold_full_set_pointer_sequence()

        self.dialogs.shift_agent_action_frame_args()

        if self.split:
            # split data into train and dev
            self.split_data()
            if self.percentile_split:
                # slice train data into multiple subsets
                self.split_train_data()
        else:
            logger.info(f'All data is merged into one file {self._dialog_merged_target}')

        self.dump_dialogs()
        self.dump_vocabs()

    def refactor1(self):
        self.dialogs.rewrite_files()

def main(args):
    """Main entry"""
    start_time = time.time()
    data_dir = re.sub(r'src/.*', 'data', pwd)
    config_dir = re.sub(r'scripts.*', args.config, pwd)
    dialog_collection_dir = data_dir + '/dialog'
    recipe_collection_dir = data_dir + '/recipe'

    dump_dir = data_dir + '/processed'
    prediction_dir = data_dir + '/prediction'
    log_dir = dump_dir + '/preprocess.log'
    logging.basicConfig(filename=log_dir, filemode='w', level=logging.DEBUG)

    vocab_config = read_json(config_dir)["vocabularies"]

    file_range = (0, args.file_range)
    exception_list = {}
    file_list = {i for i in range(*file_range)}
    file_list = file_list.difference(exception_list)
    white_list = [padding(file) for file in file_list]
    logger.info(f"File white list: {white_list}")
    logger.info(f"File black list: {exception_list}")
    logger.info("**********Preprocess starts***********\n")
    preprocess = PreProcess(dialog_collection_dir, recipe_collection_dir, white_list,
                            dump_dir, prediction_dir, vocab_config, args)
    preprocess.refactor()
    end_time = time.time()
    logger.info(f"**********Preprocess is finished. Elapsed time: {(end_time - start_time) / 60:^5.2f} min**********")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='vocab config file name')
    parser.add_argument('-fr', '--file-range', default=260, type=int, help='upper bound of the file id to be processed')
    parser.add_argument('-hw', '--history-window', default=10, type=int, help='dialogue history window size')
    # check --lowercase arg in PyCharm running env
    parser.add_argument('--lowercase', required=True, default=True, type=bool, help='all words are in lowercase or not')
    parser.add_argument('-ps', '--percentile-split', default=False, type=bool,
                        help='slices train data into multiple subsets')
    parser.add_argument('-r', '--ratio', default=(0.15, 0.1), type=tuple, help='split ratio for dev and test data')
    parser.add_argument('--seed', default=1321, type=int, help='random seed for random library')
    parser.add_argument('--split', default=False, type=bool, help='split the data into train and dev, test')
    args = parser.parse_args()
    main(args)
