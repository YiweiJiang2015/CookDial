import json
from pathlib import Path
from collections import OrderedDict

class OOVError(Exception):
    pass


class Vocabulary:
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, name=None, suffix=None, token_to_idx=None, add_unk=True, unk_token="<UNK>",
                 add_pad=False, pad_token="<PAD>",
                 add_start=False, start_token="<START>",
                 add_end=False, end_token="<END>"):
        """
        Args:
            name (str): name of this vocab, e.g. words, entity-tags, relation-tags
            suffix: json, txt, jsonl, etc. Decides how to store the vocab.
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to be added into the Vocabulary

        """
        if token_to_idx is None:
            token_to_idx = {}
        self._name = name
        self._suffix = suffix
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx, in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self._add_pad = add_pad
        self._pad_token = pad_token

        self._add_start = add_start
        self._start_token = start_token

        self._add_end = add_end
        self._end_token = end_token

        self.unk_index = -1

        if add_unk:
            self.unk_index = self.add_token(unk_token)
        if add_pad:
            self.pad_index = self.add_token(pad_token)
        if add_start:
            self.start_index = self.add_token(start_token)
        if add_end:
            self.end_index = self.add_token(end_token)


    def to_dict(self):
        """ returns a dictionary that can be serialized """
        #return {key: getattr(self, key) for key in self.__dict__.keys()}
        return {'name': self._name,
                'suffix': self._suffix,
                'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk, 'unk_token': self._unk_token,
                'add_pad': self._add_pad, 'pad_token': self._pad_token,
                'add_start': self._add_start, 'start_token': self._start_token,
                'add_end': self._add_end, 'end_token': self._end_token}

    @classmethod
    def from_dict(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary
            loaded from json file
        """
        return cls(**contents)

    def to_file(self, parent):
        r"""
        Write the dict to a file.
        :param parent:
        :return:
        """
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)

        out_dir = parent / (f"vocab.{self._name}.{self._suffix}")
        if self._suffix == 'json':
            self.to_json_file(out_dir)
        else:
            raise NotImplementedError('Vocab cannot handle writing files other json.')

    def to_json_file(self, out_dir: Path):
        with out_dir.open('w') as fp:
            json.dump(self.to_dict(), fp, indent=4)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        try:
            index = self._token_to_idx[token]
        except KeyError:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise OOVError("the index(%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    @classmethod
    def from_config(cls, config: dict):
        r"""
        Initial empty or default vocabs from cfg files.
        :param config:
        :return: A dict composed of vocabularies
        """
        vocabs = {}
        for name, args in config.items():
            vocabs[name] = cls(name, **args)
        return vocabs

    def __repr__(self):
        return f"<{self._name} vocabulary(size={len(self)})>"

    def __len__(self):
        return len(self._token_to_idx)

    def __add__(self, other):
        'Merge two vocab'
        if not isinstance(other, Vocabulary):
            raise TypeError('Only two instances of Vocabulary can be merged.')
        else:
            if self._name != other._name:
                raise ValueError('Only the same type of vocabularies can be merged.')
            else:
                new_vocab = Vocabulary(self._name)

                return NotImplementedError

    def __radd__(self, other):
        return self + other

    @property
    def size(self):
        return len(self)

    def get_index_to_token_vocabulary(self):
        'return the python dict'
        return self._idx_to_token

    def get_label_space(self, drop=None):
        """
        Get label space ordered by index
        drop: a list of tokens that won't appear in the returned label_dict
        """
        if drop is not None:
            label_dict = OrderedDict()
            for id, token in self._idx_to_token.items():
                if token not in drop:
                    label_dict[id] = token
            return label_dict
        else:
            return OrderedDict(sorted(self._idx_to_token.items()))