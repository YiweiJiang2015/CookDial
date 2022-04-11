import logging

logger = logging.getLogger(__name__)


class IllegalArgError(Exception):
    def __repr__(self):
        return 'IllegalArgError: '
    pass


class ParsedFuncString:
    'A toy class representing the function structure'
    def __init__(self, string, func_name, args):
        self.string = string
        self.name = func_name
        if not self.pos_is_legal(args):
            raise IllegalArgError('Illegal Argument order or combinations')
        self.args = args
        self.args_slots = self.flatten_args()
        self.args_num = len(self.args)
        # self.expand_args = {}
        # Attributes below only works for recipe language
        if len(args) > 0:
            #self.args = OrderedArgs(args)
            self.ing_list = self.get_ing_list()
            self.verb = self.get_verb()
            self.tool_list = self.get_tool_list()
            self.inst_id = self.get_inst_id()
            self.title = self.get_title()
        self.simplified_args = self.get_simplified_argument()

    def flatten_args(self):
        args_slots = [None]*4  # {i: None for i in range(4)}
        if len(self.args):
            args_slots[0] = self.args[0]
            if 1 in self.args:
                for j in range(len(self.args[1])):
                    args_slots[j+1] = self.args[1][j]
                    if j+1 > 3:
                        logger.warning(f'String "{self.string}" has more than 4 args')
                        break
        return args_slots

    def pos_is_legal(self, args):
        l = len(args)
        if l == 0:
            return True
        elif l == 1:
            if isinstance(args[0], str):
                return True
            else:
                return False
        elif l == 2:
            if not isinstance(args[0], str) or not isinstance(args[1], list):
                return False
            else:
                return True

    def get_verb(self):
        if self.args[0].startswith('ac'):
            return self.args[0]
        else:
            return None

    def get_tool_list(self):
        if self.args_num > 1 and 'tool' in self.args[1][0]:
            return self.args[1]
        elif self.args_num == 1 and 'tool' in self.args[0]:
            return [self.args[0]]
        else:
            return None

    def get_ing_list(self):
        if self.args_num > 1 and 'ing' in self.args[1][0]:
            return self.args[1]
        elif self.args_num == 1 and 'ing' in self.args[0]:
            return [self.args[0]]
        else:
            return None

    def get_inst_id(self):
        if self.args_num == 0:
            return None
        composed_prefix_list = ['inst-', 'ac-', 'temp-', 'dur-', 'cond-', 'purp-', 'tool-']
        for prefix in composed_prefix_list:
            if self.args[0].startswith(prefix):
                return 'inst-'+self.args[0].split('-')[1]
        single_prefix_list = ['title', 'ing-']
        for prefix in single_prefix_list:
            if self.args[0].startswith(prefix):
                return None
        # no match will raise error
        raise ValueError(f'No explicit mention of inst id in {self.string} {self.args}')
            # return None

    def get_title(self):
        if self.args[0].startswith('title'):
            return self.args[0]
        else:
            return None

    def get_simplified_argument(self):
        "Pick the salient arg"
        if self.args_num == 0:
            return None
        elif self.args_num == 1:
            black_list = ['purp', 'ac'] # ac
            for b in black_list:
                if self.args[0].startswith(b):
                    return self.get_inst_id()
                else:
                    return self.args[0]
        elif self.args_num > 1:
            if self.ing_list is not None:
                return self.get_inst_id()
            if self.tool_list is not None:
                return self.tool_list[0]
            raise ValueError(f'Cannot find a salient arg {self.string}')
            # temp, dur, cond



    def to_dict(self):
        return {'func_name': self.name,
                'args': self.args}

    def __repr__(self):
        return f"string:\t\"{self.string}\" \nfunc_name: <{self.name}>\targs: {self.args}"



class FstringParseError(Exception):
    pass

class FuncStringParser:

    def __init__(self, text):
        self.text = text
        self.reader = Reader(text)

    def parse(self):
        func = self.parse_func()
        self.reader.end_of_stream()  # verify if the reader memory (i.e. reader.text) runs out
        return func

    def parse_func(self):
        func_name = None
        args = {}
        arg_id = 0
        ch: str = self.reader.peek()

        # func name
        if ch.isalnum():
            func_name = self.parse_func_name()
            # ch = self.reader.peek()

        # func args
        while self.reader.peek() != ')':
            ## first arg
            if self.reader.peek() == '(' and arg_id == 0:
                arg_name = self.parse_unwrapped_arg()
                if arg_name is not None:
                    args[arg_id] = arg_name
                    arg_id += 1
            ## next arg
            if self.reader.peek() == ',':
                # self.reader.accept(',')
                # self.reader.skip_whitespace()
                if self.reader.peek(-1) != '[':
                    args[arg_id] = self.parse_unwrapped_arg()
                else:
                    assert arg_id > 0
                    args[arg_id] = self.parse_wrapped_arg()
                arg_id += 1
        func = ParsedFuncString(self.text, func_name, args)
        # end of node
        self.reader.accept(")")
        self.reader.accept(";")

        return func

    def parse_func_name(self):
        return self.reader.find_name()

    def parse_unwrapped_arg(self):
        'Argument without any wrapper'
        return self.reader.find_first_arg()

    def parse_wrapped_arg(self):
        'Argument wrapped in "[]"'
        return self.reader.find_wrapped_name()


class Reader:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.length = len(self.text)

    def peek(self, step = 0):
        self.skip_whitespace()
        if step == 0:
            try:
                return self.text[self.pos]
            except IndexError:
                print()
        elif step == -1: # peek the next non-space character
            k = 1
            while self.text[self.pos+k].isspace():
                k += 1
            return self.text[self.pos+k]

    def skip_whitespace(self):
        while self.pos < self.length:
            if self.text[self.pos].isspace():
                self.pos += 1
            else:
                break

    def accept(self, string):
        'check if the string passed in matches indexed raw text'
        self.skip_whitespace() # self.pos ++ if current pos points to a whitespace.
        end = self.pos + len(string)
        if end <= self.length and string == self.text[self.pos:end]:
            self.pos += len(string)
            return True
        else:
            raise FstringParseError("failed to accept '{}' in '{}^{}'".format(string, self.text[:self.pos], self.text[self.pos:]))

    def end_of_stream(self):
        self.skip_whitespace()

        if self.pos < self.length:
            raise FstringParseError("invalid end '^{}'".format(self.text[self.pos:]))

    def find_name(self):
        self.skip_whitespace()
        pos = self.pos
        while pos < self.length:
            if self.text[pos].isalnum() or self.text[pos] == '-' or self.text[pos] == '_':
                pos += 1
            else:
                break

        if pos > self.pos:
            name = self.text[self.pos:pos]
            self.pos = pos
            return name # identifier->name yiwei
        else:
            raise FstringParseError("failed to find its name in '^{}'".format(self.text[self.pos:]))

    def find_first_arg(self):
        ''
        name = ''
        state = 0
        pos = self.pos
        while pos < self.length:
            if state == 0:
                if self.text[pos] == '(' or self.text[pos] == ',':
                    state = 1
                    pos += 1
                    self.pos = pos
                else:
                    raise FstringParseError("Unwrapped argument should follow a '(' or ','")
            if state == 1:
                if self.text[pos] == ')': # if there is no argument
                    name = None
                    state = 'END'
                    continue
                name = self.find_name()
                self.skip_whitespace()
                pos = self.pos
                if self.text[pos] == ',' or self.text[pos] == ')':
                    state = 'END'
                else:
                    raise FstringParseError(f"unexpected letter '{self.text[pos]}' after the first argument")

            if state == 'END':
                break
        if state == 'END':
            self.pos = pos
            return name
        else:
            raise FstringParseError("Automata for the UNWRAPPED arg parsing goes wrong.")

    def find_wrapped_name(self):
        'find name wrapped by "[]"'
        self.skip_whitespace()
        names = []
        pos = self.pos
        state = 0
        while pos < self.length:
            if state == 0:              # start state
                if self.text[pos] == ',':
                    state = 1
                    pos += 1
                else:
                    raise FstringParseError(f"Parse failed, undetermined error")
            if state == 1:

                if self.text[pos].isspace():
                    state = 1
                    pos += 1
                elif self.text[pos] == '[':
                    state = 2
                    pos += 1
                    self.pos = pos
                else:
                    raise FstringParseError(f"Parse failed due to unexpected letter after ',' at {pos}")
            if state == 2:
                if self.text[pos].isspace():
                    state = 2
                    pos += 1
                elif self.text[pos].isalnum() or self.text[pos] == '-' or self.text[pos] == '_':
                    state = 2
                    pos += 1
                elif self.text[pos] == ']':
                    state = 'END'
                    names.append(self.text[self.pos:pos].strip())
                    pos += 1
                elif self.text[pos] == ',':
                    state = 2
                    names.append(self.text[self.pos:pos].strip())
                    pos += 1
                    self.pos = pos
                else:
                    raise FstringParseError(f"Parse failed due to unexpected letter '{self.text[pos]}' at {pos}")

            if state == 'END':  # end state
                break
        if state == 'END':
            self.pos = pos
            return names
        else:
            raise FstringParseError("Automata for the SECOND arg parsing goes wrong.")

