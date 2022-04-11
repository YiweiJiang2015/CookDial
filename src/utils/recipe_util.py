import json, re
from json import JSONDecodeError
from pathlib import Path

r"""
*===================================================================================
*      Base classes for recipe instruction, ingredient, utterance and session.
*                                Writer: Yiwei
*===================================================================================
"""


class RecipeBase:
    def __init__(self, item):
        self.text: str = item['text']
        self.id: str = item['id']
        self.eamr: str = item['eamr']
        self.type: str = item['type']

    def update_eamr(self, eamr_string):
        self.eamr = eamr_string

    def __repr__(self):
        return f"{self.id}: {self.text[:70]}" if self.id is not None else self.text[:70]

    def to_dict(self):
        raise NotImplementedError("User shall override this method.")


class UtteranceBase:
    """Utterance object: the smallest cell"""
    def __init__(self, utterance):
        self.utt_id: str = utterance['utt_id']
        self.bot: bool = utterance['bot']
        self.utterance: str = utterance['utterance']
        self.annotations = utterance['annotations']
        try:
            self.annotations_dict = json.loads(self.annotations)  # for user utterance
        except JSONDecodeError:
            self.annotations_dict = {}  # for bot utterance
        except TypeError:
            self.annotations_dict = {}
            print(f'{self} has big problems with json format.')

    def __repr__(self):
        return f"Utterance({self.utt_id}, text='{self.utterance}', bot={self.bot})"

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, serialized_content):
        # Load processed data from json files
        raise NotImplementedError


class SessionBase:
    def __init__(self, utt_cls, file=None, from_serialized=False, serialized_content=None):
        self.utterances = {}
        self.utt_cls = utt_cls
        if from_serialized:
            self.dialog_json_path = None
            self.dialog_id = serialized_content['dialog_id']
            self.load_serialized_content(serialized_content)
        else:
            self.dialog_json_path = Path(file)
            self.dialog_id = self.dialog_json_path.name.split('.')[0]
            self.load_json_file()

    def load_json_file(self):
        with open(self.dialog_json_path, 'r') as fp:
            data = json.load(fp)
            utterances = data['messages']
            self.utterances = {str(utt_id): self.utt_cls(utterance)
                               for utt_id, utterance in enumerate(utterances)}

    def load_serialized_content(self, serialized_content):
        self.utterances = {str(utt_id): self.utt_cls.from_json(utterance)
                           for utt_id, utterance in enumerate(serialized_content['utterances'])}

    def to_dict(self):
        'Serialize self.utterances dict into a jsonic object'
        ser_dict = {}
        ser_dict["messages"] = [utterance.to_dict() for utterance in self.utterances.values()]
        return ser_dict

    @classmethod
    def from_json(cls, serialized_content):
        # Load processed data from json files
        raise NotImplementedError

    def __len__(self):
        return len(self.utterances)

    def __repr__(self):
        return f"Dialog {self.dialog_id}: {len(self)} utterances"

    def dump_file(self):
        raise NotImplementedError


r"""
*============================================================
*      Parser for the toy language used in RISeC 2.0
*             Writer: Johannes, Yiwei
*============================================================
"""


def myindent(indent):
    return "".join(["\t"] * indent)

node_counter = 0

def new_node():
    global node_counter
    node_counter += 1
    return node_counter


class Node:
    'An n-ary tree. Degree and depth depend on input.'

    def __init__(self, node_name, node_span, node_label):
        self.name = node_name
        self.span = node_span
        self.label = node_label
        self.edges = []
        self._id = new_node()
        self.index = None  # a unique index mapped from the projected binary tree.
        if self.span is not None:
            self.span.parent = self

    def to_bin_tree_node(self):
        'Copy the data fields to BinTreeNode'
        return BinTreeNode(self.name, self.span, self.label, map=self)

    def get_id(self):
        return 'unamed-{}'.format(self._id) if self.name is None else self.name

    def add_edge(self, edge):
        self.edges.append(edge)

    def to_eamr(self, indent=0, multiline=True):
        'to string with newline marker'
        output = "("
        if self.name != None:
            output += " " + self.name
        if self.span != None:
            output += " {}".format(self.span)
        if self.label != None:
            output += " / {}".format(self.label)

        if len(self.edges) > 0:
            if multiline:
                output += "\n"
            for edge in self.edges:
                if multiline:
                    output += myindent(indent + 1) + edge.to_eamr(indent + 1, multiline) + "\n"
                else:
                    output += " " + edge.to_eamr(indent + 1, multiline)
            if multiline:
                output += myindent(indent) + ")"
            else:
                output += " )"
        else:
            output += " )"

        return output

    def get_title(self):
        title = self.get_id()
        if self.span != None:
            title += " {}".format(self.span)
        if self.label != None:
            title += " / {}".format(self.label)
        return title

    def update_index(self, new_index):
        self.index = new_index

    def update_span_offset(self, offset):
        if self.span is not None:
            self.span.update_span_offset(offset)
        for edge in self.edges:
            edge.end_point.update_span_offset(offset)

    def copy(self):
        src = Node(self.name, self.span.copy() if self.span is not None else None, self.label)
        for edge in self.edges:
            dst = edge.end_point.copy()
            src.add_edge(Edge(edge.label, src, dst))
        return src

    def is_match(self, text):
        if self.span is not None and not self.span.is_match(text):
            return False
        for edge in self.edges:
            if not edge.end_point.is_match(text):
                return False
        return True

    def is_overlap(self, paragraph_begin, paragraph_end):
        if self.span != None and self.span.is_overlap(paragraph_begin, paragraph_end):
            return True

        for edge in self.edges:
            if edge.end_point.is_overlap(paragraph_begin, paragraph_end):
                return True

        return False

    def __repr__(self):
        return self.to_eamr(multiline=False)

    def get_nodes(self, filter):
        """
        :param filter: a lambda expression, e.g. lambda x: x.label=="FOOD"
        :return:
        """
        nodes = []
        if filter(self):
            nodes.append(self)
        for edge in self.edges:
            nodes.extend(edge.end_point.get_nodes(filter))
        return nodes

    def get_spans(self, filter):
        """

        :param filter: a lambda expression, e.g. lambda x: x.label=="FOOD"
        :return:
        """
        spans = []
        if self.span is not None and filter(self):
            spans.append(self.span)
        for edge in self.edges:
            spans.extend(edge.end_point.get_spans(filter))
        return spans

    def get_edges(self, filter):
        result = []
        for edge in self.edges:
            if filter(edge):
                result.append(edge)
            result.extend(edge.end_point.get_edges(filter))
        return result

    def has_edge(self, label):
        for edge in self.edges:
            if edge.label == label:
                return True
        return False

    # This method expects a Visitor object as argument
    def walk(self, visitor):
        visitor.node(self)

        if self.span is not None:
            visitor.span(self.span)

        for edge in self.edges:
            edge.walk(visitor)

    def number(self):
        'This method only works with a complete graph rooted in a inst-x node.'
        LABELS = {'AC': 'ac', 'TOOL': 'tool', 'DUR': "dur", 'TEMPERATURE': 'temp', 'CONDITION_CLAUSE': 'cond',
                  'PURPOSE_CLAUSE': 'purp'}
        assert self.label == 'R'
        inst_id = self.name.split('-')[1]
        for key, label in LABELS.items():
            nodes = self.get_nodes(filter=lambda x: x.label == key)
            if len(nodes) > 0:
                for i, node in enumerate(nodes):
                    if node.name is None:
                        node.name = (label + '-' + inst_id + '-' + str(i))

    # get first character offset
    def get_begin(self):
        begin = self.span.begin
        for edge in self.edges:
            tmp = edge.end_point.get_begin()
            begin = min(begin, tmp)
        return begin

    # get last character offset
    def get_end(self):
        end = self.span.end
        for edge in self.edges:
            tmp = edge.end_point.get_end()
            end = max(end, tmp)
        return end


class Edge:

    def __init__(self, edge_label, edge_start_point, edge_end_point):
        self.label = edge_label
        self.start_point = edge_start_point
        self.end_point = edge_end_point

    def to_eamr(self, indent=0, multiline=True):
        return ":" + self.label + " " + self.end_point.to_eamr(indent, multiline)

    def __repr__(self):
        return self.to_eamr(multiline=False)

    # This method expects a Visitor object as argument
    def walk(self, visitor):
        visitor.edge(self)
        self.end_point.walk(visitor)


class Span:

    def __init__(self, mention, begin=-1, end=-1, parent=None):
        self.mention = mention
        self.begin = begin
        self.end = end
        self.parent = parent

    def copy(self):
        return Span(self.mention, self.begin, self.end, self.parent)

    def update_span_offset(self, offset):
        self.begin += offset
        self.end += offset

    def is_match(self, text):
        return text[self.begin:self.end] == self.mention

    def is_overlap(self, paragraph_begin, paragraph_end):
        """Verify if span index overflows the paragraph boundary."""
        if self.begin >= paragraph_end:
            return False
        elif self.end <= paragraph_begin:
            return False
        else:
            return True

    def __str__(self):
        return "\"{}\"@{}:{}".format(self.mention, self.begin, self.end)

    def __repr__(self):
        return str(self)


class BinTreeNode:
    'Binary tree node. It only has left and right children.'
    def __init__(self, node_name=None, node_span=None, node_label=None, map=None):
        self.name = node_name
        self.span = node_span
        self.label = node_label
        self.map = map # a reference to its original node in the n-ary tree.
        self.index = None
        self.lchild = None
        self.rchild = None

    def pop_data(self):
        return {'name': self.name, 'span': self.span, 'label': self.label}

    def is_terminal(self):
        if self.lchild is None and self.rchild is None:
            return True
        else:
            return False

    def update_index(self, new_index):
        self.index = new_index


class EamrBinaryTree:
    "The original tree is generated by Parser in Johannes' codes"
    # The difference between this class and Johannes' Node class is the edge label. After being converted to a binary tree, the original tree loses its
    # edge information unless we create another binary tree to store the relation labels correspondingly.
    def __init__(self, nary_tree: Node):
        # self.nary_tree = nary_tree # deprecated to save memory
        self.bin_tree: BinTreeNode = self.to_binary_tree(nary_tree)
        self.walk_trace = []
        self.traverse(self.bin_tree)

    def traverse(self, bin_tree_node: BinTreeNode):
        'Pre-order traverse. Give each node and its origin an unique index.'
        if bin_tree_node is None:
            return
        # Assign index
        bin_tree_node.update_index(len(self.walk_trace)) # todo shall we record the predecessor, successor?
        bin_tree_node.map.update_index(len(self.walk_trace))
        ## walk
        self.walk_trace.append(bin_tree_node.pop_data()) # todo using a global variable is a hidden risk
        self.traverse(bin_tree_node.lchild)
        self.traverse(bin_tree_node.rchild)


    def to_binary_tree(self, nary_tree: Node) -> BinTreeNode:
        'Construct a binary tree.'
        new_node = nary_tree.to_bin_tree_node()
        num_children = len(nary_tree.edges)
        if num_children == 0:
            pass
        elif num_children == 1:
            new_node.lchild = self.to_binary_tree(nary_tree.edges[0].end_point)
        else:
            new_node.lchild = self.to_binary_tree(nary_tree.edges[0].end_point)
            for j in range( num_children-1, -1, -1):
                if j == num_children-1:
                    right_bottom = self.to_binary_tree(nary_tree.edges[j].end_point)
                elif j > 0:
                    right_bottom_up = self.to_binary_tree(nary_tree.edges[j].end_point)
                    right_bottom_up.rchild = right_bottom
                    right_bottom = right_bottom_up
                else:
                    new_node.lchild.rchild = right_bottom

        return new_node


class EAMRParser:
    """
    A bottom-up parser.
    """

    def __init__(self, text):
        self.buffer = MyBuffer(text)

    def parse(self):
        node = self.parse_node()
        self.buffer.end_of_stream()
        return node

    def parse_node(self):
        node_name = None
        node_span = None
        node_label = None
        # start of node
        self.buffer.accept("(")

        ch = self.buffer.peek()

        # node name
        if ch != '"' and ch != ')':
            node_name = self.parse_node_name()

        # node span
        if self.buffer.peek() == '"':
            node_span = self.parse_span()

        # node label
        if self.buffer.peek() == '/':
            self.buffer.accept('/')
            node_label = self.buffer.find_name()

        node = Node(node_name, node_span, node_label)

        # accept edges
        while self.buffer.peek() == ':':
            node.add_edge(self.parse_edge(node))

        # end of node
        self.buffer.accept(")")

        return node

    def parse_node_name(self):
        return self.buffer.find_name()

    def parse_span(self):
        mention = self.buffer.accept_string()
        begin = -1
        end = -1
        if self.buffer.peek() == '@':
            self.buffer.accept('@')
            begin = int(self.buffer.find_name())
            self.buffer.accept(':')
            end = int(self.buffer.find_name())
        return Span(mention, begin, end)

    def parse_edge(self, edge_start_point):
        edge_label = None

        self.buffer.accept(":")
        edge_label = self.buffer.find_name()

        if self.buffer.peek() == '"':  # anonymous node
            target_span = self.parse_span()
            edge_end_point = Node(None, target_span, None)
        elif self.buffer.peek() == '(':
            edge_end_point = self.parse_node()
        else:
            target_id = self.buffer.find_name()
            edge_end_point = Node(target_id, None, None)

        return Edge(edge_label, edge_start_point, edge_end_point)


class EAMRParseError(Exception):
    pass


class MyBuffer:

    def __init__(self, text):
        self.text = self.preprocess(text)
        self.pos = 0
        self.length = len(self.text)

    def skip_comment(self, text):
        if not isinstance(text, str):
            raise TypeError(f"{text}")
        pattern = r"#.*?#"
        return re.sub(pattern, '', text)

    def preprocess(self, text):
        text = self.skip_comment(text)
        return text

    def peek(self):
        'Read ahead of the character of current position from self.text'
        self.skip_whitespace()
        return self.text[self.pos]

    def skip_whitespace(self):
        while self.pos < self.length:
            if self.text[self.pos].isspace():
                self.pos += 1
            else:
                break

    def accept(self, string):
        'yiwei: check if the string passed in matches indexed raw text'
        self.skip_whitespace()  # self.pos ++ if current pos points to a whitespace.
        end = self.pos + len(string)
        if end <= self.length and string == self.text[self.pos:end]:
            self.pos += len(string)
            return
        else:
            raise EAMRParseError(
                "failed to accept '{}' in '{}^{}'".format(string, self.text[:self.pos], self.text[self.pos:]))

    def accept_string(self):
        self.skip_whitespace()

        output = []
        pos = self.pos
        state = 0
        while pos < self.length:
            if state == 0:  # start state
                if self.text[pos] == '"':
                    state = 1
                    pos += 1
                else:
                    raise EAMRParseError("failed to accept string, '{}' should start with \"".format(self.text[self.pos:]))
            elif state == 1:  # inside string
                if self.text[pos] == '"':
                    state = 3
                    pos += 1
                elif self.text[pos] == '\\':
                    state = 2
                    pos += 1
                else:
                    output.append(self.text[pos])
                    pos += 1
            elif state == 2:  # escape marker
                output.append(self.text[pos])
                pos += 1
                state = 1
            elif state == 3:  # end state
                break

        if state == 3:
            self.pos = pos
            return "".join(output)
        else:
            raise EAMRParseError("failed to accept string '{}'".format(self.text[self.pos:]))

    def find_name(self):
        self.skip_whitespace()

        pos = self.pos
        while pos < self.length:
            if self.text[pos].isalnum() or self.text[pos] == '-' or self.text[pos] == '_' or self.text[pos] == '?':
                pos += 1
            else:
                break

        if pos > self.pos:
            name = self.text[self.pos:pos]
            self.pos = pos
            return name  # identifier->name yiwei
        else:
            raise EAMRParseError("failed to find its name in '^{}'".format(self.text[self.pos:]))

    def end_of_stream(self):
        self.skip_whitespace()

        if self.pos < self.length:
            raise EAMRParseError("invalid end '^{}'".format(self.text[self.pos:]))