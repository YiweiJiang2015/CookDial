import torch
import torch.nn as nn

from utils import batched_index_select, batched_span_select, weighted_sum, masked_softmax

class SpanEmbedding(nn.Module):
    """
    Span representation for sequence model output (RNN, Transformers)
    Optionally:
        use_self_attention: head span vector
    """
    def __init__(self, hidden_dim,
                 use_self_attention: bool = False,
                 use_dummy_span: bool = False,
                 method: str = 'end_point',
                 ):
        super(SpanEmbedding, self).__init__()
        if use_self_attention:
            self.attention = SpanSelfAttention(hidden_dim)
        self.use_self_attention = use_self_attention

        self.use_dummy_span = use_dummy_span
        self.method = method

        if method == 'end_point':
            self._dim_out = 3 * hidden_dim if self.use_self_attention else 2 * hidden_dim
        if method == 'ave':
            self._dim_out = hidden_dim
        if method == 'max':
            self._dim_out = hidden_dim
        if method == 'ave_max':
            self._dim_out = 2 * hidden_dim
        if method == 'self':
            self._self_att_mod = SelfAttentiveSpanExtractor(hidden_dim)
            self._dim_out = hidden_dim
        if method == 'ave_self':
            self._self_att_mod = SelfAttentiveSpanExtractor(hidden_dim)
            self._dim_out = 2 * hidden_dim
        if method == 'end_self':
            self._self_att_mod = SelfAttentiveSpanExtractor(hidden_dim)
            self._dim_out = 3 * hidden_dim


    def forward(self, hiddens, span_indices):
        if self.method == 'end_point':
            span_vecs = self._end_point_embed(hiddens, span_indices)
        elif self.method == 'ave':
            span_vecs = self._average_embed(hiddens, span_indices)
        elif self.method == 'max':
            span_vecs = self._max_embed(hiddens, span_indices)
        elif self.method == 'ave_max':
            span_vecs = torch.cat([self._average_embed(hiddens, span_indices), self._max_embed(hiddens, span_indices)], dim=-1)
        elif self.method == 'self':
            span_vecs = self._self_att_mod(hiddens, span_indices)
        elif self.method == 'ave_self':
            span_vecs = torch.cat([self._average_embed(hiddens, span_indices), self._self_att_mod(hiddens, span_indices)], dim=-1)
        elif self.method == 'end_self':
            span_vecs = torch.cat([self._end_point_embed(hiddens, span_indices), self._self_att_mod(hiddens, span_indices)], dim=-1)
        # if self.use_completed_step_feature:
        #     span_vecs = self.add_marker_feature(span_vecs)
        return span_vecs

    def get_output_size(self):
        return self._dim_out

    def _end_point_embed(self, hiddens, span_indices):
        span_vecs = batched_index_select(hiddens.contiguous(), span_indices)
        # end point representation
        start_vecs, end_vecs = torch.chunk(span_vecs, chunks=2, dim=2)
        start_vecs = start_vecs.squeeze(2)
        end_vecs = end_vecs.squeeze(2)
        if self.use_dummy_span:
            null_start = torch.zeros_like(hiddens[:, 0, :]).unsqueeze(1)
            null_end = torch.zeros_like(hiddens[:, 0, :]).unsqueeze(1)

            start_vecs = torch.cat([null_start, start_vecs], dim=1)
            end_vecs = torch.cat([null_end, end_vecs], dim=1)
        span_vecs = torch.cat([start_vecs, end_vecs], dim=-1)#.squeeze(0)
        if self.use_self_attention:
            head_spans = self.attention(hiddens, span_indices)
            span_vecs = torch.cat([start_vecs, end_vecs, head_spans], dim=-1).squeeze(0)

        return span_vecs

    def _average_embed(self, hiddens, span_indices):
        subseq_vecs, subseq_mask = batched_span_select(hiddens.contiguous(), span_indices)
        if self.use_self_attention:
            return self.attention(subseq_vecs, subseq_mask)
        subseq_vecs = subseq_vecs.masked_fill(~subseq_mask.unsqueeze(-1), value=0.0)
        span_vecs = subseq_vecs.mean(dim=2)
        # We have to take into account that the masked dimensions do not contribute to the average pooling.
        # Therefore, we need to calculate a weight matrix based on the subseq_mask to mitigate the mean operation.
        subseq_weight = torch.ones_like(span_vecs).float() * subseq_mask.size(-1)
        subseq_weight_prime = subseq_weight / subseq_mask.sum(dim=-1).float().unsqueeze(-1)
        span_vecs_prime = span_vecs * subseq_weight_prime
        # if self.use_self_attention:
        #     span_vecs *= (subseq_weight * attention_weights)
        # else:
        #     span_vecs *= subseq_weight
        return span_vecs_prime

    def _max_embed(self, hiddens, span_indices):
        subseq_vecs, subseq_mask = batched_span_select(hiddens, span_indices)
        subseq_vecs = subseq_vecs.masked_fill(~subseq_mask.unsqueeze(-1), value=-1e7)
        span_vecs = subseq_vecs.max(dim=2)[0]
        return span_vecs

    @classmethod
    def from_config(cls, config: dict):
        hidden_dim = config['hidden_dim']
        use_self_attention = config.get('use_self_attention', False)
        use_dummy_span = config.get('use_dummy_span', False)
        method = config.get('method', 'end_point')
        temporal_feature = config.get('temporal_feature', 0)
        return cls(hidden_dim=hidden_dim, use_self_attention=use_self_attention,
                   use_dummy_span=use_dummy_span, method=method)

class SpanSelfAttention(nn.Module):
    """
    # Deprecated: Span self attention adapted from paper "End-to-end Neural Coreference Resolution".
    Now we compute the token weights within each span.
    Use a FFNN to compute the attention weights.
    """
    def __init__(self, hidden_dim):
        super(SpanSelfAttention, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.ReLU(),
            # nn.Linear(50, 1)
        )

    def forward(self, subseq_vecs, subseq_mask):
        weights = self.ffnn(subseq_vecs).squeeze(-1)
        weights = weights.masked_fill(~subseq_mask, value=-1e7)
        weights = torch.softmax(weights, dim=-1)
        subseq_vecs = subseq_vecs.masked_fill(~subseq_mask.unsqueeze(-1), value=0.0)
        subseq_vecs_prime = subseq_vecs * weights.unsqueeze(-1)
        return subseq_vecs_prime.sum(dim=2)

    # def forward(self, inputs, span_indices):
    #     start_indice, end_indice = torch.chunk(span_indices, chunks=2, dim=1)
    #     start_indice = start_indice.squeeze(0).squeeze(0)
    #     end_indice = end_indice.squeeze(0).squeeze(0)
    #     head_vectors = [torch.zeros((1, 1, self.dim_output), device=inputs.device)]
    #     for j in range(span_indices.size(2)):
    #         width_indices = torch.arange(start_indice[j], end_indice[j]+1)
    #         width_indices = width_indices.long().unsqueeze(0).to(inputs.device)
    #         vectors = batched_index_select(inputs, width_indices)
    #         scores = self.ffnn(vectors).squeeze(-1)
    #         attention_weights = torch.softmax(scores, -1)
    #         head_vector = torch.matmul(attention_weights, vectors)
    #         head_vectors.append(head_vector)
    #     head_vectors = torch.stack(head_vectors, dim=2) #.squeeze(0)
    #     return head_vectors


class SelfAttentiveSpanExtractor(nn.Module):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Registered as a `SpanExtractor` with name "self_attentive".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.

    # Returns

    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = torch.nn.Linear(input_dim, 1)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)

        concat_output, span_mask = batched_span_select(concat_tensor, span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits, span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1)

        return attended_text_embeddings
