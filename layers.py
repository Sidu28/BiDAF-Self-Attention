"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        # here linear_input = 300
        linear_input = word_vectors.size(1)
        self.char_embed = None


        # char_vectors shape (vocab_size, char_embedding) 1376, 64

        cnn_input_size = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)

        cnn_output_size = 16
        self.cnn = CNN(cnn_input_size, cnn_output_size)
        self.maxpool = nn.MaxPool1d(cnn_output_size)
        # here linear_input = 308
        linear_input += cnn_output_size

        self.proj = nn.Linear(linear_input, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x, char_ids):
        """
        x shape: 64, 279
        char_ids shape: 64, 279, 16
        """
        word_emb = self.word_embed(x)  # (batch_size, seq_len, embed_size)
        char_emb = self.char_embed(char_ids)                                                                    #torch.size[64, 308, 16, 64] --> (batch_size, sequence_length, max_word_length, pretrained_char_vec_size)

        char_emb = char_emb.view(char_emb.shape[0] * char_emb.shape[1], char_emb.shape[2], char_emb.shape[3])   #torch.Size([19712, 16, 64])
        char_emb = char_emb.transpose(1, 2)                                                                     #torch.Size([19712, 64, 16])
        char_emb = self.cnn(char_emb)                                                                           #torch.Size([19712, 16, 12])

        char_emb = self.maxpool(F.relu(char_emb))                                                               #torch.size([19712, 16, 1])
        char_emb = char_emb.view(word_emb.size(0), word_emb.size(1), char_emb.size(1))                          #torch.size([64, 308, 16])


        concat_emb = torch.cat((word_emb, char_emb), dim=-1)
        concat_emb = F.dropout(concat_emb, self.drop_prob, self.training)
        concat_emb = self.proj(concat_emb)  # (batch_size, seq_len, hidden_size)
        concat_emb = self.hwy(concat_emb)  # (batch_size, seq_len, hidden_size)

        return concat_emb


class CNN(nn.Module):
    """Character CNN
    """

    def __init__(self, input, output, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input, out_channels=output, kernel_size=kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv1d(input)



class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class SelfAttentionSimple(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, hidden_size))                        # (1,8d)    8d = hidden_size
        self.weight_matrix_j = nn.Parameter(torch.zeros(hidden_size, hidden_size))     # (8d,8d)
        self.weight_matrix_t = nn.Parameter(torch.zeros(hidden_size, hidden_size))     # (8d,8d)

        for weight in (self.weight, self.weight_matrix_t, self.weight_matrix_j):
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        B,S,D = x.size()
        
        weight = self.weight
        weight_matrix_1 = self.weight_matrix_j
        weight_matrix_2 = self.weight_matrix_t

        matrix_prod_1 = x@weight_matrix_1 
        matrix_prod_2 = x@weight_matrix_2 
        matrix_prod_1 = matrix_prod_1.unsqueeze(1)
        matrix_prod_2 = matrix_prod_2.unsqueeze(2)
        tanh = torch.tanh(matrix_prod_1 + matrix_prod_2)        # (64, c_len, c_len, hidden_size, 1)
        
        similarity_matrix = tanh@weight.squeeze(0)          # (64, c_len, c_len)
        att = F.softmax(similarity_matrix, -1)                  # (64, c_len, c_len)
        c = att @ x
        return c






class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #take transpose of input so the input is 8d x c_len.  Then we can take each column of size 8d x 1

        self.weight = nn.Parameter(torch.zeros(1, hidden_size))                        #(1,8d)    8d = hidden_size
        self.weight_matrix_j = nn.Parameter(torch.zeros(hidden_size, hidden_size))     #(8d,8d)
        self.weight_matrix_t = nn.Parameter(torch.zeros(hidden_size, hidden_size))     #(8d,8d)

        for weight in (self.weight, self.weight_matrix_t, self.weight_matrix_j):
            nn.init.xavier_uniform_(weight)


    def forward(self, input):

        input_t = input.permute(0, 2, 1)
        batch_size, hidden_size, c_len = input_t.size()

        weight = self.weight
        weight_matrix_1 = self.weight_matrix_j
        weight_matrix_2 = self.weight_matrix_t

        weight = weight.unsqueeze(0).expand(batch_size, 1, hidden_size)
        weight_matrix_1 = weight_matrix_1.unsqueeze(0).expand(batch_size, hidden_size, hidden_size)
        weight_matrix_2 = weight_matrix_2.unsqueeze(0).expand(batch_size, hidden_size, hidden_size)



        matrix_prod_1_3d, matrix_prod_2_3d = self.get_weight_matrix_products(input_t, weight_matrix_1, weight_matrix_2)

        tanh = torch.tanh(matrix_prod_1_3d + matrix_prod_2_3d)                          # (64, c_len, c_len, hidden_size, 1)

        weight = weight.unsqueeze(1)                                                    # (64, 1, 1, hidden_size)
        weight = weight.unsqueeze(1)                                                    # (64, 1, 1, 1, hidden_size)
        weight = weight.expand(batch_size, c_len, c_len, 1, hidden_size)                # (64, c_len, c_len, 1, hidden_size)

        s = torch.matmul(weight, tanh)                                  # (64, c_len, c_len, 1, 1)
        s = s.squeeze(4)                                                # (64, c_len, c_len, 1)
        similarity_matrix = s.squeeze(3)                                # (64, c_len, c_len)

        att = F.softmax(similarity_matrix, -1)                          # (64, c_len, c_len)

        c = self.get_att_pool_matrix(att, input)                        # (64, c_len, c_len)

    def get_weight_matrix_products(self, input_t, weight_matrix_1, weight_matrix_2):

        batch_size, hidden_size, c_len = input_t.size()

        matrix_prod_1 = torch.matmul(weight_matrix_1, input_t)              # W_1 x example - (64, hidden_size, c_len)
        matrix_prod_1 = torch.transpose(matrix_prod_1, 1, 2)                # (64, c_len, hidden_size)
        matrix_prod_1 = matrix_prod_1.unsqueeze(1)                          # (64, 1, c_len, hidden_size)
        matrix_prod_1 = matrix_prod_1.unsqueeze(4)                          # (64, 1, c_len, hidden_size, 1)
        matrix_prod_1_3d = matrix_prod_1.expand(batch_size, c_len, c_len, hidden_size,
                                                1)                          # (64, c_len, c_len, hidden_size, 1)

        matrix_prod_2 = torch.matmul(weight_matrix_2, input_t)  # W_1 x example - (64, hidden_size, c_len)
        matrix_prod_2 = torch.transpose(matrix_prod_2, 1, 2)  # (64, c_len, hidden_size)
        matrix_prod_2 = matrix_prod_2.unsqueeze(1)  # (64, 1, c_len, hidden_size)
        matrix_prod_2 = matrix_prod_2.unsqueeze(4)  # (64, 1, c_len, hidden_size, 1)
        matrix_prod_2_3d = matrix_prod_2.expand(batch_size, c_len, c_len, hidden_size,
                                                1)  # (64, c_len, c_len, hidden_size, 1)

        return matrix_prod_1_3d, matrix_prod_2_3d

    def get_att_pool_matrix(self, att, input):
        batch_size, c_len, hidden_size = input.size()

        att = att.unsqueeze(3)               # (64, c_len, c_len, 1)
        att = att.unsqueeze(4)               # (64, c_len, c_len, 1, 1)

        input = input.reshape(batch_size, c_len, 1, hidden_size)        # (64, c_len, 1, hidden_size)
        input = input.unsqueeze(0)                                      # (64, 1, c_len, 1, hidden_size)
        input = input.expand(batch_size, c_len, c_len, 1, hidden_size)  # (64, c_len, c_len, 1, hidden_size)

        #CHECK THIS
        c_temp = att * input                                            # (64, c_len, c_len, 1, hidden_size)

        c = torch.sum(c_temp, 2)                                        # (64, c_len, 1, hidden_size)

        c = c.unsqueeze(2)                                              # (64, c_len, hidden_size)

        return c


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
