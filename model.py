import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, pad_id, embeddings=None, **kwargs):
        super(RNNEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, kwargs['vocab_embed_dim'])
        self.embeddings.padding_idx = pad_id
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(embeddings)
            self.embeddings.weight.requires_grad = False

        self.input_dropout = nn.Dropout(kwargs['input_dropout'])
        self.rnn = getattr(nn, kwargs['rnn_type'])(input_size=self.embeddings.embedding_dim,
                                                   hidden_size=kwargs['hidden_size'],
                                                   num_layers=kwargs['rnn_layer'],
                                                   dropout=kwargs['rnn_dropout'],
                                                   bidirectional=kwargs['bidirectional'],
                                                   batch_first=True)

    def forward(self, input_seqs, input_lengths=None):
        embedded = self.embeddings(input_seqs)  # batch x s_len x emb_dim
        embedded = self.input_dropout(embedded)

        if input_lengths is not None:
            embedded = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)

        if input_lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)  # unpack (back to padded)

        return output, hidden



class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class RNNDecoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, hidden_size, sos_id, eos_id, pad_id, embeddings,
                 bidirectional_encoder, use_attention, **kwargs):
        super(RNNDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, kwargs['vocab_embed_dim'])
        self.embeddings.padding_idx = pad_id
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(embeddings)
            self.embeddings.weight.requires_grad = False
        self.input_dropout = nn.Dropout(kwargs['input_dropout'])

        self.vocab_embed_dim = self.embeddings.embedding_dim
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, kwargs['rnn_type'])(input_size=self.embeddings.embedding_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layers=kwargs['rnn_layer'],
                                                   dropout=kwargs['rnn_dropout'],
                                                   batch_first=True)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.use_attention = use_attention
        self.bidirectional_encoder = bidirectional_encoder

        if self.use_attention:
            self.attention = Attention(self.hidden_size)

    def _init_state(self, encoder_hidden):
        """ Init decoder start with last state of the encoder """

        def _fix_enc_hidden(h):
            """ If encoder is bidirectional, do the following transformation.
            [layer*directions x batch x dim] -> [layer x batch x directions*dim]
            """
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):  # LSTM
            encoder_hidden = tuple([_fix_enc_hidden(h) for h in encoder_hidden])
        else:
            encoder_hidden = _fix_enc_hidden(encoder_hidden)
        return encoder_hidden

    def forward_step(self, input_seqs, hidden, encoder_outputs):
        embedded = self.embeddings(input_seqs)
        assert embedded.dim() == 3  # [batch x len x emb_dim]

        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        # TODO: check dropout layer for output
        return output, hidden, attn, embedded

    def forward(self, input_seqs, encoder_hidden, encoder_outputs,
                teacher_forcing_ratio=0,
                src_input=None):
        pass


def validate_args(input_seqs, encoder_hidden, encoder_output, use_attention, rnn_type, sos_id, teacher_forcing_ratio):
    if use_attention:
        if encoder_output is None:
            raise ValueError("Argument encoder_output cannot be None "
                             "when attention is used.")

    if input_seqs is None and encoder_hidden is None:
        batch_size = 1
    else:
        if input_seqs is not None:
            batch_size = input_seqs.size(0)  # [batch x max_len]
        else:
            if rnn_type == 'LSTM':
                batch_size = encoder_hidden[0].size(1)
            elif rnn_type == 'GRU':
                batch_size = encoder_hidden.size(1)
            else:
                raise ValueError("Unknown rnn mode is provided.")

    # set default input and max decoding length
    if input_seqs is None:
        if teacher_forcing_ratio > 0:
            raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        if rnn_type == 'LSTM':
            device = encoder_hidden[0].device
        elif rnn_type == 'GRU':
            device = encoder_hidden.device
        else:
            raise ValueError("Unknown rnn mode is provided.")

        input_seqs = torch.LongTensor([sos_id] * batch_size).view(batch_size, 1).to(device)

        if use_attention:
            max_length = int(encoder_output.size(1) * 1.5)
        else:
            max_length = 200
    else:
        max_length = input_seqs.size(1) - 1  # minus the start of sequence symbol

    return input_seqs, batch_size, max_length


class CopyDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_dim):
        super(CopyDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_dim = vocab_dim
        self.linear = nn.Linear(hidden_size, output_size)
        self.gen_linear = nn.Linear(hidden_size*2 + vocab_dim, 1)

    def forward(self, context, attn, src_input, decoder_input, decoder_hidden):
        batch_size, de_len = context.size(0), context.size(1)
        logits = self.linear(context.view(-1, self.hidden_size))
        vocab_prob = F.softmax(logits, dim=-1).view(batch_size, de_len, self.output_size)

        # assume that decoder is LSTM
        # decoder_hidden is only for last timestamp, but we need states all timestamps
        # hidden = torch.cat(decoder_hidden).transpose(0, 1)
        # TODO: OpenNMT-py 처럼 decoder를 LSTMCell로 작성해야할 듯

        p_gen_input = torch.cat((context.view(-1, self.hidden_size), 
                                 decoder_hidden[0][-1],
                                 decoder_input.view(-1, self.vocab_dim)), dim=1)
        gen_prob = torch.sigmoid(self.gen_linear(p_gen_input))
        vocab_prob = vocab_prob.view(-1, self.output_size) * gen_prob
        copy_prob = attn.view(-1, attn.size(2)) * (1 - gen_prob)

        vocab_prob = vocab_prob.view(batch_size, de_len, self.output_size)
        copy_prob = copy_prob.view(batch_size, de_len, -1)
        for i in range(copy_prob.size(1)):
            vocab_prob[:, i, :].scatter_add_(1, src_input, copy_prob[:, i, :])

        vocab_prob.log_()
        symbols = vocab_prob.topk(1, dim=2)[1]
        return vocab_prob, symbols


class RNNDecoderPointer(RNNDecoder):
    def __init__(self, vocab_size, hidden_size, sos_id, eos_id, pad_id,
                 embeddings=None,
                 bidirectional_encoder=True,
                 use_attention=True,
                 **kwargs):
        super().__init__(vocab_size, hidden_size, sos_id, eos_id, pad_id,
                         embeddings=embeddings,
                         bidirectional_encoder=bidirectional_encoder,
                         use_attention=use_attention,
                         **kwargs)

        self.output_size = vocab_size
        self.decoder = CopyDecoder(self.hidden_size, self.output_size, self.vocab_embed_dim)
        self.concat = nn.Linear(self.hidden_size * 2 + self.vocab_embed_dim, self.hidden_size)

    def forward(self, input_seqs, encoder_hidden, encoder_outputs,
                teacher_forcing_ratio=0, src_input=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[RNNDecoder.KEY_ATTN_SCORE] = list()

        # valid arguments
        input_seqs, batch_size, max_length = validate_args(input_seqs, encoder_hidden, encoder_outputs,
                                                           use_attention=self.attention,
                                                           rnn_type=self.rnn.mode,
                                                           sos_id=self.sos_id,
                                                           teacher_forcing_ratio=teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def post_decode(step_output, step_symbols, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[RNNDecoder.KEY_ATTN_SCORE].append(step_attn)
            sequence_symbols.append(step_symbols)

            eos_batches = step_symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

        # Manual unrolling is used to support teacher forcing
        if use_teacher_forcing:
            decoder_input = input_seqs[:, 0].unsqueeze(1)
            concated_contexts = []
            for di in range(max_length):
                context, decoder_hidden, attn, decoder_input_embed = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs)
                decoder_output, symbols = self.decoder(context, attn, src_input, decoder_input_embed, decoder_hidden)

                step_output = decoder_output.squeeze(1)
                step_symbols = symbols.squeeze(1)
                post_decode(step_output, step_symbols, attn)
                decoder_input = input_seqs[:, di+1].unsqueeze(1)

                concated_vec = self.concat(torch.cat([context, decoder_hidden[0][-1].unsqueeze(1),
                                                      self.embeddings(symbols.squeeze(2))], dim=2))
                concated_contexts.append(concated_vec)

            concated_contexts = torch.cat(concated_contexts, dim=1)
        else:
            decoder_input = input_seqs[:, 0].unsqueeze(1)

            concated_contexts = []
            for di in range(max_length):
                context, decoder_hidden, attn, decoder_input_embed = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs)
                decoder_output, symbols = self.decoder(context, attn, src_input, decoder_input_embed, decoder_hidden)

                step_output = decoder_output.squeeze(1)
                step_symbols = symbols.squeeze(1)
                post_decode(step_output, step_symbols, attn)
                decoder_input = step_symbols

                concated_vec = self.concat(torch.cat([context, decoder_hidden[0][-1].unsqueeze(1),
                                                      self.embeddings(symbols.squeeze(2))], dim=2))
                concated_contexts.append(concated_vec)

            concated_contexts = torch.cat(concated_contexts, dim=1)

        ret_dict[RNNDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[RNNDecoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, concated_contexts, ret_dict


class CRFTagger(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via backpropagation.
    """

    def __init__(self, hidden_size, num_tags):
        super(CRFTagger, self).__init__()

        self.num_tags = num_tags
        self.projection = nn.Linear(hidden_size, num_tags)
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def infer(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        feats = self.projection(feats)
        return self._viterbi(feats)

    def forward(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar]
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        feats = self.projection(feats)
        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match')

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1)  # [batch_size, num_tags]

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)  # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i]  # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1)  # [batch_size, num_tags], [batch_size, num_tags]

            paths.append(idx)
            v = (v + feat)  # [batch_size, num_tags]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()


class KMAModel(nn.Module):
    def __init__(self, encoder, decoder, tagger=None):
        super(KMAModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tagger = tagger

    def forward(self, input_seqs, target_lex_seqs, target_pos_seqs, input_lengths=None, teaching_force_ratio=0):
        # input_seqs: [batch x max_length]
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        decoder_outputs, last_hidden, others = self.decoder(target_lex_seqs, encoder_hidden, encoder_outputs,
                                                            teacher_forcing_ratio=teaching_force_ratio,
                                                            src_input=input_seqs)
        if self.tagger:
            tagger_loss = self.tagger(last_hidden, target_pos_seqs[:, 1:])
            return decoder_outputs, tagger_loss, others
        else:
            return decoder_outputs, last_hidden, others

    def infer(self, input_seqs, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        decoder_outputs, last_hidden, others = self.decoder(None, encoder_hidden, encoder_outputs,
                                                            teacher_forcing_ratio=0,
                                                            src_input=input_seqs)
        if self.tagger:
            tagger_outputs = self.tagger.infer(last_hidden)
            return decoder_outputs, tagger_outputs, others
        else:
            return decoder_outputs, last_hidden, others
