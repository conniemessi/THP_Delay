import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    # print("len_s", len_s)
    # print(seq)
    # print(delta_matrix.item())
    # delay = int(delay_score.item())
    # print(delay)
    # subsequent_mask = new_mask
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)  # diagonal=1 + delay
    # print(delay_score)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model
        #
        # time delay
        # self.delta_matrix = nn.Parameter(torch.Tensor(1, num_types).uniform_(0.4, 0.5))
        # print(self.delta_matrix)

        # self.delta_matrix = nn.Parameter(torch.tensor(1.0))

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            # device=torch.device('cuda'))
            device=torch.device('cpu'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        # print(self.layer_stack)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask, new_mask, epoch):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                new_mask,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

    # def forward(self, enc_output, non_pad_mask, event_type, new_mask, epoch):
    #     """ Encode event sequences via masked self-attention. """
    #
    #     # prepare attention masks
    #     # slf_attn_mask is where we cannot look, i.e., the future and the padding
    #     slf_attn_mask_subseq = get_subsequent_mask(event_type, self.delay_score, new_mask)
    #     slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
    #     slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
    #     slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
    #
    #     # if epoch % 2 == 1:
    #     for enc_layer in self.layer_stack:
    #         enc_output, _ = enc_layer(
    #             enc_output,
    #             new_mask,
    #             non_pad_mask=non_pad_mask,
    #             slf_attn_mask=slf_attn_mask)
    #     return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out

class Masker(nn.Module):
    def __init__(self, num_types, n_events, n_hidden=256, n_mask=4):
        super(Masker, self).__init__()
        # self.fc1 = nn.Linear(n_events * 2, n_hidden)
        self.fc1 = nn.Linear(32, n_hidden)
        self.fc2 = nn.Linear(n_hidden, num_types * num_types)

    def forward(self, x, event_type, event_time):
        # x = torch.normal(15, 3, size=(1, 64))  # noise
        # x = torch.cat((event_time, event_type), 1)
        z = self.fc2(F.relu(self.fc1(x)))
        # m = nn.Dropout(p=0.5)  # sparse
        # z = m(z)
        z = abs(z)
        return z

    # def __init__(self, num_types, n_events, n_hidden=100, n_mask=4):
    #     super(Masker, self).__init__()
    #     self.fc1 = nn.Linear(16 * n_events, n_hidden)
    #     self.fc2 = nn.Linear(n_hidden, num_types * num_types)
    #
    # def forward(self, x):
    #     z = self.fc2(F.relu(self.fc1(x)))
    #     return z

def frange_cycle_linear(start, stop, n_epoch=200, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            epoch_num, num_types, num_events, d_model=16, d_rnn=8, d_inner=64,
            n_layers=1, n_head=4, d_k=4, d_v=4, dropout=0.1):
        super().__init__()

        self.masker = Masker(num_types = num_types, n_events=num_events)
        # print(self.masker)

        # self.mask_latent = nn.Parameter(torch.rand(num_events, num_events), requires_grad=True)
        # self.mask = torch.triu(
        #         torch.ones((num_events, num_events), dtype=torch.uint8), diagonal=1)

        # self.delay_score = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # delay_init = torch.Tensor([[10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        #                            [10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        #                            [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
        #                            [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
        #                            [0.0, 0.0, 0.0, 0.0, 10.0, 10.0],
        #                            [0.0, 0.0, 0.0, 0.0, 10.0, 10.0]])
        # self.delta_matrix = nn.Parameter(delay_init)
        self.delta_matrix = nn.Parameter(torch.Tensor(num_types, num_types).uniform_(10, 10))


        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types
        self.num_events = num_events

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # parameter for sigmoid temperature
        # self.sig_temp = nn.Parameter(torch.tensor(5.0))
        self.sig_temp = frange_cycle_linear(0.1, 1.0)
        # print(self.sig_temp)
        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            # device=torch.device('cuda'))
            device=torch.device('cpu'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, epoch, batch_i):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        non_pad_mask = get_non_pad_mask(event_type)

        # np.savetxt('log_64_time.txt', event_time[0].detach().numpy(), fmt='%.4f')
        # np.savetxt('log_64_type.txt', (event_type[0]-1).detach().numpy(), fmt='%.4f')

        # data = {}
        # for i in range(self.num_types):
        #     data[i] = event_time[0][np.where(event_type[0] == i + 1)]
        # print(data)
        #
        # delay_effect = {}
        # for i in range(self.num_types):
        #     # recording of an event in a certain dimension has a delay effect on the intensity variations of other dimensions
        #     delay_effect[i] = torch.unsqueeze(self.delta_matrix[:, i], dim=1).repeat(1, len(data[i]))
        # print(delay_effect)
        #
        # for dim_idx in range(self.num_types):
        #     for _, cur_t in enumerate(data[dim_idx]):
        #         for neighbor_idx in range(self.num_types):
        #             neighbor_history = torch.Tensor(data[neighbor_idx])
        #             delay = delay_effect[neighbor_idx][dim_idx, : ]
        #             k = cur_t - delay - neighbor_history
        #             print(k)

        # tem_enc = self.temporal_enc(event_time, non_pad_mask)
        # enc_output = self.event_emb(event_type)
        # x = enc_output + tem_enc
        # delta_matrix_1d = self.masker(x.flatten()).detach()
        # np.random.seed(epoch)

        delta_matrix_1d = torch.zeros(self.num_types, self.num_types)
        self.num_events = len(event_type[0])
        batch_size = len(event_type)
        new_mask = torch.ones(self.num_events, self.num_events)
        new_mask = new_mask.repeat(batch_size,1)
        new_mask = new_mask.view(batch_size, self.num_events, self.num_events)
        if 25 < epoch <= 50 or 75 < epoch <= 100:
            x = torch.normal(10, 1, size=(1, 32))  # noise
            delta_matrix_1d = self.masker(x, event_type, event_time)
            # delta_matrix_1d = self.masker(event_type, event_time).detach()
            # delta_matrix_hyper = delta_matrix_1d.reshape([self.num_types, self.num_types])
            # sparse_pattern = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0])
            # delta_matrix_1d = delta_matrix_1d * sparse_pattern

            for k in range(batch_size):
                for i in range(self.num_events):
                    for j in range(self.num_events):
                        delay_i = int((event_type[k] - 1)[i].item())
                        delay_j = int((event_type[k] - 1)[j].item())

                        # delay = self.delta_matrix[delay_i][delay_j]
                        delay = delta_matrix_1d[0][delay_i * self.num_types + delay_j]
                        new_mask[k][i][j] = (event_time[k])[i] - ((event_time[k])[j] + delay)

            # with open('data/toy/3dim_2000seq_32ev/mask.txt', 'a') as f:
            #     for line in new_mask.detach().numpy():
            #         f.write("".join(str(line)) + "\n")
            #     f.write("\n")
            # print("original mask", new_mask.detach().numpy())

            new_mask = torch.sigmoid(10 * self.sig_temp[epoch % 100] * new_mask)

            # ---  tensor
            # x = torch.normal(10, 1, size=(1, 64))  # noise
            # delta_matrix_1d = self.masker(x, event_type, event_time).detach()
            # delta_matrix_hyper = delta_matrix_1d.reshape([self.num_types, self.num_types])
            # print(delta_matrix_hyper)
            #
            # event_type_0 = (event_type[0] - 1)
            # event_time_0 = event_time[0]
            # print(event_type_0)
            # print(event_time_0)
            #
            # # Compute delay for all pairs of events using broadcasting
            # delay_i = (event_type_0.unsqueeze(1)).unsqueeze(2).expand(-1, self.num_events, self.num_events)
            # delay_j = (event_type_0.unsqueeze(0)).unsqueeze(2).expand(self.num_events, -1, self.num_events)
            #
            # print(delay_i)
            # print(delay_j)
            # delay = delta_matrix_hyper[delay_i, delay_j]
            # print(delay)
            #
            # # Compute time differences using broadcasting
            # event_time_i = event_time_0.unsqueeze(0).unsqueeze(2).expand(self.num_events, self.num_events, -1)
            # event_time_j = event_time_0.unsqueeze(1).unsqueeze(0).expand(self.num_events, -1, -1)
            #
            # time_difference = event_time_i - (event_time_j + delay)
            # print(time_difference)
            # new_mask = torch.sigmoid(10 * self.sig_temp[batch_i % 100] * time_difference)
            # print(new_mask.size)
            # --- end


            # with open('data/toy/3dim_2000seq_32ev/mask_sigmoid.txt', 'a') as f:
            #     for line in new_mask.detach().numpy():
            #         f.write("".join(str(line)) + "\n")
            #     f.write("\n")
            # print(new_mask)

            # new_mask = torch.relu(1000 * new_mask)
            # new_mask = F.relu(1000 * new_mask)
            # print(new_mask)

            # tem_enc = self.temporal_enc(event_time, non_pad_mask)
            # enc_output = self.event_emb(event_type)
            # print(tem_enc.shape)
            # print(enc_output.shape)

            # x = enc_output + tem_enc
            # print(x.shape)
            # print(x)

            # x = torch.matmul(self.mask, x)
            # print(x.shape)
            # print(self.mask)

            # enc_output = self.encoder(event_type, event_time, non_pad_mask)
            # enc_output = self.encoder(x, non_pad_mask, event_type, new_mask, epoch)

        enc_output = self.encoder(event_type, event_time, non_pad_mask, new_mask, epoch)
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)
        self.delta_matrix.data.clamp_(min=0)

        return enc_output, (type_prediction, time_prediction), delta_matrix_1d
        # return enc_output, (type_prediction, time_prediction), self.delta_matrix
