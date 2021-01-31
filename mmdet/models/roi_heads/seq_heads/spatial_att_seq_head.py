import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from mmdet.core.utils.charset import Charset

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out


def check_all_done(seqs):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True


class Attn(nn.Module):
    def __init__(self, method, hidden_size, embed_size, onehot_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(256 + self.hidden_size + onehot_size, hidden_size)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, H*W, hidden_size)
        attn_energies = self.score(
            H, encoder_outputs
        )  # compute attention score (B, H*W)
        return F.softmax(attn_energies, dim=1).unsqueeze(
            1
        )  # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(
            self.attn(torch.cat([hidden, encoder_outputs], 2))
        )  # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2, 1)  # (B, hidden_size, H*W)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(
            1
        )  # (B, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (B, 1, H*W)
        return energy.squeeze(1)  # (B, H*W)


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(
            self,
            hidden_size,
            embed_size,
            output_size,
            n_layers=1,
            dropout_p=0,
            bidirectional=False,
            onehot_size=(8, 32)
    ):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data = torch.eye(embed_size)
        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(256 + hidden_size + onehot_size[0] + onehot_size[1], hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        # Get the embedding of the current input word (last output word)
        word_embedded_onehot = self.embedding(word_input).view(
            1, word_input.size(0), -1
        )  # (1,B,embed_size)
        word_embedded = self.word_linear(word_embedded_onehot)  # (1, B, hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs)  # (B, 1, H*W)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1)
        )  # (B, 1, H*W) * (B, H*W, C) = (B,1,C)
        context = context.transpose(0, 1)  # (1,B,C)
        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        rnn_input = torch.cat((word_embedded, context), 2)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)
        rnn_input = rnn_input.view(word_input.size(0), -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)
        # Return final output, hidden state
        # print(output.shape)
        return output, hidden, attn_weights


@HEADS.register_module()
class SpatialAttSeqHead(nn.Module):

    def __init__(self,
                 in_roi_size=(16, 64),
                 in_channels=256,
                 vocab_path='/home/wbl/workspace/data/ICDAR2021/vocab.json',
                 hidden_dim=512,
                 max_len=32,
                 teacher_force_ratio=1.0):
        super(SpatialAttSeqHead, self).__init__()
        self.charset = Charset(vocab_path)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_len
        self.teacher_force_ratio = teacher_force_ratio

        self.seq_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.x_onehot_size = int(in_roi_size[1] / 2)
        self.y_onehot_size = int(in_roi_size[0] / 2)

        self.seq_decoder = BahdanauAttnDecoderRNN(
            self.hidden_dim, len(self.charset), len(self.charset), n_layers=1, dropout_p=0.1,
            onehot_size=(self.y_onehot_size, self.x_onehot_size)
        )

        self.criterion_seq_decoder = nn.NLLLoss(ignore_index=-1, reduction="none")
        # self.rescale = nn.Upsample(size=(16, 64), mode="bilinear", align_corners=False)
        self.rescale = nn.Upsample(size=in_roi_size, mode="bilinear",
                                   align_corners=False)

        self.x_onehot = nn.Embedding(self.x_onehot_size, self.x_onehot_size)
        self.x_onehot.weight.data = torch.eye(self.x_onehot_size)
        self.y_onehot = nn.Embedding(self.y_onehot_size, self.y_onehot_size)
        self.y_onehot.weight.data = torch.eye(self.y_onehot_size)

    def init_weights(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    @auto_fp16()
    def forward(self, x, decoder_targets=None, word_targets=None, use_beam_search=False, is_training=False):
        rescale_out = self.rescale(x)
        seq_decoder_input = self.seq_encoder(rescale_out)

        x_t, y_t = np.meshgrid(np.linspace(0, self.x_onehot_size - 1, self.x_onehot_size),
                               np.linspace(0, self.y_onehot_size - 1, self.y_onehot_size))
        x_t = torch.LongTensor(x_t).cuda()
        y_t = torch.LongTensor(y_t).cuda()
        x_onehot_embedding = (
            self.x_onehot(x_t)
                .transpose(0, 2)
                .transpose(1, 2)
                .repeat(seq_decoder_input.size(0), 1, 1, 1)
        )
        y_onehot_embedding = (
            self.y_onehot(y_t)
                .transpose(0, 2)
                .transpose(1, 2)
                .repeat(seq_decoder_input.size(0), 1, 1, 1)
        )
        seq_decoder_input_loc = torch.cat(
            [seq_decoder_input, x_onehot_embedding, y_onehot_embedding], 1
        )
        seq_decoder_input_reshape = (
            seq_decoder_input_loc.view(
                seq_decoder_input_loc.size(0), seq_decoder_input_loc.size(1), -1
            )
                .transpose(0, 2)
                .transpose(1, 2)
        )
        if is_training:
            bos_onehot = np.zeros(
                (seq_decoder_input_reshape.size(1), 1), dtype=np.int32
            )
            bos_onehot[:, 0] = self.charset.BOS_TOKEN
            decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)
            decoder_hidden = torch.zeros(
                (seq_decoder_input_reshape.size(1), self.hidden_dim), device=gpu_device
            )
            use_teacher_forcing = (
                True
                if random.random() < self.teacher_force_ratio
                else False
            )
            target_length = decoder_targets.size(1)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    decoder_input = decoder_targets[:, di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(
                        1
                    ).detach()  # detach from history as input
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
            loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.size(0)
            loss_seq_decoder = 0.2 * loss_seq_decoder
            return loss_seq_decoder
        else:
            words = []
            decoded_scores = []
            detailed_decoded_scores = []
            # real_length = 0
            decoder_hiddens = torch.zeros((seq_decoder_input_reshape.size(1), self.hidden_dim),
                                          device=gpu_device)
            if use_beam_search:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    word = []
                    char_scores = []
                    detailed_char_scores = []
                    top_seqs = self.beam_search(
                        seq_decoder_input_reshape[:, batch_index: batch_index + 1, :],
                        decoder_hiddens[batch_index: batch_index + 1, ...],
                        beam_size=6,
                        max_len=self.max_seq_len,
                    )
                    top_seq = top_seqs[0]
                    for character in top_seq[1:]:
                        character_index = character[0]
                        if character_index == self.charset.EOS_TOKEN:
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                            break
                        else:
                            if character_index == self.charset.UNKNOWN_TOKEN:
                                word.append("~")
                                char_scores.append(0.0)
                            else:
                                word.append(self.charset.label_to_char(character_index))
                                char_scores.append(character[1])
                                detailed_char_scores.append(character[2])
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
                    detailed_decoded_scores.append(detailed_char_scores)
            else:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    bos_onehot = np.zeros((1, 1), dtype=np.int32)
                    bos_onehot[:, 0] = self.charset.BOS_TOKEN
                    decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)
                    decoder_hidden = torch.zeros((1, self.hidden_dim), device=gpu_device)
                    word = []
                    char_scores = []
                    for di in range(self.max_seq_len):
                        decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                            decoder_input,
                            decoder_hidden,
                            seq_decoder_input_reshape[
                            :, batch_index: batch_index + 1, :
                            ],
                        )
                        # decoder_attentions[di] = decoder_attention.data
                        topv, topi = decoder_output.data.topk(1)
                        char_scores.append(topv.item())
                        if topi.item() == self.charset.EOS_TOKEN:
                            break
                        else:
                            if topi.item() == self.charset.UNKNOWN_TOKEN:
                                word.append("~")
                            else:
                                word.append(self.charset.label_to_char(topi.item()))

                        # real_length = di
                        decoder_input = topi.squeeze(1).detach()
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
            return words, decoded_scores, detailed_decoded_scores

    def get_targets(self, sampling_results, gt_seqs):
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        word_targets_list, decoder_targets_list = [], []
        for i, seqs in enumerate(gt_seqs):
            word_targets, decoder_targets = seqs
            word_targets, decoder_targets = word_targets[pos_assigned_gt_inds[i]], decoder_targets[
                pos_assigned_gt_inds[i]]
            word_targets_list.append(word_targets)
            decoder_targets_list.append(decoder_targets)

        if len(word_targets_list) > 0:
            word_targets = torch.cat(word_targets_list)
            decoder_targets = torch.cat(decoder_targets_list)
        else:
            word_targets = torch.tensor(word_targets_list)
            decoder_targets = torch.tensor(decoder_targets_list)

        return word_targets, decoder_targets
