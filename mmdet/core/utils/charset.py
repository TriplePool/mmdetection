import mmcv
import numpy as np
import torch


class Charset:
    def __init__(self, vocab_path):
        self.vocab_dict = mmcv.load(vocab_path)
        self.BOS_TOKEN = self.vocab_dict['stoi']['<s>']
        self.EOS_TOKEN = self.vocab_dict['stoi']['</s>']
        self.UNKNOWN_TOKEN = self.vocab_dict['stoi']['<unk>']
        self.BLANK_TOKEN = self.vocab_dict['stoi']['<blank>']

    def string_to_labels(self, string_input, max_size=32):
        word_target = np.ones((max_size,))
        decoder_target = np.ones((max_size,))
        word_target *= self.BLANK_TOKEN
        decoder_target *= self.EOS_TOKEN
        for i, char in enumerate(string_input):
            if i > max_size - 1:
                break
            decoder_target[i] = self.vocab_dict['stoi'].get(char, self.UNKNOWN_TOKEN)
            word_target[i] = decoder_target[i]
        end_point = min(max(1, len(string_input)), max_size - 1)
        word_target[end_point] = self.EOS_TOKEN
        word_target = torch.as_tensor(word_target).long()
        decoder_target = torch.as_tensor(decoder_target).long()
        return word_target, decoder_target

    def strings_to_labels_tensor(self, string_input_list, max_size=32):
        word_targets, decoder_targets = [], []
        for string_input in string_input_list:
            word_target, decoder_target = self.string_to_labels(string_input, max_size)
            word_targets.append(word_target)
            decoder_targets.append(decoder_target)
        word_targets = torch.stack(word_targets)
        decoder_targets = torch.stack(decoder_targets)
        return word_targets, decoder_targets

    def label_to_string(self, label):
        res = ''
        for c in label:
            if c == self.EOS_TOKEN:
                break
            res += self.vocab_dict['itos'][c]
        return res

    def label_to_char(self, label):
        return self.vocab_dict['itos'][label]

    def __len__(self):
        return len(self.vocab_dict['itos'])


if __name__ == '__main__':
    charset = Charset('/home/wbl/workspace/data/ICDAR2021/vocab.json')
    print(charset.string_to_labels('123'))
