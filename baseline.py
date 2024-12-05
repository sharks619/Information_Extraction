import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

import numpy as np
import os, math
import matplotlib.pyplot as plt


# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class LabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[SOS]', '[EOS]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=0):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.

        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        init_batch_max_length = max(max(length), batch_max_length)
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = np.zeros((len(text), init_batch_max_length + 1), dtype=np.int64)

        for i, t in enumerate(text):
            text = list(t)
            text.append('[EOS]')
            text = [self.dict[char] for char in text]
            if len(text) > batch_max_length:
                batch_max_length = len(text)
            batch_text[i][1:1 + len(text)] = text

        return batch_text[:, 0:batch_max_length + 1], batch_max_length + 1

    def decode(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index in range(text_index.shape[0]):
            text = ''.join([self.character[i] for i in text_index[index, :] if not i in [0, 1]])
            texts.append(text)
        return texts


class MyDataset(data.Dataset):
    def __init__(self, datas, labels, vocab_size, max_len, phase = 'train', org_datas =None):
        self.datas = datas
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.order = np.random.permutation(len(self.datas))
        self.phase = phase
        self.org_datas = org_datas

    def __getitem__(self, index):
        if self.phase=='train' :
            return self.datas[self.order[index]], self.labels[self.order[index]]
        elif self.phase=='test':
            return self.datas[self.order[index]], self.labels[self.order[index]]

    def __len__(self):
        return len(self.datas)

    def rand_order(self):
        self.order = np.random.permutation(len(self.datas))


class ExtModel(nn.Module):
    def __init__(self, embed_size, vocab_size, output_size, dropout_keep=0.2, max_sequence_length=100):
        """
        :param embed_size:
        :param vocab_size:
        :param output_size:
        :param dropout_keep:
        """
        super(ExtModel, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.dropout_keep = dropout_keep
        self.max_sequence_length = max_sequence_length

        self.FeatureExtraction = nn.Embedding(self.vocab_size, self.embed_size)
        # Fully-Connected Layer
        self.linear = nn.Linear(self.embed_size * max_sequence_length, self.output_size * 2)
        self.fc = nn.Linear(self.output_size * 2, self.vocab_size * max_sequence_length)

    def forward(self, input):
        '''
        :param input: [B, L, V]
                    B : Batch size
                    L : Max sequence length
                    V : Vocab Size
        :return:
        '''
        embed_sent = self.FeatureExtraction(input)
        batch_size = embed_sent.shape[0]
        embed_sent = embed_sent.view(batch_size, -1)
        embed_linear = self.linear(embed_sent)
        final_out = self.fc(embed_linear)
        final_out = final_out.view(batch_size, -1, self.vocab_size)
        return final_out


def train_epoch(net, optimizer, scheduler,
                train_loader, device, criterion, all_step, softmax, converter, show_labels=False):
    net.train()
    train_loss = 0.
    pos_cnt = 0
    neg_cnt = 0
    predictions = []
    ground_truths = []

    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward
        y1 = net(input=data)
        loss = criterion(y1.view(-1, y1.shape[-1]), labels.contiguous().view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Accuracy
        y1 = softmax(y1)
        output_arr = np.argmax(y1.detach().cpu().numpy(), axis=-1)
        if converter is not None:
            output = converter.decode(output_arr)
            gt = converter.decode(labels.cpu().numpy())

            for oo in range(len(output)):
                if not output[oo] == gt[oo]:
                    neg_cnt += 1
                else:
                    pos_cnt += 1

                # Store predictions and ground truths for debugging
                if show_labels and len(predictions) < 10:  # Show only first 10 samples
                    predictions.append(output[oo])
                    ground_truths.append(gt[oo])

        train_loss += loss.item()

    train_accuracy = pos_cnt / (neg_cnt + pos_cnt)

    # Print predictions and ground truths if enabled
    if show_labels:
        print("Sample predictions (train):")
        for pred, gt in zip(predictions, ground_truths):
            print(f"Prediction: {pred}, Ground Truth: {gt}")

    return train_loss / all_step, train_accuracy


def eval_epoch(net, eval_loader, criterion, device, softmax, converter, show_labels=False):
    net.eval()
    loss = 0
    pos_cnt = 0
    neg_cnt = 0
    predictions = []
    ground_truths = []

    for i, (data, labels) in enumerate(eval_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            y1 = net(input=data)
            loss_t = criterion(y1.view(-1, y1.shape[-1]), labels.contiguous().view(-1))
            loss += loss_t.item()

            y1 = softmax(y1)
            output_arr = np.argmax(y1.cpu().numpy(), axis=-1)

            if converter is not None:
                output = converter.decode(output_arr)
                gt = converter.decode(labels.cpu().numpy())

                for oo in range(len(output)):
                    if not output[oo] == gt[oo]:
                        neg_cnt += 1
                    else:
                        pos_cnt += 1

                    # Store predictions and ground truths for debugging
                    if show_labels and len(predictions) < 10:  # Show only first 10 samples
                        predictions.append(output[oo])
                        ground_truths.append(gt[oo])

    # Print predictions and ground truths if enabled
    if show_labels:
        print("Sample predictions (validation):")
        for pred, gt in zip(predictions, ground_truths):
            print(f"Prediction: {pred}, Ground Truth: {gt}")

    accuracy = pos_cnt / (neg_cnt + pos_cnt) if (neg_cnt + pos_cnt) > 0 else 0
    return loss / len(eval_loader), accuracy


def load_data(_path, max_text_length=100, max_batch_size=64, shuffle=False, num_workers=0):
    flist = os.listdir(_path)
    train_data_list = []
    train_label_list = []

    char_path = './char_map_codeT.txt'
    f = open(char_path, 'r', -1, "utf-8")
    line = f.readline()
    character = line.strip('\n')
    converter = LabelConverter(character)

    for i in range(len(flist)):
        fid_data =open(os.path.join(_path,flist[i]),'r',-1, "cp949")
        datalines = fid_data.readlines()
        fid_data.close()

        data = [line_.split(':____GT:')[0].strip('\n') for line_ in datalines]
        label= [line_.split(':____GT:')[1].strip('\n') for line_ in datalines]

        train_data_list+=data
        train_label_list+=label

    encoded_train_data_list, max_len = converter.encode(train_data_list, batch_max_length=max_text_length)
    encoded_train_label_list, output_max_len = converter.encode(train_label_list, batch_max_length=max_len - 1)

    train_dataset = MyDataset(datas=encoded_train_data_list,
                              labels=encoded_train_label_list,
                              vocab_size=len(converter.character),
                              max_len=max_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=max_batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    return train_loader, None, converter


def save_check_point(model, model_info, checkpoint_path):
    state = {'state_dict': model.state_dict(),
             'model_info': model_info}
    torch.save(state, checkpoint_path)


def nn_model(train_iter, valid_iter, converter, device, embed_size=64, ckpt_save_path='./output'):
    model = ExtModel(embed_size=embed_size,
                     vocab_size=train_iter.dataset.vocab_size,
                     output_size=train_iter.dataset.vocab_size,
                     max_sequence_length=train_iter.dataset.max_len)
    model = model.to(device)

    model_info = {'embed_size': model.embed_size,
                  'vocab_size': model.vocab_size,
                  'output_size': model.output_size,
                  'dropout_keep': model.dropout_keep,
                  'max_sequence_length': model.max_sequence_length}

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    softmax = torch.nn.Softmax(dim=2)

    start_epoch = int(0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.9)

    all_step = len(train_iter)
    best_model = {'eval_loss': math.inf, 'eval_accuracy': 0, 'models': ''}
    total_epochs = int(100)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_iter.dataset.rand_order()
            # train
            train_loss, train_acc = train_epoch(model, optimizer,
                                                scheduler, train_iter,
                                                device,
                                                criterion, all_step, softmax, converter, show_labels=False)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            print(f'[{epoch}/{total_epochs}] - train_loss {train_loss}, train_acc {train_acc}')

            # valid
            if valid_iter is not None:
                eval_v, eval_ratio = eval_epoch(model, valid_iter, criterion, device, softmax, converter, show_labels=False)
                scheduler.step(eval_ratio)

                val_losses.append(eval_v)
                val_accuracies.append(eval_ratio)

                # save_checkpoint
                if eval_ratio > best_model['eval_accuracy']:
                    best_model['eval_loss'] = eval_v
                    best_model['eval_accuracy'] = eval_ratio
                    best_model['models'] = 'epoch_%d_model' % epoch

                    if not os.path.exists(ckpt_save_path):
                        os.makedirs(ckpt_save_path, exist_ok=True)
                    save_check_point(model, model_info, os.path.join(ckpt_save_path, 'best_accuracy.pth'))
                best_accuracy = best_model['eval_accuracy']
                print(f'[{epoch}/{total_epochs}] - valid loss {eval_v}, valid accuracy {eval_ratio} [best accuracy: {best_accuracy}]')

    except Exception as e:
        print('Exception occurred:', e)

    # Plotting train and val accuracy and loss side-by-side
    if len(train_losses) == len(val_losses):  # Check for matching lengths
        epochs = range(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Loss plot
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Validation Loss', color='orange')
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Add the main title
        fig.suptitle('Baseline model', fontsize=16)

        plt.tight_layout()
        plt.show()

    else:
        print("Mismatch in lengths of train and validation metrics. Skipping plot.")

    return {}


def load_check_point(net, checkpoint_path_state):
    pre_dict = checkpoint_path_state

    stat = net.state_dict()
    for key in stat:
        if key in pre_dict:
            stat[key] = pre_dict[key]
        else:
            print(key, '<-- None')

    net.load_state_dict(stat)

    return net


def inference(test_iter, converter, device, ckpt_path):
    model_infos = torch.load(ckpt_path)
    model_info = model_infos['model_info']

    model = ExtModel(embed_size=model_info['embed_size'],
                     vocab_size=model_info['vocab_size'],
                     output_size=model_info['output_size'],
                     max_sequence_length=model_info['max_sequence_length'])

    model = model.to(device)
    model = load_check_point(model, model_infos['state_dict'])
    softmax = torch.nn.Softmax(dim=2)

    pos_cnt = 0
    neg_cnt = 0

    predictions = []
    references = []

    with torch.no_grad():
        model.eval()
        for i, (data, labels) in enumerate(test_iter):
            data, labels = data.to(device), labels.to(device)
            y1 = softmax(model(input=data))
            output_arr = np.argmax(y1.to(torch.device('cpu')).numpy(), axis=-1)
            if converter is not None:
                output = converter.decode(output_arr)
                gt = converter.decode(labels.to(torch.device('cpu')).numpy())
                predictions.extend(output)
                references.extend(gt)

                for oo in range(len(output)):
                    if not output[oo] == gt[oo]:
                        neg_cnt += 1
                    else:
                        pos_cnt += 1

    # 출력: 예측값과 실제값 10개씩 프린트
    print("Sample Predictions and References:")
    for pred, ref in zip(predictions[:10], references[:10]):
        print(f"Predicted: {pred} | Reference: {ref}")

    test_result = pos_cnt / (neg_cnt + pos_cnt)
    print('test_results %.3f' % test_result)

    return predictions, references



train_loader, _, converter = load_data('./data/train', max_text_length=100, shuffle=True)
eval_loader, _, _ = load_data('./data/valid', max_text_length=100, shuffle=False)

nn_model(train_loader, eval_loader, converter, torch.device('cuda:0'), embed_size=128, ckpt_save_path='./output/model_ckpt')

test_loader, _, convertrt =  load_data('./data/test', max_text_length=100, shuffle=False)
_ = inference(test_loader, converter, torch.device('cuda:0'), ckpt_path='./output/model_ckpt/best_accuracy.pth')