import io
import json
import os
import random
import zipfile

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import Conv2D, Linear, Embedding

src_path = "/home/aistudio/work/data20519/Rumor_Dataset.zip"
target_path = "/home/aistudio/work/Chinese_Rumor_Dataset-master"
if not os.path.isdir(target_path):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()

# paths
rumor_class_dirs = os.listdir(target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
non_rumor_class_dirs = os.listdir(target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
original_microblog = target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"
# rumors 0, non-rumors 1
rumor_label = "0"
non_rumor_label = "1"

rumor_num = 0
non_rumor_num = 0
all_rumor_list = []
all_non_rumor_list = []

# Parse rumor data
for rumor_class_dir in rumor_class_dirs:
    if rumor_class_dir != '.DS_Store':
        with open(original_microblog + rumor_class_dir, 'r') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1

# Parse non-rumor data
for non_rumor_class_dir in non_rumor_class_dirs:
    if non_rumor_class_dir != '.DS_Store':
        with open(original_microblog + non_rumor_class_dir, 'r') as f2:
            non_rumor_content = f2.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1

print("Total rumors: " + str(rumor_num))
print("Total non-rumors: " + str(non_rumor_num))

# Shuffle all data
data_list_path = "/home/aistudio/data/"
all_data_path = data_list_path + "all_data.txt"
all_data_list = all_rumor_list + all_non_rumor_list
random.shuffle(all_data_list)

with open(all_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a') as f:
    for data in all_data_list:
        f.write(data)
print('all_data.txt created')


# create dic
def create_dict(data_path, dict_path):
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate()

    dict_set = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # Convert set to dic format with each char map to index
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # Add an unknown char token
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("Dictionary created!", '\t', 'Dictionary length:', len(dict_list))


def create_data_list(data_list_path):
    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])  # Load dic data

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, \
            open(os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            title = line.split('\t')[-1].replace('\n', '')
            lab = line.split('\t')[0]
            t_ids = ""
            # For every 8 data point, to eval set
            if i % 8 == 0:
                for s in title:
                    temp = str(dict_txt[s])  # Convert char to ID
                    t_ids = t_ids + temp + ','
                t_ids = t_ids[:-1] + '\t' + lab + '\n'  # Format string for eval
                f_eval.write(t_ids)
            else:
                for s in title:
                    temp = str(dict_txt[s])  # Convert char to ID
                    t_ids = t_ids + temp + ','
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_train.write(t_ids)
            i += 1

    print("Data lists created!")


data_root_path = "/home/aistudio/data/"
data_path = os.path.join(data_root_path, 'all_data.txt')
dict_path = os.path.join(data_root_path, "dict.txt")

create_dict(data_path, dict_path)

with open(os.path.join(data_root_path, 'train_list.txt'), 'w', encoding='utf-8') as f_eval:
    f_eval.seek(0)
    f_eval.truncate()

with open(os.path.join(data_root_path, 'eval_list.txt'), 'w', encoding='utf-8') as f_train:
    f_train.seek(0)
    f_train.truncate()

create_data_list(data_root_path)


# Load data in batch
def data_reader(file_path, phrase, shuffle=False):
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) != 2:
                continue
            label = int(cols[1])  # Label as int
            wids = cols[0].split(",")  # Word ID
            all_data.append((wids, label))

    if shuffle:
        if phrase == "train":
            random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label

    return reader


class SentaProcessor(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_data(self, data_dir, shuffle):
        return data_reader(self.data_dir + "train_list.txt", "train", shuffle)

    def get_eval_data(self, data_dir, shuffle):
        return data_reader(self.data_dir + "eval_list.txt", "eval", shuffle)

    # Generate data batch for training or evaluation
    def data_generator(self, batch_size, phase='train', shuffle=True):
        if phase == "train":
            return paddle.batch(
                self.get_train_data(self.data_dir, shuffle),
                batch_size,
                drop_last=True)
        elif phase == "eval":
            return paddle.batch(
                self.get_eval_data(self.data_dir, shuffle),
                batch_size,
                drop_last=True)
        else:
            raise ValueError("Unknown phase, should be 'train' or 'eval'")


# Text classification with CNN BiLSTM Multi-Head Attention
class TextClassifier(fluid.dygraph.Layer):
    def __init__(self, vocab_size, emb_dim=128, hid_dims=[32, 64], lstm_dim=128, class_dim=2, seq_len=150, num_heads=4):
        super(TextClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dims = hid_dims
        self.lstm_dim = lstm_dim
        self.class_dim = class_dim
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Embedding layer
        self.embedding = Embedding(size=[vocab_size + 1, self.emb_dim], dtype='float32', is_sparse=False)

        # Multi-kernel CNN layers
        self.conv1 = Conv2D(num_channels=1, num_filters=self.hid_dims[0], filter_size=(3, self.emb_dim), act="relu")
        self.conv2 = Conv2D(num_channels=1, num_filters=self.hid_dims[1], filter_size=(4, self.emb_dim), act="relu")

        # BiLSTM layer
        self.bilstm = paddle.nn.BiRNN(
            paddle.nn.LSTM(input_size=self.emb_dim, hidden_size=self.lstm_dim // 2),
            direction="bidirectional"
        )

        # Multi-Head Attention layer
        self.multi_head_attention = fluid.dygraph.MultiHeadAttention(
            embed_dim=self.lstm_dim,
            num_heads=self.num_heads,
            dropout_rate=0.1
        )

        # Global max pooling layer
        self.global_max_pool = fluid.dygraph.Pool2D(pool_type="max", global_pooling=True)

        # Dropout layer
        self.dropout = fluid.dygraph.Dropout(p=0.5)

        # Fully connected layers
        self.fc1 = Linear(input_dim=sum(self.hid_dims) + self.lstm_dim, output_dim=128, act="relu")
        self.fc2 = Linear(input_dim=128, output_dim=self.class_dim, act="softmax")

    def forward(self, inputs):
        # Embedding layer
        emb = self.embedding(inputs)
        emb = fluid.layers.reshape(emb, shape=[-1, 1, self.seq_len, self.emb_dim])

        # CNN module
        conv1_out = self.global_max_pool(self.conv1(emb))  # [batch_size, 32, 1, 1]
        conv2_out = self.global_max_pool(self.conv2(emb))  # [batch_size, 64, 1, 1]

        # BiLSTM module
        emb = fluid.layers.reshape(emb, shape=[-1, self.seq_len, self.emb_dim])
        lstm_output, _ = self.bilstm(emb)  # [batch_size, seq_len, lstm_dim]

        # Multi-Head Attention layer
        attention_output = self.multi_head_attention(lstm_output)  # [batch_size, seq_len, lstm_dim]

        # Outputs from CNN and Attention modules
        cnn_features = concat([conv1_out, conv2_out], axis=1)
        cnn_features = fluid.layers.reshape(cnn_features, shape=[-1, sum(self.hid_dims)])

        # Global feature fusion
        combined_features = concat([cnn_features, attention_output], axis=1)

        # Dropout layer
        combined_features = self.dropout(combined_features)

        # Fully connected layers for classification
        fc1_out = self.fc1(combined_features)
        prediction = self.fc2(fc1_out)

        return prediction

    # Model training parameters
    train_parameters = {
        "epoch": 30,
        "batch_size": 16,
        "lr": 0.001,
        "padding_size": 150,
        "vocab_size": 4409,
        "skip_steps": 30,
        "save_steps": 60,
        "checkpoints": "data/"
    }

    def train():
        with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
            processor = SentaProcessor(data_dir="data/")
            train_data_generator = processor.data_generator(
                batch_size=train_parameters["batch_size"],
                phase='train',
                shuffle=True)

            model = TextClassifier(vocab_size=train_parameters["vocab_size"])
            sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=train_parameters["lr"],
                                                    parameter_list=model.parameters())
            steps = 0
            Iters, total_loss, total_acc = [], [], []

            for eop in range(train_parameters["epoch"]):
                for batch_id, data in enumerate(train_data_generator()):
                    steps += 1
                    # Convert data to tensor format
                    doc = to_variable(
                        np.array([
                            np.pad(x[0][0:train_parameters["padding_size"]],
                                   (
                                       0, train_parameters["padding_size"] - len(
                                           x[0][0:train_parameters["padding_size"]])),
                                   'constant',
                                   constant_values=(train_parameters["vocab_size"]))  # Pad with <unk> id
                            for x in data
                        ]).astype('int64').reshape(-1))

                    # Convert labels to tensor format
                    label = to_variable(
                        np.array([x[1] for x in data]).astype('int64').reshape(
                            train_parameters["batch_size"], 1))

                    model.train()
                    prediction, acc = model(doc, label)
                    loss = fluid.layers.cross_entropy(prediction, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()
                    sgd_optimizer.minimize(avg_loss)
                    model.clear_gradients()

                    if steps % train_parameters["skip_steps"] == 0:
                        Iters.append(steps)
                        total_loss.append(avg_loss.numpy()[0])
                        total_acc.append(acc.numpy()[0])
                        print("Epoch: %d, Step: %d, Avg loss: %f, Avg acc: %f" %
                              (eop, steps, avg_loss.numpy(), acc.numpy()))
                    if steps % train_parameters["save_steps"] == 0:
                        save_path = train_parameters["checkpoints"] + "/" + "save_dir_" + str(steps)
                        print('Saving model to: ' + save_path)
                        fluid.dygraph.save_dygraph(model.state_dict(), save_path)

        draw_train_process(Iters, total_loss, total_acc)

    train()

    def to_eval():
        with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
            processor = SentaProcessor(data_dir="data/")
            eval_data_generator = processor.data_generator(
                batch_size=train_parameters["batch_size"],
                phase='eval',
                shuffle=False)

            model_eval = TextClassifier(vocab_size=train_parameters["vocab_size"])
            model, _ = fluid.load_dygraph("data//save_dir_29040.pdparams")
            model_eval.load_dict(model)
            model_eval.eval()

            total_eval_cost, total_eval_acc = [], []
            for eval_batch_id, eval_data in enumerate(eval_data_generator()):
                eval_np_doc = np.array([np.pad(x[0][0:train_parameters["padding_size"]],
                                               (0, train_parameters["padding_size"] - len(
                                                   x[0][0:train_parameters["padding_size"]])),
                                               'constant',
                                               constant_values=(train_parameters["vocab_size"]))
                                        for x in eval_data
                                        ]).astype('int64').reshape(-1)
                eval_label = to_variable(
                    np.array([x[1] for x in eval_data]).astype(
                        'int64').reshape(train_parameters["batch_size"], 1))
                eval_doc = to_variable(eval_np_doc)
                eval_prediction, eval_acc = model_eval(eval_doc, eval_label)
                loss = fluid.layers.cross_entropy(eval_prediction, eval_label)
                avg_loss = fluid.layers.mean(loss)
                total_eval_cost.append(avg_loss.numpy()[0])
                total_eval_acc.append(eval_acc.numpy()[0])

        print("Final validation results: Avg loss: %f, Avg accuracy: %f" %
              (np.mean(total_eval_cost), np.mean(total_eval_acc)))

    to_eval()

    def load_data(sentence):
        with open('data/dict.txt', 'r', encoding='utf-8') as f_data:
            dict_txt = eval(f_data.readlines()[0])
        dict_txt = dict(dict_txt)

        keys = dict_txt.keys()
        data = []
        for s in sentence:
            if s not in keys:
                s = '<unk>'
            data.append(int(dict_txt[s]))
        return data

    train_parameters["batch_size"] = 1
    labels = ['Rumor', 'Not Rumor']

    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        data = load_data('Example input text for rumor detection')
        data_np = np.array(data)
        data_np = np.array(
            np.pad(data_np, (0, 150 - len(data_np)), "constant",
                   constant_values=train_parameters["vocab_size"])).astype(
            'int64').reshape(-1)

        infer_np_doc = to_variable(data_np)

        model_infer = TextClassifier(vocab_size=train_parameters["vocab_size"])
        model, _ = fluid.load_dygraph("data/save_dir_29040.pdparams")
        model_infer.load_dict(model)
        model_infer.eval()
        result = model_infer(infer_np_doc)
        print('Prediction:', labels[np.argmax(result.numpy())])
