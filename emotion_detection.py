import paddle
import numpy as np
import random
import warnings


# The IMDB dataset is a commonly used NLP dataset in PaddlePaddle, containing 25,000 movie reviews labeled as positive or negative
class IMDBDataset(paddle.io.Dataset):

    def __init__(self, sents, labels):
        super(IMDBDataset, self).__init__()
        assert len(sents) == len(labels)
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.sents)

# Load and preprocess IMDB data for fixed-length input
def get_data_loader(mode, seq_len=250, batch_size=128, pad_token='<pad>'):

    imdb_data = paddle.text.Imdb(mode=mode)
    word_dict = imdb_data.word_idx  # word-to-index dic
    word_dict[pad_token] = len(word_dict)  # add padding token to dic
    pad_id = word_dict[pad_token]

    # Uniform length
    def create_padded_dataset(dataset):
        padded_sents, labels = [], []
        for _, data in enumerate(dataset):
            sent, label = data[0], data[1]
            padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))])
            padded_sents.append(padded_sent)
            labels.append(label)
        return np.array(padded_sents, dtype='int64'), np.array(labels, dtype='int64')

    sents_padded, labels_padded = create_padded_dataset(imdb_data)
    dataset_obj = IMDBDataset(sents_padded, labels_padded)
    shuffle = True if mode == 'train' else False  # shuffle data
    data_loader = paddle.io.DataLoader(dataset_obj, shuffle=shuffle, batch_size=batch_size, drop_last=True)
    return data_loader


# LSTM
class LSTMModel(paddle.nn.Layer):
    def __init__(self, embedding_dim, hidden_size, num_layers=2, dropout_rate=0.3):
        super(LSTMModel, self).__init__()

        # Embedding layer  convert word to vectors
        self.emb = paddle.nn.Embedding(num_embeddings=5149, embedding_dim=embedding_dim)

        # LSTM layers  feature extraction
        self.lstm_stack = paddle.nn.LayerList([
            paddle.nn.LSTM(embedding_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        # Dropout layer
        self.dropout = paddle.nn.Dropout(dropout_rate)

        # Batch normalization layer
        self.batch_norm = paddle.nn.BatchNorm1D(hidden_size)

        # Fully connected layer
        self.fc = paddle.nn.Linear(in_features=hidden_size, out_features=2)

        self.softmax = paddle.nn.Softmax()

    def forward(self, x):
        x = self.emb(x)

        # Pass through each LSTM layer in the stack
        for lstm_layer in self.lstm_stack:
            x, (h, c) = lstm_layer(x)

        x = self.batch_norm(h[-1])
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)

# Hyperparameters
seq_len = 250
emb_size = 20
hidden_size = 32
num_layers = 2
dropout_rate = 0.3

train_data_loader = get_data_loader('train', seq_len=seq_len)
test_data_loader = get_data_loader('test', seq_len=seq_len)

model = paddle.Model(
    LSTMModel(embedding_dim=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
)
model.summary(input_size=(None, seq_len), dtype='int64')
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(use_softmax=False),
              metrics=paddle.metric.Accuracy())

model.fit(train_data_loader, epochs=5, verbose=1, eval_data=test_data_loader)
print('Test results:', model.evaluate(test_data_loader, verbose=0))