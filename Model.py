import torch

class LayersCNN(torch.nn.Module):
    """ CNN layers of net """
    def __init__(self, activation='tanh', pooling='avg'):
        super(LayersCNN, self).__init__()

        if activation == 'tanh':
            activation_function = torch.nn.Tanh()
        elif activation == 'relu':
            activation_function  = torch.nn.ReLU()
        else:
            raise NotImplementedError

        kernel_size = [5, 5, 3, 3, 3]
        features_num = [3, 32, 64, 128, 128, 256]
        pooling_ksize = pooling_stride = [(2,2), (2,2), (2,2), (2,1), (2,1)]

        self.rnn_input = features_num[-1]

        self.conv1 = torch.nn.Conv2d(in_channels=features_num[0], out_channels=features_num[1], kernel_size=kernel_size[0], padding=1)
        self.act1 = activation_function
        
        if pooling == 'avg':
            self.pool1 = torch.nn.AvgPool2d(kernel_size=pooling_ksize[0], stride=pooling_stride[0])
        elif pooling == 'max':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=pooling_ksize[0], stride=pooling_stride[0])
        else:
            raise NotImplementedError
        
        self.conv2 = torch.nn.Conv2d(in_channels=features_num[1], out_channels=features_num[2], kernel_size=kernel_size[1], padding=1)
        self.act2 = activation_function

        if pooling == 'avg':
            self.pool2 = torch.nn.AvgPool2d(kernel_size=pooling_ksize[1], stride=pooling_stride[1])
        elif pooling == 'max':
            self.pool2 = torch.nn.MaxPool2d(kernel_size=pooling_ksize[1], stride=pooling_stride[1])
        else:
            raise NotImplementedError

        self.conv3 = torch.nn.Conv2d(in_channels=features_num[2], out_channels=features_num[3], kernel_size=kernel_size[2], padding=1)
        self.act3 = activation_function

        if pooling == 'avg':
            self.pool3 = torch.nn.AvgPool2d(kernel_size=pooling_ksize[2], stride=pooling_stride[2])
        elif pooling == 'max':
            self.pool3 = torch.nn.MaxPool2d(kernel_size=pooling_ksize[2], stride=pooling_stride[2])
        else:
            raise NotImplementedError

        self.conv4 = torch.nn.Conv2d(in_channels=features_num[3], out_channels=features_num[4], kernel_size=kernel_size[3], padding=1)
        self.act4 = activation_function

        if pooling == 'avg':
            self.pool4 = torch.nn.AvgPool2d(kernel_size=pooling_ksize[3], stride=pooling_stride[3])
        elif pooling == 'max':
            self.pool4 = torch.nn.MaxPool2d(kernel_size=pooling_ksize[3], stride=pooling_stride[3])
        else:
            raise NotImplementedError

        self.conv5 = torch.nn.Conv2d(in_channels=features_num[4], out_channels=features_num[5], kernel_size=kernel_size[4], padding=1)
        self.act5 = activation_function
        
        if pooling == 'avg':
            self.pool5 = torch.nn.AvgPool2d(kernel_size=pooling_ksize[4], stride=pooling_stride[4])
        elif pooling == 'max':
            self.pool5 = torch.nn.MaxPool2d(kernel_size=pooling_ksize[4], stride=pooling_stride[4])
        else:
            raise NotImplementedError
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool5(x)

        return x

class LayersRNN(torch.nn.Module):
    """ RNN layers of net """
    def __init__(self, num_in, num_hidden, num_out):
        super(LayersRNN, self).__init__()

        self.rnn = torch.nn.LSTM(num_in, num_hidden, bidirectional=True)
        self.embedding = torch.nn.Linear(num_hidden * 2, num_out)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() # size of LSTM output: [seq_len, batch, 2(bidirectional) * hidden_size]
        t_rec = recurrent.view(T * b, h) # resize for Linear embedding layer

        output = self.embedding(t_rec)  # size of Linear output: [T * b, num_out]
        output = output.view(T, b, -1)

        return output


class Model(torch.nn.Module):
    """ NN Model for Todo Bicig recognition """
    def __init__(self, rnn_hidden, num_char):
        super(Model, self).__init__()
        self.cnn = LayersCNN()
        self.rnn = torch.nn.Sequential(
            LayersRNN(self.cnn.rnn_input, rnn_hidden, rnn_hidden),
            LayersRNN(rnn_hidden, rnn_hidden, num_char))
        
    def forward(self, x):
        #print(x.size())
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2)
        cnn_out = cnn_out.permute(2, 0, 1)  # [w, b, c]
        #print(cnn_out.size())
        rnn_out = self.rnn(cnn_out)

        # add log_softmax to converge output
        output = torch.nn.functional.log_softmax(rnn_out, dim=2)

        return output