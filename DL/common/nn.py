import numpy as np
import pandas as pd
from collections import OrderedDict
from common.layers import Convolution, MaxPooling, ReLU, Affine, Dropout, SoftmaxWithLoss

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        input_size : 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : 畳み込みの条件, dict形式　　例、{'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
        hidden_size : 隠れ層のノード数
        output_size : 出力層のノード数
        weight_init_std ：　重みWを初期化する際に用いる標準偏差
        """
                
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 重みの初期化
        self.params = {}
        std = weight_init_std
        self.params['W1_1'] = std * np.random.randn(16, 1, 3,3) 
        self.params['b1_1'] = np.zeros(16)
        self.params['W1_3'] = std * np.random.randn(16, 16, 3,3) 
        self.params['b1_3'] = np.zeros(16)

        self.params['W2_1'] = std * np.random.randn(32, 16, 3,3) 
        self.params['b2_1'] = np.zeros(32)
        self.params['W2_3'] = std * np.random.randn(32, 32, 3,3) 
        self.params['b2_3'] = np.zeros(32)

        self.params['W3_1'] = std * np.random.randn(64, 32, 3,3) 
        self.params['b3_1'] = np.zeros(64)
        self.params['W3_3'] = std * np.random.randn(64, 64, 3,3) 
        self.params['b3_3'] = np.zeros(64)
        
        
        self.params['W4_1'] = std * np.random.randn(64*4*4, hidden_size)
        self.params['b4_1'] = np.zeros(hidden_size)
        self.params['W5_1'] = std * np.random.randn(hidden_size, output_size)
        self.params['b5_1'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1_1'] = Convolution(self.params['W1_1'], self.params['b1_1'],1,1)
        self.layers['ReLU1_2'] = ReLU()
        self.layers['Conv1_3'] = Convolution(self.params['W1_3'], self.params['b1_3'],1,1)
        self.layers['ReLU1_4'] = ReLU()
        self.layers['Pool1_5'] = MaxPooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2_1'] = Convolution(self.params['W2_1'], self.params['b2_1'],1,1)
        self.layers['ReLU2_2'] = ReLU()
        self.layers['Conv2_3'] = Convolution(self.params['W2_3'], self.params['b2_3'],1,2)
        self.layers['ReLU2_4'] = ReLU()
        self.layers['Pool2_5'] = MaxPooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3_1'] = Convolution(self.params['W3_1'], self.params['b3_1'],1,1)
        self.layers['ReLU3_2'] = ReLU()
        self.layers['Conv3_3'] = Convolution(self.params['W3_3'], self.params['b3_3'],1,1)
        self.layers['ReLU3_4'] = ReLU()
        self.layers['Pool3_5'] = MaxPooling(pool_h=2, pool_w=2, stride=2)        
        
        
        self.layers['Affine4_1'] = Affine(self.params['W4_1'], self.params['b4_1'])
        self.layers['ReLU4_2'] = ReLU()        
        self.layers['Dropout4_3'] = Dropout(0.5)
        
        self.layers['Affine5_1'] = Affine(self.params['W5_1'], self.params['b5_1'])
        self.layers['Dropout5_2'] = Dropout(0.5)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1_1'], grads['b1_1'] = self.layers['Conv1_1'].dW, self.layers['Conv1_1'].db
        grads['W1_3'], grads['b1_3'] = self.layers['Conv1_3'].dW, self.layers['Conv1_3'].db
        grads['W2_1'], grads['b2_1'] = self.layers['Conv2_1'].dW, self.layers['Conv2_1'].db
        grads['W2_3'], grads['b2_3'] = self.layers['Conv2_3'].dW, self.layers['Conv2_3'].db
        grads['W3_1'], grads['b3_1'] = self.layers['Conv3_1'].dW, self.layers['Conv3_1'].db
        grads['W3_3'], grads['b3_3'] = self.layers['Conv3_3'].dW, self.layers['Conv3_3'].db
        grads['W4_1'], grads['b4_1'] = self.layers['Affine4_1'].dW, self.layers['Affine4_1'].db
        grads['W5_1'], grads['b5_1'] = self.layers['Affine5_1'].dW, self.layers['Affine5_1'].db

        return grads
