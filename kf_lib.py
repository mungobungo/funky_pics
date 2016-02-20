import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
import cPickle as pickle

FTRAIN = '~/gamma/training.csv'
FTEST = '~/gamma/test.csv'

def load(test=False, cols = None):
	fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))

	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:
		df = df[list(cols) + ['Image']]
	print(df.count())
	df = df.dropna()
	
	X = np.vstack(df['Image'].values) / 255.
	X = X.astype(np.float32)
	
	if not test:
		y = df[df.columns[:-1]].values
		y = (y-48) /48 # scale coordinates to [-1,1]
		X, y = shuffle(X, y, random_state = 42)
		y = y.astype(np.float32)
	else:
		y = None

	return X, y


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def load2d(test=False, cols = None):
	X, y = load(test=test)
	X = X.reshape(-1,1,96,96)
	return X,y


def plot_train(net, name):
	
	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	fig = pyplot.figure(1)	
	pyplot.plot(train_loss, linewidth=3, label=name+"_train")
	pyplot.plot(valid_loss, linewidth=3, label=name+"_valid1")

	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.yscale("log")
	pyplot.savefig(name + ".png")
	pyplot.close()


def plot_faces(X, y, name):
	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust( left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y[i], ax)

	fig.savefig(name + "_faces.png")
	pyplot.close()

from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


import theano

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net6 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
	('dropout5',layers.DropoutLayer),
	('hidden6',layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=1000,
    dropout4_p=0.4,  # !
    hidden5_num_units=750,
    dropout5_p=0.5,
    hidden6_num_units=300,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=10000,
    verbose=1,
    )


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

import sys
sys.setrecursionlimit(10000)

X,y = load()
## net1.fit(X,y)
net1 = pickle.load(open('net1.pickle','rb'))

testX, _ = load(test=True)
y_pred = net1.predict(testX)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
# import cPickle as pickle
# with open('net1.pickle', 'wb') as f:
#    pickle.dump(net1, f, -1)

plot_train(net1, "net1")
plot_faces(testX, y_pred, "net1")


X, y = load2d()  # load 2-d data
net6.fit(X, y)

with open('net7.pickle','wb') as f:
	pickle.dump(net6, f, -1)

net6 = pickle.load(open('net7.pickle', 'rb'))

plot_train(net6, "net7")
convX,_ = load2d(test=True)
y_pred6 = net6.predict(convX)

with open('y_pred7.pickle', 'wb') as f:
	pickle.dump(y_pred6, f,-1)

plot_faces(convX, y_pred6, "net7")


