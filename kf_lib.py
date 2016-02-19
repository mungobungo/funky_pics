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


#X, y = load2d()  # load 2-d data
#net2.fit(X, y)

net2 = pickle.load(open('net2.pickle', 'rb'))

plot_train(net2, "net2")
convX,_ = load2d(test=True)
y_pred2 = net2.predict(convX)

plot_faces(convX, y_pred2, "net2")

