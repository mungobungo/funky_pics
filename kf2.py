import cPickle as pickle
##with open('net2.pickle', 'wb') as f:
net2 = pickle.load(open( 'net2.pickle', 'rb'))
def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

