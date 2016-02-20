import cPickle as pickle
import pandas as pd
from pandas  import read_csv

ys = pickle.load(open('y_pred7.pickle','rb'))

df = read_csv('../training.csv')

idlook = read_csv('../idlook.csv')

mapper = dict(zip(df.columns[:-1], xrange(0, len(df.columns))))

arr = []

for x in idlook.values:
    unscaled_y =ys[x[1]-1][mapper[x[2]]]
    scaled_y = int(round((unscaled_y + 1) * 48))
    arr.append([x[0], scaled_y])


pdf2 = pd.DataFrame(arr, columns=['RowId','Location'])
pdf2.to_csv('submission7.csv', columns=['RowId','Location'], index=False)

