import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import sys, codecs
import numpy as np
import _pickle as pickle
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None

BREAKPOINT=430000
WINDOW=180
FORECAST=240
TRAIN_TEST_PERCENTAGE=1
MAIN_FILE = 'data/MAIN_1min_onlyFX.csv'
SAVE_PATH = 'model_1min_onlyFX_no_regime.pickle'
regimes = [0, 1, 2, 3]
DIFF = .00000

def write_file(hold):
    with codecs.open('results_rf_1min_onlyFX_no_regime.csv', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

def save_model(rf):
    with codecs.open(SAVE_PATH, mode='wb') as f:
        pickle.dump(rf, f)

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    # shuffling of training data
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def create_targets(df):
    df.reset_index(drop=True, inplace=True)
    y_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if df['EURUSD Bid'][i + FORECAST] > df['EURUSD Ask'][i]:
            y_b = [1,0,0]
        elif df['EURUSD Ask'][i + FORECAST] < df['EURUSD Bid'][i]:
            y_b = [0,1,0]
        else:
            y_b = [0,0,1]

        y_i.append(np.array(y_b))
    y_i = np.array(y_i)
    return y_i

def window(df):
    x_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        x_i.append(df[i-WINDOW+1:i+1].as_matrix())

    x_i = np.array(x_i)
    return(x_i)

def reshape_vals(df):
    nsamples, nx, ny = df.shape
    return df.reshape((nsamples, nx * ny))

#Load dataset
df = pd.read_csv(MAIN_FILE, index_col=False)#.astype(np.float32)
df = df.drop(columns=['Datetime'], axis=1)
'''df['EURUSD Spread'] = df['EURUSD Ask'] - df['EURUSD Bid']
df['EURJPY Spread'] = df['EURJPY Ask'] - df['EURJPY Bid']
df['EURGBP Spread'] = df['EURGBP Ask'] - df['EURGBP Bid']
df['EURCAD Spread'] = df['EURCAD Ask'] - df['EURCAD Bid']
df['EURCHF Spread'] = df['EURCHF Ask'] - df['EURCHF Bid']
df['XAU Spread'] = df['XAU Ask'] - df['XAU Bid']
df['EURHKD Spread'] = df['EURHKD Ask'] - df['EURHKD Bid']
df['USDCNH Spread'] = df['USDCNH Ask'] - df['USDCNH Bid']
df['EURSGD Spread'] = df['EURSGD Ask'] - df['EURSGD Bid']'''

#df = df.drop(columns=['Y_regime_0'], axis=1)
#df = df.drop(columns=['X_regime_0', 'Y_regime_0'], axis=1)

ydata = create_targets(df[BREAKPOINT:])
print("y data ready")

for column in df:
    #if not 'regime' in column:
    #df[column] = np.log(df[column])

    mean = np.mean(df[column][:BREAKPOINT])
    std = np.std(df[column][:BREAKPOINT]) #, ddof=1

    df[column] = (df[column] - mean) / std

for column in df:
    if df[column].dtype == np.float64:
        df[column] = df[column].astype(np.float32)
    else:
        df[column] = df[column].astype(np.int8)

'''pca = PCA(.95) #n_components
pca = pca.fit(df[:BREAKPOINT])'''

df = df[BREAKPOINT:]
df.reset_index(drop=True, inplace=True)

'''principalComponents = pca.transform(df)
df = pd.DataFrame(data=principalComponents)'''

xdata = df
xdata = window(xdata)
print("x data ready")

del df

q = len(xdata)
p = int(q*TRAIN_TEST_PERCENTAGE)

#Creat and fit model
model = Sequential()

model.add(Dropout(0.2))
model.add(LSTM(40))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(0.2))
model.add(Dense(30, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

x_train, y_train = xdata[:p], ydata[:p]

del xdata
del ydata

x_train = reshape_vals(x_train)

x_train, y_train = shuffle_in_unison(x_train, y_train)

#Train the model using the training sets
clf.fit(x_train, y_train)

del x_train
del y_train

#save_model(clf)

del clf