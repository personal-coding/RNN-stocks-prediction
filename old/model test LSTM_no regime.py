import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
from sklearn import metrics
import sys, codecs
import numpy as np
import _pickle as cPickle
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
pd.options.mode.chained_assignment = None

BREAKPOINT=840000
STEP=1440*20
WINDOW=180
FORECAST=240
BATCH_SIZE=64*2
EPOCHS=25
VALID_PERCENTAGE=0.0
TRAIN_TEST_PERCENTAGE=0.90 - VALID_PERCENTAGE
MAIN_FILE = 'data/MAIN_MAIN_1min_onlyFX_min_diff_more_data - Copy.csv'
SAVE_PATH = 'model_LSTM_1min_onlyFX_no_regime_400000_novalid.pickle'
regimes = [0, 1, 2, 3]
DIFF = .00000

def write_file(hold):
    with codecs.open('results_LSTM_1min_onlyFX_no_regime_400000_novalid.csv', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

def save_model(rf):
    with codecs.open(SAVE_PATH, mode='wb') as f:
        cPickle.dump(rf, f)

def create_targets(df, min_diff, xau_ask_volume):
    df.reset_index(drop=True, inplace=True)
    y_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if not min_diff[i - WINDOW + 1:i + 1].any() and not xau_ask_volume[i - WINDOW + 1:i + 1].any():
            if df['EURUSD Bid'][i + FORECAST] > df['EURUSD Ask'][i] + DIFF:
                y_b = [1,0,0]
            elif df['EURUSD Ask'][i + FORECAST] < df['EURUSD Bid'][i] - DIFF:
                y_b = [0,1,0]
            else:
                y_b = [0,0,1]

            y_i.append(np.array(y_b))
    y_i = np.array(y_i)
    return y_i

def window(df, min_diff, xau_ask_volume):
    x_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if i % 10000 == 0:
            print(i)

        if not min_diff[i - WINDOW + 1:i + 1].any() and not xau_ask_volume[i - WINDOW + 1:i + 1].any():
            #x_i.append(df[i-WINDOW+1:i+1].to_numpy())
            x_i.append(df[i - WINDOW + 1:i + 1])

    x_i = np.array(x_i)
    return(x_i)

def reshape_vals(df):
    nsamples, nx, ny = df.shape
    return df.reshape((nsamples, nx * ny))

#Load dataset
df = pd.read_csv(MAIN_FILE, index_col=False)
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

min_diff = df['Min Diff'][BREAKPOINT:]
min_diff.reset_index(drop=True, inplace=True)

xau_ask_volume = df['XAU Ask Volume'][BREAKPOINT:].eq(0) & df['Hour'][BREAKPOINT:].isin([14])
xau_ask_volume.reset_index(drop=True, inplace=True)

df = df.drop(columns=['Min Diff','Hour'], axis=1)

ydata = create_targets(df[['EURUSD Ask','EURUSD Bid']][BREAKPOINT:], min_diff, xau_ask_volume)
print("y data ready")

for column in df:
    #mean = np.mean(df[column][:BREAKPOINT])
    #std = np.std(df[column][:BREAKPOINT]) #, ddof=1

    #df[column] = (df[column] - mean) / std

    iqr = np.subtract(*np.percentile(df[column][:BREAKPOINT], [75, 25]))
    median = np.median(df[column][:BREAKPOINT])

    df[column] = (df[column] - median) / iqr

for column in df:
    if df[column].dtype == np.float64:
        df[column] = df[column].astype(np.float32)

df = df[BREAKPOINT:]
df.reset_index(drop=True, inplace=True)

xdata = window(df.to_numpy(), min_diff, xau_ask_volume)
print("x data ready")

del df
del min_diff
del xau_ask_volume

q = len(xdata)
p = int(q*TRAIN_TEST_PERCENTAGE)
m = int(q*(TRAIN_TEST_PERCENTAGE+VALID_PERCENTAGE))

# es = EarlyStopping(monitor='val_loss', min_delta=3e-4, patience=10, verbose=1, mode='auto')

#x_train, x_valid, x_test, y_train, y_valid, y_test = xdata[:p], xdata[p:m], xdata[m:q], ydata[:p], ydata[p:m], ydata[m:q]
x_train, x_test, y_train, y_test = xdata[:p], xdata[m:q], ydata[:p], ydata[m:q]

'''class_weights = class_weight.compute_class_weight('balanced_subsample',
                                                 np.unique(y_train),
                                                 y_train)'''

x_train, y_train = shuffle(x_train, y_train, random_state=0)

del xdata
del ydata

'''x_train = reshape_vals(x_train)
x_test = reshape_vals(x_test)'''

#Create and fit model
model = Sequential()

model.add(Dropout(0.2))
model.add(LSTM(40, input_shape=x_train.shape))

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

#Train the model using the training sets
model.fit(x_train,
          y_train,
          batch_size=BATCH_SIZE, #int(x_train.shape[0] / 5)
          epochs=EPOCHS,
          verbose=2,
          validation_split=0.0,
          #validation_data=(x_valid, y_valid),
          shuffle=False,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          callbacks=[])

del x_train
del y_train

#Predict the response for test dataset
y_pred = model.predict(x_test)

# Model Accuracy: how often is the classifier correct?
'''print("Accuracy:",metrics.accuracy_score(y_test, y_pred), str(WINDOW), str(FORECAST))
print(metrics.classification_report(y_test,y_pred))'''

y_pred = model.predict_proba(x_test)
y_pred2 = model.predict_classes(x_test)
to_write = list()

for z, a, b in zip(y_pred, y_pred2, y_test):
    new_write = list()

    new_write.append(str(z[0]))
    new_write.append(str(z[1]))
    new_write.append(str(z[2]))

    new_write.append(str(a))

    for i in b:
        new_write.append(str(i))

    to_write.append(new_write)

del x_test
del y_test

write_file(to_write)

save_model(model)

del model