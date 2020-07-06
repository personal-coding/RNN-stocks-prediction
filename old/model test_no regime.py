import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys, codecs
import numpy as np
import _pickle as cPickle
import talib
pd.options.mode.chained_assignment = None

MAX_FEATURES = 5
rng = np.random.RandomState(42)
BREAKPOINT=840000
BREAK=0
PAIR_ASK = 'EURUSD Ask'
PAIR_BID = 'EURUSD Bid'
WINDOW=int(60*3)
TIMESTEP=int(60*2)
FORECAST=int(60*6)
TRAIN_TEST_PERCENTAGE=0.90
MAIN_FILE = 'data/MAIN_1min_onlyFX_min_diff_more_data - Copy Copy.csv'
SAVE_PATH = 'model_1min_onlyFX_no_regime_more_data_balanced.pickle'
DIFF = 0.00000
tickers = []

def write_file(hold):
    with codecs.open('results_rf_1min_onlyFX_no_regime_min_diff_balanced.csv', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

def save_model(rf):
    with codecs.open(SAVE_PATH, mode='wb') as f:
        cPickle.dump(rf, f, -1)

def create_targets(df, test):
    df.reset_index(drop=True, inplace=True)
    y_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if i % 10000 == 0:
            print('y', i)

        if not test[i]:
            if df[PAIR_BID][i + FORECAST] > df[PAIR_ASK][i] + DIFF:
                y_b = [1,0,0]
            elif df[PAIR_ASK][i + FORECAST] < df[PAIR_BID][i] - DIFF:
                y_b = [0,1,0]
            else:
                y_b = [0,0,1]

            y_i.append(np.array(y_b))
    y_i = np.array(y_i)
    return y_i

def window(df, test):
    x_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if i % 10000 == 0:
            print('x', i)

        if not test[i]:
            x_i.append(df[i - WINDOW + 1:i + 1])

    x_i = np.array(x_i)
    return(x_i)

def reshape_vals(df):
    nsamples, nx, ny = df.shape
    return df.reshape((nsamples, nx * ny))

#Load dataset
df = pd.read_csv(MAIN_FILE, index_col=False)
df = df.drop(columns=['Datetime'], axis=1)

min_diff = df['Min Diff'][BREAKPOINT:]
min_diff.reset_index(drop=True, inplace=True)

xau_ask_volume = df['XAU Ask Volume'][BREAKPOINT:].eq(0) & df['Hour'][BREAKPOINT:].isin([14])
xau_ask_volume.reset_index(drop=True, inplace=True)

df = df.drop(columns=['Min Diff','Hour'], axis=1)

df['test'] = True
test = df['test'][BREAKPOINT:]
test.reset_index(drop=True, inplace=True)

for i in range(WINDOW, len(df[BREAKPOINT:]) - FORECAST):
    if not min_diff[i - WINDOW + 1:i + 1].any() and not xau_ask_volume[i - WINDOW + 1:i + 1].any():
        test[i] = False

df = df.drop(columns=['test'], axis=1)

ydata = create_targets(df[[PAIR_ASK,PAIR_BID]][BREAKPOINT:], test)
print("y data ready")

'''high, low, close, open = df['EURUSD Ask High'],df['EURUSD Ask Low'], df['EURUSD Ask'], df['EURUSD Ask Open']
sma = talib.SMA(close, timeperiod=TIMESTEP)
ema = talib.EMA(close, timeperiod=TIMESTEP)
atr = talib.ATR(high,low, close, timeperiod=TIMESTEP)
cci = talib.CCI(high,low, close, timeperiod=TIMESTEP)
roc = talib.ROC(close, timeperiod=TIMESTEP)
rsi = talib.RSI(close, timeperiod=TIMESTEP)
willr = talib.WILLR(high,low, close, timeperiod=TIMESTEP)
fastk, fastd = talib.STOCHF(high,low, close, fastk_period=TIMESTEP, fastd_period=int(TIMESTEP/2), fastd_matype=0)

df['sma'] = sma
df['ema'] = ema
df['atr'] = atr
df['cci'] = cci
df['roc'] = roc
df['rsi'] = rsi
df['willr'] = willr
df['fastk'] = fastk
df['fastd'] = fastd'''

'''for column in df:
    if df[column].dtype == np.float64:
        #mean = np.mean(df[column][:BREAKPOINT])
        #std = np.std(df[column][:BREAKPOINT]) #, ddof=1

        #df[column] = (df[column] - mean) / std

        iqr = np.subtract(*np.percentile(df[column][TIMESTEP*2:BREAKPOINT], [75, 25]))
        median = np.median(df[column][TIMESTEP*2:BREAKPOINT])

        df[column] = (df[column] - median) / iqr'''

for column in df:
    if df[column].dtype == np.float64:
        df[column] = df[column].astype(np.float32)
    else:
        df[column] = df[column].astype(np.uint8)

scaler = MinMaxScaler()
scaler.fit(df[df.columns][TIMESTEP*2:BREAKPOINT])
df[df.columns] = scaler.transform(df[df.columns])

df = df[BREAKPOINT:]
df.reset_index(drop=True, inplace=True)

xdata = window(df.to_numpy(), test)
print("x data ready")

del test, df, min_diff, xau_ask_volume, scaler

q = len(xdata)-BREAK
p = int(q*TRAIN_TEST_PERCENTAGE)

#Create a RF Classifier
clf = BalancedRandomForestClassifier(n_estimators=400, max_features=MAX_FEATURES, criterion='gini', n_jobs=-2, bootstrap=True, #len(df2.columns)
                             random_state=0, class_weight='balanced') #class_weight='balanced_subsample, max_depth=200

x_train, x_test, y_train, y_test = xdata[:p], xdata[p:q], ydata[:p], ydata[p:q]

del xdata, ydata

x_train = reshape_vals(x_train)
x_test = reshape_vals(x_test)

print(x_train.shape[0])

print('x_train and x_test have been reshaped')
#x_train, y_train = shuffle(x_train, y_train, random_state=0)

print('training the model')

y_train, y_test = y_train.astype(np.uint8), y_test.astype(np.uint8)
print(x_train.dtype, y_train.dtype)

#Train the model using the training sets
clf.fit(x_train, y_train)

del x_train, y_train

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred), str(WINDOW), str(FORECAST))
print(metrics.classification_report(y_test,y_pred))

y_pred = clf.predict_proba(x_test)
y_pred2 = clf.predict(x_test)
to_write = list()

for x, y, z, a, b in zip(y_pred[0], y_pred[1], y_pred[2], y_pred2, y_test):
    new_write = list()
    for i in x:
        new_write.append(str(i))

    for i in y:
        new_write.append(str(i))

    for i in z:
        new_write.append(str(i))

    for i in a:
        new_write.append(str(i))

    for i in b:
        new_write.append(str(i))

    to_write.append(new_write)

del x_test, y_test

write_file(to_write)

#save_model(clf)

del clf