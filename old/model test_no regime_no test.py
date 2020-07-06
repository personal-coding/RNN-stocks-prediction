import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys, codecs
import numpy as np
import _pickle as cPickle
pd.options.mode.chained_assignment = None

BREAKPOINT=840000
WINDOW=180
FORECAST=360
TRAIN_TEST_PERCENTAGE=1
MAIN_FILE = 'data/MAIN_1min_onlyFX_min_diff_more_data.csv'
SAVE_PATH = 'model_1min_onlyFX_no_regime_more_data_balanced.pickle'
regimes = [0, 1, 2, 3]
DIFF = .00000

def write_file(hold):
    with codecs.open('results_rf_1min_onlyFX_no_regime.csv', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

def save_model(rf):
    with codecs.open(SAVE_PATH, mode='wb') as f:
        cPickle.dump(rf, f, -1)

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

#Create a RF Classifier
clf = BalancedRandomForestClassifier(n_estimators=400, max_features=5, criterion='gini', n_jobs=-2, bootstrap=True, #len(df2.columns)
                             random_state=0, class_weight='balanced') #class_weight='balanced_subsample, max_depth=200

x_train, y_train = xdata[:p], ydata[:p]

del xdata
del ydata

x_train = reshape_vals(x_train)

print(x_train.shape[0])

print('x_train and x_test have been reshaped')
x_train, y_train = shuffle(x_train, y_train, random_state=0)

print('training the model')

y_train = y_train.astype(np.uint8)
print(x_train.dtype, y_train.dtype)

print('fitting the model')

#Train the model using the training sets
clf.fit(x_train, y_train)

del x_train
del y_train

save_model(clf)

del clf