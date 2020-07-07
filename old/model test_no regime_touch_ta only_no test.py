import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys, codecs, talib
import numpy as np
import _pickle as cPickle
pd.options.mode.chained_assignment = None

MAX_FEATURES = 5
rng = np.random.RandomState(42)
BREAKPOINT=840000
BREAK=0
PAIR_ASK = 'EURUSD Ask'
PAIR_BID = 'EURUSD Bid'
WINDOW=int(60*3)
TIMESTEP=int(30)
FORECAST=int(60*6)
TRAIN_TEST_PERCENTAGE=1
MAIN_FILE = 'data/MAIN_1min_onlyFX_min_diff_more_data - Copy Copy.csv'
SAVE_PATH = 'model_1min_onlyFX_no_regime_more_data_balanced.pickle'
regimes = [0, 1, 2, 3]
DIFF = 0.00000

bid_ask = ['Ask', 'Bid']
tickers = ['EURUSD', 'EURJPY', 'EURGBP', 'EURCAD', 'EURCHF', 'XAU', 'XAG', 'EURHKD', 'USDCNH', 'EURSGD']

#Write the Excel file for testing
def write_file(hold):
    with codecs.open('results_rf_1min_onlyFX_no_regime_min_diff_balanced.csv', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

#This saves the model into a pickle, uses -1 to reduce the final file
def save_model(rf):
    with codecs.open(SAVE_PATH, mode='wb') as f:
        cPickle.dump(rf, f, -1)

#This creates the y-data
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

#This creates the x-data
def window(df, test):
    x_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if i % 10000 == 0:
            print('x', i)

        if not test[i]:
            x_i.append(df[i - WINDOW + 1:i + 1])

    x_i = np.array(x_i)
    return(x_i)

#This reshapes the x-data into a 2d array
def reshape_vals(df):
    nsamples, nx, ny = df.shape
    return df.reshape((nsamples, nx * ny))

#Load dataset
df = pd.read_csv(MAIN_FILE, index_col=False)
df = df.drop(columns=['Datetime'], axis=1)

#df['Min Diff'] = False

#Make sure we don't include data that spans between weekends - we can't trade these times and they appear to be highly volatile
min_diff = df['Min Diff'][BREAKPOINT:]
min_diff.reset_index(drop=True, inplace=True)

#Make sure we don't include data where gold is not being traded (i.e. only at 2-3PM PST)
xau_ask_volume = df['XAU Ask Volume'][BREAKPOINT:].eq(0) & df['Hour'][BREAKPOINT:].isin([14])
xau_ask_volume.reset_index(drop=True, inplace=True)

df = df.drop(columns=['Min Diff','Hour'], axis=1)

#This is so we can combine the two checks, and so we don't have to check twice (i.e. once for y-data and one for x-data)
df['test'] = True
test = df['test'][BREAKPOINT:]
test.reset_index(drop=True, inplace=True)

for i in range(WINDOW, len(df[BREAKPOINT:]) - FORECAST):
    if not min_diff[i - WINDOW + 1:i + 1].any() and not xau_ask_volume[i - WINDOW + 1:i + 1].any():
        test[i] = False

df = df.drop(columns=['test'], axis=1)

#Create the y-data
ydata = create_targets(df[[PAIR_ASK,PAIR_BID]][BREAKPOINT:], test)
print("y data ready")

#Create the technical indicators, only using three to limit the dataset - MOM has too many NaNs
##Drop the original data after the indicators are created

MAX = 0
MAX_TEST = 0

for b_a in bid_ask:
    for ticker in tickers:
        high, low, close, open, volume = df[ticker + ' ' + b_a + ' High'],df[ticker + ' ' + b_a + ' Low'], \
                                      df[ticker + ' ' + b_a], df[ticker + ' ' + b_a + ' Open'], \
                                      df[ticker + ' ' + b_a + ' Volume']

        # overlap ind
        df[ticker + b_a + 'midprice'] = talib.MIDPRICE(high, low, timeperiod=TIMESTEP)

        # volume ind
        #ad = talib.AD(high, low, close, volume)
        # adosc = talib.ADOSC(high, low, close, volume, fastperiod=int(TIMESTEP/3), slowperiod=TIMESTEP)
        df[ticker + b_a + 'volume'] = volume

        # volatitility ind
        # df[ticker + b_a + 'adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=int(TIMESTEP/3), slowperiod=TIMESTEP)
        #df[ticker + b_a + 'ad'] = talib.AD(high, low, close, volume)
        df[ticker + b_a + 'natr'] = talib.NATR(high, low, close, timeperiod=TIMESTEP)

        # momentum ind
        #df[ticker + b_a + 'mfi'] = talib.MFI(high, low, close, volume, timeperiod=TIMESTEP)
        df[ticker + b_a + 'bop'] = talib.BOP(open, high, low, close)
        df[ticker + b_a + 'adxr'] = talib.ADXR(high, low, close, timeperiod=TIMESTEP)
        # df[ticker + b_a + 'mom'] = talib.MOM(close, timeperiod=TIMESTEP)

        # stats ind
        df[ticker + b_a + 'stddev'] = talib.STDDEV(close, timeperiod=TIMESTEP, nbdev=1)
        df[ticker + b_a + 'correl'] = talib.CORREL(high, low, timeperiod=TIMESTEP)

        # cycle ind
        df[ticker + b_a + 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        # price ind
        #df[ticker + b_a + 'WCLPRICE'] = talib.WCLPRICE(high, low, close)

        df = df.drop(columns=[ticker + ' ' + b_a + ' High', ticker + ' ' + b_a + ' Low', ticker + ' ' + b_a + ' Open',
                              ticker + ' ' + b_a, ticker + ' ' + b_a + ' Volume'], axis=1)

        MAX_TEST = max(
            df[ticker + b_a + 'natr'].isnull().sum(),
            df[ticker + b_a + 'adxr'].isnull().sum(),
            df[ticker + b_a + 'midprice'].isnull().sum(),
            #df[ticker + b_a + 'WCLPRICE'].isnull().sum(),
            df[ticker + b_a + 'HT_TRENDMODE'].isnull().sum(),
            df[ticker + b_a + 'correl'].isnull().sum(),
            df[ticker + b_a + 'stddev'].isnull().sum(),
            df[ticker + b_a + 'bop'].isnull().sum(),)
            #df[ticker + b_a + 'ad'].isnull().sum(),
            #df[ticker + b_a + 'adosc'].isnull().sum()
            #df[ticker + b_a + 'mom'].isnull().sum()
            #df[ticker + b_a + 'mfi'].isnull().sum(),

        if MAX_TEST > MAX: MAX = MAX_TEST

        del high, low, close, open, volume

'''#Normalize the data by subtracting the median and divide by the quartile range
##This is rather than subtracting the mean and dividing by the standard deviation - this helps prevent outliers from impacting the normalization
for column in df:
    if df[column].dtype == np.float64:
        #mean = np.mean(df[column][:BREAKPOINT])
        #std = np.std(df[column][:BREAKPOINT]) #, ddof=1

        #df[column] = (df[column] - mean) / std

        iqr = np.subtract(*np.percentile(df[column][MAX:BREAKPOINT], [75, 25]))
        median = np.median(df[column][MAX:BREAKPOINT])

        df[column] = (df[column] - median) / iqr'''

scaler = MinMaxScaler()
scaler.fit(df[df.columns][MAX:BREAKPOINT])
df[df.columns] = scaler.transform(df[df.columns])

#Reduce the memory usage by each column
for column in df:
    if df[column].dtype == np.float64:
        df[column] = df[column].astype(np.float32)
    else:
        df[column] = df[column].astype(np.uint8)

#Break the the data into a smaller subset to test and train, using a longer dataset to create the normalization
df = df[BREAKPOINT:]
df.reset_index(drop=True, inplace=True)

#Create the x-data
xdata = window(df.to_numpy(), test)
print("x data ready")

del test, df, min_diff, xau_ask_volume, scaler

q = len(xdata)-BREAK
p = int(q*TRAIN_TEST_PERCENTAGE)

#Create a RF Classifier
clf = RandomForestClassifier(n_estimators=400, max_features=MAX_FEATURES, criterion='gini', n_jobs=-2, bootstrap=True, #len(df2.columns)
                             random_state=0, class_weight='balanced') #class_weight='balanced_subsample, max_depth=200

#clf = make_pipeline(FunctionSampler(func=outlier_rejection), clf)

#Break into training set and testing set
x_train, y_train = xdata[:p], ydata[:p]

del xdata, ydata

#Reshape the x-data into a 2d array
x_train = reshape_vals(x_train)

print(x_train.shape[0])

print('x_train and x_test have been reshaped')
#x_train, y_train = shuffle(x_train, y_train, random_state=0)

print('training the model')

#Reduce the y-data memory usage
y_train = y_train.astype(np.uint8)
print(x_train.dtype, y_train.dtype)

#Train the model using the training sets
clf.fit(x_train, y_train)

del x_train, y_train

save_model(clf)

del clf