from __future__ import division

import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import datetime
import locale
locale.setlocale(locale.LC_NUMERIC, "")


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

ASK_FNAME = "data/ASK main.csv"
BID_FNAME = "data/BID main.csv"
WINDOW=120
FORECAST=180
EMB_SIZE=11
STEP=1  #best is 75
TRAIN_TEST_PERCENTAGE=0.9
SAVE_NAME = "classification_model.hdf5"
LOAD_NAME = "classification_model.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV = "classification"
TRAINING = 1
TESTING = 0
NUMBER_EPOCHS = 100
TRADING_DAYS = 5
BREAK = 80000


def df_sample(df):
    print '\nINSPECTING DATABASE..\n'
    print 'DATABASE SIZE [', len(df), ']'

    print 'SAMPLE VALUES..'

    r = int(random.random() * len(df))
    sample = df[r:r + WINDOW]
    target = df[r:r + WINDOW + FORECAST]

    plt.plot(sample.Open)
    plt.plot(sample.High)
    plt.plot(sample.Low)
    plt.plot(target.Close)
    # plt.plot(sample.Volume)
    plt.show()


def df_plot(df):
    print '\nINSPECTING DATABASE..\n'
    print 'DATABASE SIZE [', len(df), ']'

    for col in df.columns:
        if col != 'Datetime' and col != 'Volume_x' and col != 'Volume_y':
            plt.plot(df[col])
    plt.show()


def df_dead(df):
    _is_dead = 0

    print '\nINSPECTING DATABASE..\n'
    print 'DATABASE SIZE [', len(df), ']'

    deads = []

    for i in xrange(len(df.Close.values)):
        if (df.Close.values[i] == df.Open.values[i] == df.High.values[i] == df.Low.values[i]):
            if (_is_dead == 0):
                # print '_is_dead from', i
                deads.append(i)
            _is_dead = 1
        if (df.Close.values[i] != df.Open.values[i]):
            if (_is_dead == 1):
                # print '_is_dead to', i
                # print '-'
                deads.append(i)
            _is_dead = 0

    return deads


def health_check(df):
    flag = 1
    deads = df_dead(df)
    long_deads, i = [], 0

    while i < len(deads) - 1:
        dead_len = deads[i + 1] - deads[i]

        if (dead_len > 100):
            long_deads.append(deads[i])
            long_deads.append(deads[i + 1])
            print 'WARNING long dead period [', deads[i], '] to [', deads[i + 1], ']'
            flag = 0
        i = i + 2

    print('Data from ' + df.Datetime[df.index[0]] + ' to ' + (df.Datetime[df.index[len(df.index) - 1]]))
    print('..check completed')
    if flag:
        print "no DEAD periods"
    return long_deads


def dataset(rtf_input, START=0, END=0, SLICING=0):
    # if(!slicing) the function just download and not clean the whole dataset
    # if slicing the function is slicing and cleaning the dataset from START to END
    print 'dataset preparation.. (cleaning size ', END - START, ')'
    df = pd.read_table(rtf_input, header=8, sep=",")
    if (SLICING):
        df = df[START:END]

    df = df.rename(
        columns={df.columns[0]: 'Datetime', df.columns[1]: 'Open', df.columns[2]: 'High', df.columns[3]: 'Low',
                 df.columns[4]: 'Close', df.columns[5]: 'Volume'})

    def clean_string(s):
        ss = '0'
        for i in range(len(s)):
            if s[i] != '\\':
                ss = ss + s[i]
        return float(ss)

    for i in range(START, END):
        df.loc[i, ('Volume')] = df.loc[i, ('Volume')] #clean_string()
        if i % 1000 == 0:
            print 'cleaning dataset ', i, '/', len(df.Volume)

    def find_hour(df, hour='00:00:00'):
        vec = []
        for i in range(0, len(df.Datetime)):
            if (df.Datetime[i][11:19] == hour):
                weekday = datetime.datetime.strptime(df.loc[i, ('Datetime')], '%d.%m.%Y %H:%M:%S.%f').weekday()
                if (weekday != 5 and weekday != 6):
                    vec.append(i)
        return vec

    if (SLICING == 0):
        STARTS = find_hour(df, '00:00:00')
        ENDS = find_hour(df, '23:59:00')

        return df, STARTS, ENDS

    else:
        print('dataset ready to use')
        return df


def dataset_check(RTF_INPUT=ASK_FNAME):
    # check the overall database health with no cleaning
    df, STARTS, ENDS = dataset(RTF_INPUT)
    health_check(df)
    print('\nDATASET NOT CLEANED\n')
    print(df.head())
    print(df.tail())
    return STARTS, ENDS


# procedure for dataframe (clean dataset) preparation
# _START and _END are the two vector with starting and end values for the slicing

def full_df(STARTS, END, ASK_FNAME=ASK_FNAME, BID_FNAME=BID_FNAME):
    ask_df = []
    bid_df = []
    '''for i in range(0, len(STARTS)):
        ask_df.append(dataset(ASK_FNAME, STARTS[i], ENDS[i], SLICING=0))
        bid_df.append(dataset(BID_FNAME, STARTS[i], ENDS[i], SLICING=0))

    ask_df = pd.concat(ask_df)
    bid_df = pd.concat(bid_df)

    deads = health_check(bid_df)

    df = pd.merge(ask_df, bid_df, on="Datetime")'''

    df = pd.read_csv('data/MAIN.csv')

    '''for i in xrange(len(df)):
        df.loc[i, ('Datetime')] = datetime.datetime.strptime(df.loc[i, ('Datetime')], '%d.%m.%Y %H:%M:%S.%f')
        if i % 1000 == 0:
            print 'parsing time ', i, '/', len(df.Datetime)'''

    print('\nDATASET CLEANED\n')
    print(df.head())
    print(df.tail())
    # df_plot(df)

    return df

STARTS, ENDS = dataset_check(BID_FNAME)
df = full_df(STARTS,ENDS)
#df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S.%f')

def ternary_tensor(Time, aO, aH, aL, aC, aV, bO, bH, bL, bC, bV):
    count = 0
    X, Y = [], []
    for i in range(0, len(Time) - WINDOW - FORECAST, STEP):
        count = count + 1
        #try:
        # ask open, ask high.. bid close, bid volume
        ao = aO[i:i + WINDOW]
        ah = aH[i:i + WINDOW]
        al = aL[i:i + WINDOW]
        ac = aC[i:i + WINDOW]
        av = aV[i:i + WINDOW]
        # zscore on time window interval

        bo = bO[i:i + WINDOW]
        bh = bH[i:i + WINDOW]
        bl = bL[i:i + WINDOW]
        bc = bC[i:i + WINDOW]
        bv = bV[i:i + WINDOW]

        s = ((np.array(ac) - np.array(bc)) - s_mean) / s_std

        # zscore on time window interval
        ao = (np.array(ao) - ao_mean) / ao_std
        ah = (np.array(ah) - ah_mean) / ah_std
        al = (np.array(al) - al_mean) / al_std
        ac = (np.array(ac) - ac_mean) / ac_std
        av = (np.array(av) - av_mean) / av_std
        bo = (np.array(bo) - bo_mean) / bo_std
        bh = (np.array(bh) - bh_mean) / bh_std
        bl = (np.array(bl) - bl_mean) / bl_std
        bc = (np.array(bc) - bc_mean) / bc_std
        bv = (np.array(bv) - bv_mean) / bv_std

        # no action scenario

        bet_ask = aC[i + WINDOW - 1]
        bet_bid = bC[i + WINDOW - 1]
        #bet_time = Time[i + WINDOW]
        prediction_ask = aC[i + WINDOW + FORECAST - 1]
        prediction_bid = bC[i + WINDOW + FORECAST - 1]
        #prediction_time = Time[i + WINDOW + FORECAST]

        #if (bet_time.hour < 25):

        if (prediction_bid > bet_ask):
            # if the bid price at prediction time is greater then ask price at bet time: is a call
            y_i = [1, 0, 0]
        elif (prediction_ask < bet_bid):
            # if the ask price at prediction time is lower then bid price at bet time: is a put
            y_i = [0, 1, 0]
        else:
            y_i = [0, 0, 1]

        x_i = np.column_stack((ao, ah, al, ac, av, bo, bh, bl, bc, bv, s))
        # print y_i,bet_ask,bet_bid,bet_time,prediction_ask,prediction_bid,prediction_time

        X.append(x_i)
        Y.append(y_i)

        if count % 500 == 0:
            print count, 'windows sized'

        #except Exception as e:
            #print e
            #pass

    return X, Y


def format_data(df, ternary=1, binary=0):
    Time = df.Datetime
    aO = df.Open_x.tolist()
    aH = df.High_x.tolist()
    aL = df.Low_x.tolist()
    aC = df.Close_x.tolist()
    aV = df.Volume_x.tolist()
    bO = df.Open_y.tolist()
    bH = df.High_y.tolist()
    bL = df.Low_y.tolist()
    bC = df.Close_y.tolist()
    bV = df.Volume_y.tolist()

    if (ternary == 1):
        return ternary_tensor(Time, aO, aH, aL, aC, aV, bO, bH, bL, bC, bV)

    elif (binary == 1):
        return binary_tensor(Time, aO, aH, aL, aC, aV, bO, bH, bL, bC, bV)

def spread(Y):
    still = 0
    for vec in Y:
        if vec[2]==1:
            still=still+1
    spread =still*100/len(Y)
    print spread,"%"
    return spread

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

def remove_nan_examples(data):
    #some basic util functions
    newX = []
    for i in range(len(data)):
        if np.isnan(data[i]).any() == False:
            newX.append(data[i])
    return newX


def train_test(X, y, percentage=TRAIN_TEST_PERCENTAGE):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]

    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

df['Spread_close'] = df['Close_x'] - df['Close_y']
df2 = df[:BREAK]

ao_mean = np.mean(df2['Open_x'])
ah_mean = np.mean(df2['High_x'])
al_mean = np.mean(df2['Low_x'])
ac_mean = np.mean(df2['Close_x'])
av_mean = np.mean(df2['Volume_x'])
s_mean = np.mean(df2['Spread_close'])

bo_mean = np.mean(df2['Open_y'])
bh_mean = np.mean(df2['High_y'])
bl_mean = np.mean(df2['Low_y'])
bc_mean = np.mean(df2['Close_y'])
bv_mean = np.mean(df2['Volume_y'])

ao_std = np.std(df2['Open_x'])
ah_std = np.std(df2['High_x'])
al_std = np.std(df2['Low_x'])
ac_std = np.std(df2['Close_x'])
av_std = np.std(df2['Volume_x'])
s_std = np.std(df2['Spread_close'])

bo_std = np.std(df2['Open_y'])
bh_std = np.std(df2['High_y'])
bl_std = np.std(df2['Low_y'])
bc_std = np.std(df2['Close_y'])
bv_std = np.std(df2['Volume_y'])

X,Y = format_data(df[BREAK:])
print(Y)

_spread = spread(Y)

X , Y = np.array(X) , np.array(Y)

DAY_SIZE = 780


#MODEL DEFINITION
print 'initializing model..'
model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(3))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=SAVE_NAME, verbose=1, save_best_only=True)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def alg01(pred, alfa, y_test):
    # alg01 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong, = 0, 0,
    actions = []

    for x in xrange(len(pred)):
        if pred[x][0] > alfa:
            action = [1, 0, 0]  # put
            actions.append(action)

            if y_test[x][0] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

        elif pred[x][1] > alfa:
            action = [0, 1, 0]
            actions.append(action)  # call

            if y_test[x][1] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg01 alfa=", alfa, "- RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / len(actions)), "% "

        return ((right * 100) / len(actions))


def alg02(pred, alfa, y_test):
    # alg02 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong = 0, 0
    for i in xrange(len(pred)):
        if pred[i][1] > alfa:
            if y_test[i][1] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg02 alfa=", alfa, "- RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / (right + wrong)), "%"
        return ((right * 100) / (right + wrong))


def alg03(pred, alfa, y_test):
    # alg03 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong = 0, 0
    for i in xrange(len(pred)):
        if pred[i][0] > alfa:

            if y_test[i][0] == 1:
                right = right + 1
                # print pred[i],y_test[i],"RIGHT"
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg03 alfa=", alfa, "- RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / (right + wrong)), "%"
        return ((right * 100) / (right + wrong))


def alg04(pred, delta, gamma, y_test):
    actions = []
    right, wrong = 0, 0
    for x in xrange(len(pred)):
        # print pred[x]
        deltap = pred[x][0] - pred[x][1]

        if (abs(deltap) > delta):
            if pred[x][0] > pred[x][1] and pred[x][2] < gamma:
                actions.append([1, 0])
            elif pred[x][0] < pred[x][1] and pred[x][2] < gamma:
                actions.append([0, 1])
        try:
            if actions[len(actions) - 1][0] == y_train[x][0] == 1 or actions[len(actions) - 1][1] == y_train[x][1] == 1:
                # print actions[len(actions)-1],y_train[x],"  GIUSTO"
                right = right + 1
            else:
                # print actions[len(actions)-1],y_train[x],"  SBAGLIATO"
                wrong = wrong + 1
        except:
            pass

    print "\nalg04 delta=", delta, " gamma=", gamma, "- RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                (right * 100) / (right + wrong)), "%"
    return ((right * 100) / (right + wrong))


def alg05(pred, y_test):
    # alg03 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong = 0, 0
    for i in xrange(len(pred)):
        if pred[i][0] > pred[i][1] + pred[i][2]:
            # call
            if y_test[i][0] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

        if pred[i][1] > pred[i][0] + pred[i][2]:
            # put
            if y_test[i][1] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg05 - RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / (right + wrong)), "%"
        return ((right * 100) / (right + wrong))


def alg06(pred, y_test):
    # alg03 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong = 0, 0
    for i in xrange(len(pred)):
        if pred[i][0] > pred[i][1] + pred[i][2]:
            # call
            if y_test[i][0] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg06 - RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / (right + wrong)), "%"
        return ((right * 100) / (right + wrong))


def alg07(pred, y_test):
    # alg03 trading algorithm, returns the right bets percentage with alfa parameters
    right, wrong = 0, 0
    for i in xrange(len(pred)):
        if pred[i][1] > pred[i][0] + pred[i][2]:
            # put
            if y_test[i][1] == 1:
                right = right + 1
            else:
                wrong = wrong + 1

    if (right != 0):
        print "alg07 - RIGHT: ", right, "| WRONG: ", wrong, "| RIGHT PERCENTAGE:", (
                    (right * 100) / (right + wrong)), "%"
        return ((right * 100) / (right + wrong))

def alg08(pred,alfa, y_test):
    right,wrong, calls, puts = 0,0,0,0
    for i in xrange(len(pred)):
        if pred[i][0]-pred[i][2]>alfa:
            #call
            calls += 1
            if y_test[i][0]==1:
                right=right+1
            else:
                wrong=wrong+1
        elif pred[i][1]-pred[i][2]>alfa:
            #put
            puts += 1
            if y_test[i][1]==1:
                right=right+1
            else:
                wrong=wrong+1
    if(right!=0):
        print "alg08 ",alfa," - RIGHT: ", right,"| WRONG: ", wrong, "| RIGHT PERCENTAGE:", ((right*100)/(right+wrong)),"%", "| CALLS:", calls, "| PUTS:", puts
        return ((right*100)/(right+wrong))

def alg09(pred,alfa, y_test):
    right,wrong = 0,0
    for i in xrange(len(pred)):
        if pred[i][1]-pred[i][2]>alfa:
            #put
            if y_test[i][1]==1:
                right=right+1
            else:
                wrong=wrong+1
        elif pred[i][0]-pred[i][2]>alfa:
            #put
            if y_test[i][0]==1:
                right=right+1
            else:
                wrong=wrong+1
    if(right!=0):
        print "alg09 ",alfa," - RIGHT: ", right,"| WRONG: ", wrong, "| RIGHT PERCENTAGE:", ((right*100)/(right+wrong)),"%"
        return ((right*100)/(right+wrong))


def iter_alg01(pred, y_test):
    # iterate alg01 returing an array with the percentages varying alfa
    _perc = []
    for i in range(33, 100):
        _perc.append(alg01(pred, i * 0.01, y_test))
    return _perc


def iter_alg02(pred, y_test):
    # iterate alg02 returing an array with the percentages varying alfa
    _perc = []
    for i in range(33, 100):
        _perc.append(alg02(pred, i * 0.01, y_test))
    return _perc


def iter_alg03(pred, y_test):
    # iterate alg02 returing an array with the percentages varying alfa
    _perc = []
    for i in range(33, 100):
        _perc.append(alg03(pred, i * 0.01, y_test))
    return _perc


def iter_alg04(pred, y_test):
    _perc = []
    for delta in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 30, 40, 50, 60, 70, 80, 90, 100]:
        for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5]:
            try:
                _perc.append(alg04(pred, delta * 0.01, gamma, y_test))
            except:
                pass
    return _perc


def iter_alg05(pred, y_test):
    _perc = []
    val = alg05(pred, y_test)
    for i in xrange(100):
        _perc.append(val)
    return _perc


def iter_alg06(pred, y_test):
    _perc = []
    val = alg06(pred, y_test)
    for i in xrange(100):
        _perc.append(val)
    return _perc


def iter_alg07(pred, y_test):
    _perc = []
    val = alg07(pred, y_test)
    for i in xrange(100):
        _perc.append(val)
    return _perc


def iter_alg08(pred, y_test):
    # iterate alg02 returing an array with the percentages varying alfa
    _perc = []
    for i in range(33, 100):
        _perc.append(alg08(pred, i * 0.01, y_test))
    return _perc


def iter_alg09(pred, y_test):
    # iterate alg02 returing an array with the percentages varying alfa
    _perc = []
    for i in range(33, 100):
        _perc.append(alg09(pred, i * 0.01, y_test))
    return _perc


def algo_rating(pred, y_test, a, b):
    maxx = [0, 0]
    for i in range(a, b):
        out = alg01(pred, i * 0.01, y_test)
        try:
            if (out > maxx[0]):
                maxx[0] = out
                maxx[1] = i
        except:
            pass
    alg01_max = maxx

    maxx = [0, 0]
    for i in range(a, b):
        out = alg02(pred, i * 0.01, y_test)
        if (out > maxx[0]):
            maxx[0] = out
            maxx[1] = i
    alg02_max = maxx

    maxx = [0, 0]
    for i in range(a, b):
        out = alg03(pred, i * 0.01, y_test)
        try:
            if (out > maxx[0]):
                maxx[0] = out
                maxx[1] = i
        except:
            pass
    alg03_max = maxx

    alg05_max = [alg05(pred, y_test), 0]

    alg06_max = [alg06(pred, y_test), 0]

    alg07_max = [alg07(pred, y_test), 0]

    maxx = [0, 0]
    for i in range(a, b):
        out = alg08(pred, i * 0.01, y_test)
        try:
            if (out > maxx[0]):
                maxx[0] = out
                maxx[1] = i
        except:
            pass
    alg08_max = maxx

    maxx = [0, 0]
    for i in range(a, b):
        out = alg09(pred, i * 0.01, y_test)
        try:
            if (out > maxx[0]):
                maxx[0] = out
                maxx[1] = i
        except:
            pass
    alg09_max = maxx

    return alg01_max, alg02_max, alg03_max, alg05_max, alg06_max, alg07_max, alg08_max, alg08_max

def algo_summary(n):
    average = 0
    for i in xrange(len(rating)):
        average = average + rating[i][n][0]
    average = (average)/18
    print average,"% for alg0",n+1


model.load_weights(SAVE_NAME)
'''for i in xrange(TRADING_DAYS):
    pred = model.predict(np.array(X[-DAY_SIZE * i:-(DAY_SIZE * (i - 1)) - 1]))
    y = Y[-DAY_SIZE * i:-(DAY_SIZE * (i - 1)) - 1]

    algo_rating(pred, y_test, 50, 51)'''

#for i in [0,1,2,6,7]:
#    algo_summary(i)

model.load_weights(SAVE_NAME)
alg01_40, alg03_40 = [], []
alg01_50, alg03_50 = [], []
alg01_55, alg03_55 = [], []
alg09_50, alg09_55 = [], []
for i in xrange(1, TRADING_DAYS + 1):
    print(i)
    pred = model.predict(np.array(X[-DAY_SIZE * i:-(DAY_SIZE * (i - 1)) - 1]))
    y = Y[-DAY_SIZE * i:-(DAY_SIZE * (i - 1)) - 1]

    '''palg01 = plt.plot(iter_alg01(pred, y), label="alg01")
    palg02 = plt.plot(iter_alg02(pred, y), label="alg02")
    palg03 = plt.plot(iter_alg03(pred, y), label="alg03")
    palg04 = plt.plot(iter_alg04(pred, y), label="alg04")
    palg05 = plt.plot(iter_alg05(pred, y), label="alg05")
    palg06 = plt.plot(iter_alg06(pred, y), label="alg06")
    palg07 = plt.plot(iter_alg07(pred, y), label="alg07")'''
    #palg08 = plt.plot(iter_alg08(pred, y), label="alg08")
    palg09 = plt.plot(iter_alg09(pred, y), label="alg09")

    '''alg01_40.append(alg01(pred, 0.40, y))
    alg03_40.append(alg03(pred, 0.40, y))
    alg01_50.append(alg01(pred, 0.50, y))
    alg03_50.append(alg03(pred, 0.50, y))
    alg01_55.append(alg01(pred, 0.55, y))
    alg03_55.append(alg03(pred, 0.55, y))
    alg09_50.append(alg09(pred, 0.50, y))
    alg09_50.append(alg09(pred, 0.55, y))'''

    plt.legend(title="spread:" + str(_spread) + "%")
    plt.show()


#rating