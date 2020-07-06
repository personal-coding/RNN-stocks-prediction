import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import sys, codecs
import numpy as np
import _pickle as cPickle
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None

MINUS=1440*0
BREAKPOINT=50000# - 28669
WINDOW=180
FORECAST=120
TRAIN_TEST_PERCENTAGE=0.9
MAIN_FILE = 'data/MAIN_1min_onlyFX.csv'
SAVE_PATH = 'rf_model_1min_onlyFX_minus_{0}.pickle'
regimes = [2]
DIFF = .00000

def write_file(hold, regime):
    with codecs.open('results_rf_1min_onlyFX_{0}.csv'.format(str(regime)), mode='w', encoding='utf-8') as f:
        f.write('\n'.join(','.join(e) for e in hold))

def save_model(rf, regime):
    with codecs.open(SAVE_PATH.format(str(regime)), mode='wb') as f:
        cPickle.dump(rf, f)

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

def create_targets(df, regimes, df_x):
    df.reset_index(drop=True, inplace=True)
    y_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if df_x[i] in regimes:
            if df['EURUSD Bid'][i + FORECAST] > df['EURUSD Ask'][i]:
                y_b = [1, 0, 0]
            elif df['EURUSD Ask'][i + FORECAST] < df['EURUSD Bid'][i]:
                y_b = [0, 1, 0]
            else:
                y_b = [0, 0, 1]

            y_i.append(np.array(y_b))
    y_i = np.array(y_i)
    return y_i

def window(df, regimes, df_x):
    x_i = []
    for i in range(WINDOW, len(df)-FORECAST):
        if df_x[i] in regimes:
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

regime_column = df['X_regime_0'][BREAKPOINT-MINUS:]
regime_column.reset_index(drop=True, inplace=True)

df = df.drop(columns=['X_regime_0', 'Y_regime_0'], axis=1)

ydata = create_targets(df[BREAKPOINT-MINUS:], regimes, regime_column)
print("y data ready")

for column in df:
    #if not 'regime' in column:
    #df[column] = np.log(df[column])

    mean = np.mean(df[column][:BREAKPOINT-MINUS])
    std = np.std(df[column][:BREAKPOINT-MINUS]) #, ddof=1

    df[column] = (df[column] - mean) / std

for column in df:
    if df[column].dtype == np.float64:
        df[column] = df[column].astype(np.float32)
    else:
        df[column] = df[column].astype(np.int8)

'''pca = PCA(.95) #n_components
pca = pca.fit(df[:BREAKPOINT-MINUS])'''

df = df[BREAKPOINT-MINUS:]
df.reset_index(drop=True, inplace=True)

'''principalComponents = pca.transform(df)
df = pd.DataFrame(data=principalComponents)'''

xdata = df
xdata = window(xdata, regimes, regime_column)
print("x data ready")

del df

for regime in regimes:
    '''rows = regime_column[WINDOW:len(df) - FORECAST] == regime

    df_xdata = xdata[rows]
    df_ydata = ydata[rows]'''
    df_xdata = xdata[:-MINUS if MINUS > 0 else None]
    df_ydata = ydata[:-MINUS if MINUS > 0 else None]

    #df2 = df[WINDOW:len(df) - FORECAST][rows]

    q = len(df_xdata)
    p = int(q*TRAIN_TEST_PERCENTAGE)

    #Create a RF Classifier
    clf = RandomForestClassifier(n_estimators=400, max_features=5, criterion='gini', n_jobs=-2, bootstrap=True, #len(df_xdata[0][0])
                                 random_state=0, class_weight='balanced_subsample') #, class_weight='balanced_subsample'

    x_train, x_test, y_train, y_test = df_xdata[:p], df_xdata[p:q], df_ydata[:p], df_ydata[p:q]

    del df_xdata
    del df_ydata

    x_train = reshape_vals(x_train)
    x_test = reshape_vals(x_test)

    x_train, y_train = shuffle_in_unison(x_train, y_train)

    #Train the model using the training sets
    clf.fit(x_train, y_train)

    del x_train
    del y_train

    #Predict the response for test dataset
    y_pred = clf.predict(x_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred),'Regime:', str(regime), str(WINDOW), str(FORECAST))
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

    write_file(to_write, regime)

    #save_model(clf, regime)

del clf