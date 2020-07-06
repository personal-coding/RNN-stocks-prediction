from gevent.pool import Pool
from gevent import monkey; monkey.patch_all()
from hmmlearn.hmm import GaussianHMM
import requests, calendar, time, codecs, numpy as np
import pandas as pd
import _pickle as cPickle
import talib

BASE_URL = 'https://freeserv.dukascopy.com/2.0/index.php?path=chart%2Fjson3&instrument={0}&offer_side={1}&interval=1MIN' \
           '&splits=true&limit={2}&time_direction=P&timestamp={3}&jsonp="'

#This reshapes the x-data into a 2d array
def reshape_vals(df):
    nsamples, nx, ny = df.shape
    return df.reshape((nsamples, nx * ny))

def response(site):
    qq = requests.get(site, headers=headers, stream=True)
    return qq.text

def process(site,ticker,b_a):
    global l
    response_body = response(site)[2:-2]
    response_list = eval(response_body)
    l[ticker + '_' + b_a] = np.array(response_list[::-1])

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Referer': 'https://demo-login.dukascopy.com/binary/',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

bid_ask = ['A', 'B']
tickers = ['EUR/USD', 'EUR/JPY', 'EUR/GBP', 'EUR/CAD', 'EUR/CHF', 'XAU/USD', 'EUR/HKD', 'USD/CNH', 'EUR/SGD']

num_worker_threads = len(bid_ask) * len(tickers)
pool = Pool(num_worker_threads)

limit = 180 + 90
last = 1591651080000
most_recent = last
BREAK = 840000
TIMESTEP = 30
csv_filepath = 'data/MAIN_1min_onlyFX_min_diff_more_data - Copy Copy.csv'
rf_0_pickle_path = 'rf_model_0.pickle'
rf_1_pickle_path = 'rf_model_1.pickle'
rf_2_pickle_path = 'rf_model_2.pickle'
rf_3_pickle_path = 'rf_model_3.pickle'

df = pd.read_csv(csv_filepath)[:BREAK]
df = df.drop(columns=['Datetime', 'Min Diff','Hour'], axis=1)
means = []
stds = []

for column in df:
    means.append(np.mean(df[column][:BREAK]))
    stds.append(np.std(df[column][:BREAK]))

while last == most_recent:
    now = calendar.timegm(time.gmtime())
    l = {}

    for b_a in bid_ask:
        for ticker in tickers:
            pool.apply_async(process, args=(
            BASE_URL.format(ticker,
               b_a,
               str(limit),
               str(now) + '000'),ticker,b_a))
    pool.join()

    #print(l)
    most_recent = l[list(l)[0]][-1][0]
    if not all(most_recent == l[value][-1][0] for value in l):
        most_recent = last
        df2 = pd.DataFrame()

        for b_a in bid_ask:
            for ticker in tickers:
                high, low, close, open, volume = l[ticker + '_' + b_a][:, 2], l[ticker + '_' + b_a][:, 3],  \
                                                 l[ticker + '_' + b_a][:, 4], l[ticker + '_' + b_a][:, 1], \
                                                 l[ticker + '_' + b_a][:, 5]

                df2[ticker + b_a + 'midprice'] = talib.MIDPRICE(high, low, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'volume'] = volume[TIMESTEP*3:]
                df2[ticker + b_a + 'natr'] = talib.NATR(high, low, close, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'bop'] = talib.BOP(open, high, low, close)[TIMESTEP*3:]
                df2[ticker + b_a + 'adxr'] = talib.ADXR(high, low, close, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'stddev'] = talib.STDDEV(close, timeperiod=TIMESTEP, nbdev=1)[TIMESTEP*3:]
                df2[ticker + b_a + 'correl'] = talib.CORREL(high, low, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)[TIMESTEP*3:]

                print(talib.HT_TRENDMODE(close))

                del high, low, close, open, volume

        print(df2)
    elif most_recent != last:
        df2 = pd.DataFrame()

        for b_a in bid_ask:
            for ticker in tickers:
                high, low, close, open, volume = l[ticker + '_' + b_a][:, 2], l[ticker + '_' + b_a][:, 3], \
                                                 l[ticker + '_' + b_a][:, 4], l[ticker + '_' + b_a][:, 1], \
                                                 l[ticker + '_' + b_a][:, 5]

                df2[ticker + b_a + 'midprice'] = talib.MIDPRICE(high, low, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'volume'] = volume[TIMESTEP*3:]
                df2[ticker + b_a + 'natr'] = talib.NATR(high, low, close, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'bop'] = talib.BOP(open, high, low, close)[TIMESTEP*3:]
                df2[ticker + b_a + 'adxr'] = talib.ADXR(high, low, close, timeperiod=TIMESTEP)[TIMESTEP*3:]

                print(talib.ADXR(high, low, close, timeperiod=TIMESTEP))

                df2[ticker + b_a + 'stddev'] = talib.STDDEV(close, timeperiod=TIMESTEP, nbdev=1)[ TIMESTEP*3:]
                df2[ticker + b_a + 'correl'] = talib.CORREL(high, low, timeperiod=TIMESTEP)[TIMESTEP*3:]
                df2[ticker + b_a + 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)[TIMESTEP*3:]

                #print(talib.ADXR(high, low, close, timeperiod=TIMESTEP))

                del high, low, close, open, volume

        for column in df2:
            print(df2[column])

        A = []
        df2 = df2.to_numpy()

        A.append(df2.tolist())
        df2 = np.array(A)
        df2 = reshape_vals(df2)

        #print(df2)

        last = most_recent

    time.sleep(.5)