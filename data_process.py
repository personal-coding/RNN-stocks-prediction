#This code helps line up rows of data by time. I would generally remove rows of data where EUR/USD had zero volume,
# assuming that this generally meant the data was a weekend or holiday.

import pandas as pd, sys

bid_ask = ['ASK', 'BID']
tickers = ['EUR/JPY', 'EUR/GBP', 'EUR/CAD', 'EUR/CHF', 'XAU/USD', 'XAG/USD', 'EUR/HKD', 'USD/CNH', 'EUR/SGD']
#tickers = ['EUR/TRY', 'EUR/SEK', 'EUR/RUB', 'EUR/PLN', 'EUR/NOK', 'EUR/CZK', 'EUR/DKK']

MAIN_CSV = r'C:\\Users\\russi\Downloads/EURUSD_Candlestick_1_M_ASK_27.06.2020-04.07.2020.csv'
DATA_CSV = r'C:\\Users\\russi\Downloads/EURUSD_Candlestick_1_M_BID_27.06.2020-04.07.2020.csv'

df = pd.read_csv(MAIN_CSV)
df2 = pd.read_csv(DATA_CSV)

#Match rows where local time is the same. I would delete zero volume data rows from EUR/USD first.
filter = df2['Local time'].isin(df['Local time'])

df2[filter == True].to_csv(DATA_CSV)

del df

DATA_CSV = r'C:\\Users\\russi\Downloads/EURUSD_Candlestick_1_M_BID_27.06.2020-04.07.2020.csv'
df2 = pd.read_csv(DATA_CSV)
df2[filter == True].to_csv(DATA_CSV)

DATA_CSV = 'C:\\Users\\russi\Downloads/{0}_Candlestick_1_M_{1}_27.06.2020-04.07.2020.csv'

for b_a in bid_ask:
    for ticker in tickers:
        df2 = pd.read_csv(DATA_CSV.format(ticker.replace('/',''), b_a))
        df2.reset_index(drop=True, inplace=True)
        df2[filter == True].to_csv(DATA_CSV.format(ticker.replace('/',''), b_a))