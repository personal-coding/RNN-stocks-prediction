# regime_hmm_train.py
# C:\Users\russi\AppData\Local\Programs\Python\Python36

from __future__ import print_function

import datetime
import _pickle as pickle
import warnings

from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np
import pandas as pd
import seaborn as sns
import codecs

def write_file(hold, column, n, break_val, norm):
    with codecs.open('hmm_1min/{0}_{1}_{2}_{3}_GaussianHMM.csv'.format(str(column), str(n), str(break_val), str(norm)), mode='w', encoding='utf-8') as f:
        f.write('\n'.join(str(e) for e in hold))

def obtain_prices_df(csv_filepath, end_date, norm, column, break_val):
    """
    Obtain the prices DataFrame from the CSV file,
    filter by the end date and calculate the
    percentage returns.
    """
    df = pd.read_csv(
        csv_filepath,
        index_col=False,
        parse_dates=False
    )
    if norm: df[column] = (df[column] - np.mean(df[column][:break_val])) / np.std(df[column][:break_val])
    df["Returns"] = df[column]#.pct_change()*1
    #df = df
    df.dropna(inplace=True)
    return df


def plot_in_sample_hidden_states(hmm_model, df, column, n, break_val, norm):
    """
    Plot the adjusted closing prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets)
    write_file(hidden_states, column, n, break_val, norm)


if __name__ == "__main__":
    # Hides deprecation warnings for sklearn
    warnings.filterwarnings("ignore")

    csv_filepath = "data/MAIN_1min_onlyFX.csv"
    pickle_path = 'hmm_model_1min/model_{0}_{1}_{2}_{3}_GaussianHMM.pkl'
    norm = True
    iter = [4]
    columns = ['EURUSD Ask', 'EURUSD Bid']
    breaks = np.arange(20000, 25000, 5000).tolist()
    breaks = np.append(breaks, None)
    end_date = datetime.datetime(2004, 12, 31)

    for n in iter:
        for column in columns:
            for break_val in breaks:
                spy = obtain_prices_df(csv_filepath, end_date, norm, column, break_val)
                rets = np.column_stack([spy["Returns"]])

                hmm_model = GaussianHMM(
                    n_components=n, covariance_type="full", n_iter=10000000
                ).fit(rets)
                print("Model Score:", hmm_model.score(rets), column, n, break_val, norm)

                # Plot the in sample hidden states closing values
                plot_in_sample_hidden_states(hmm_model, spy, column, n, break_val, norm)

                print("Pickling HMM model...")
                pickle.dump(hmm_model, open(pickle_path.format(str(column), str(n), str(break_val), str(norm)), "wb"))
                print("...HMM model pickled.")

    norm = False

    for n in iter:
        for column in columns:
            spy = obtain_prices_df(csv_filepath, end_date, norm, column, None)
            rets = np.column_stack([spy["Returns"]])

            hmm_model = GaussianHMM(
                n_components=n, covariance_type="full", n_iter=10000000
            ).fit(rets)
            print("Model Score:", hmm_model.score(rets), column, n, None, norm)

            # Plot the in sample hidden states closing values
            plot_in_sample_hidden_states(hmm_model, spy, column, n, None, norm)

            print("Pickling HMM model...")
            pickle.dump(hmm_model, open(pickle_path.format(str(column), str(n), str(None), str(norm)), "wb"))
            print("...HMM model pickled.")