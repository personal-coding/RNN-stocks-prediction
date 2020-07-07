# Trying to predict EUR/USD binary and touch options

The base code started from [https://github.com/SolbiatiAlessandro/RNN-stocks-prediction](https://github.com/SolbiatiAlessandro/RNN-stocks-prediction). SolbiatiAlessandro tried to predict EUR/USD binary options on a Dukascopy demo account. The prediction was created using a RNN based on EUR/USD ask & bid open, high, low, close, volume and spread prices (non-transformed).

His code scraped the online demo account using Selenium every minute to extract the live data. Also, Selenium is used to click the buy call and buy put buttons in order to execute trades. The issue I found with scraping the webpage for live data is that 1) many times the data would skip to the next minute, so inaccurate data was scraped for certain minutes 2) the ending minute data shown on the online demo account generally didn't match up with the actual data available from Dukascopy (i.e. if you refreshed the page's data, you'd see different data loaded for historical minutes) and 3) a user would have to wait 90 minutes to grab sufficient data in order to run the model. Even with these issues, I was still able to generate a healthy profit using his model that was created 3 years ago (~$2.2k profit in three days - however, this is of course pure luck).

# Yuri updates and tests

1. The biggest change I made was pulling live data from Dukascopy at the end of each minute (see the [latest framework](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/blob/master/v.0.5.5-framework%20-%20RF.py)).
    - By doing this, I had reliable live data and could execute a trade within 1-2 seconds after a minute had finished.

2. Tested ten instruments ```['EUR/USD', 'EUR/JPY', 'EUR/GBP', 'EUR/CAD', 'EUR/CHF', 'XAU/USD', 'XAG/USD', 'EUR/HKD', 'USD/CNH', 'EUR/SGD'] 'XAU/USD' is spot gold and  'XAG/USD' is spot silver``` with a random forest model and a separate LSTM model.
    - Sstarted by training on minute data from April 1, 2020 to present, then testing live for a week. After the week was up, I'd add the data to the dataset, and re-train the model and test another week. April 1 was used as a cutoff to avoid including the COVID volatile activity in March. Generally, I tested on three hours of data and forecasted six hours ahead. I found that forecasting at least 4 hours ahead lead to increasing accuracy. Also, I tested various alpha thresholds and found that 55% through 60% predicted by the models was encouraging. Below this range and the models were too inaccurate, and above this range, the models traded too infrequently.
    - The first tests were using non-transformed price and volume data. The second tests were using technical indicators from the price data and using volume data. I found that volume data was highly additive to the models, but I could not consistently generate accuracies over 53% needed for profit. Some of the historical volume data provided by Dukascopy is suspect to me, as there are occasional massive outliers.
    - With the random forest model, I tested the random forest from sklearn and an imbalanced random forest from imblearn. The imbalanced random forest showed promise on non-transformed price data, but not the technical indicators.
    - Both the LSTM and random forest models generally over generalized to buying puts in my model.
    - A major hurdle is that March data is highly volatile and disrupts the model's ability to learn. Also, I ran into memory issues when trying to train more historical data. See the data [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/tree/master/data).

3. Tested Hidden Markov Models to access which regime the EUR/USD pair was in (tested 2, 3 and 4 regimes). The first test added the regime as a variable to the models, and a second test created separate models for each regime (i.e. broke the data into groups based on regimes, then trained/tested each group as a separate model). I only ran these tests using non-transformed price data. There was some promise here, but certain regimes had higher accuracies than others.

4. Tested using Principal Component Analysis (PCA) to reduce the features being trained on. However, any test using PCA significantly reduced the model's accuracy.

5. Briefly tested touch binaries. Touch binaries appear to need long forecast windows with large touch windows (i.e. 25+ pips over 12+ hours) to be profitable.

# Conclusion

I could not generate a consistent 53%+ accuracy model neded to make a profit. Some weeks had positive profit, and other weeks had a loss. Perhaps there are better trading signals from the data, making machine learning overly complicated for this exercise. Or, data of this short of time frame may be too difficult to predict reliably. Feel free to test using this repo. If nothing else, there is an efficient framework for grabbing live data from Dukascopy and submitting call and put options.

# Helpful suggestions if you'd like to test

1. Many of the old regime / PCA / other test code may be found [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/tree/master/old).

2. Again, the data is located [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/tree/master/data). The data is ask & bid open, high, low, close and volume data for the ten instruments noted above.

3. You can create a free demo account at [https://demo-login.dukascopy.com/binary/](https://demo-login.dukascopy.com/binary/). Add your username and password [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/blob/master/tmp/config.cfg).

4. You'll need ChromeDriver. You can download it [here](https://chromedriver.chromium.org/downloads). Once you've download, update the [config file](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/blob/master/tmp/config.cfg) with its location.
