# Predicting EUR/USD Binary Options Using Deep Learning and Random Forest: A Comparative Study

## Introduction:

This study aims to predict EUR/USD binary options using deep learning and random forest models, and explore the challenges and strategies for improving the accuracy of these models. The study uses live data from Dukascopy, and builds upon the work of [SolbiatiAlessandro](https://github.com/SolbiatiAlessandro/RNN-stocks-prediction), who used a recurrent neural network (RNN) to predict EUR/USD binary options on a Dukascopy demo account.

The original code faced issues with scraping live data and executing trades accurately. To overcome these issues, the study proposes a new framework for pulling live data from Dukascopy at the end of each minute, and executing trades within 1-2 seconds of a minute finishing.

## Methodology:

The study uses a comparative approach to test ten instruments, including EUR/USD, EUR/JPY, EUR/GBP, EUR/CAD, EUR/CHF, XAU/USD, XAG/USD, EUR/HKD, USD/CNH, and EUR/SGD. Two models were tested, including a random forest and an LSTM model. The models were trained on minute data from April 1, 2020, to the present, and tested live for a week. The testing process involved adding the week's data to the dataset, retraining the models, and testing another week. The study also explored using technical indicators and volume data in the models, but found that volume data was highly additive, and technical indicators did not consistently generate accuracies over 53% needed for profit.

## Results:

The study found that forecasting at least 4 hours ahead led to increasing accuracy. The random forest model was tested using sklearn and an imbalanced random forest from imblearn. The imbalanced random forest showed promise on non-transformed price data, but not the technical indicators. Both the LSTM and random forest models generally over-generalized to buying puts in the model. The study also tested Hidden Markov Models to access which regime the EUR/USD pair was in and found that certain regimes had higher accuracies than others.

## Conclusion:

The study concludes that it could not generate a consistent 53%+ accuracy model needed to make a profit. Some weeks had positive profit, and other weeks had a loss. The study suggests that there may be better trading signals from the data, making machine learning overly complicated for this exercise, or data of this short of time frame may be too difficult to predict reliably. The study provides helpful suggestions for testing and improving the models, and notes that there is an efficient framework for grabbing live data from Dukascopy and submitting call and put options available in the repo.

## Helpful suggestions if you'd like to test

1. The data is located [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/tree/master/data). The data is ask & bid open, high, low, close and volume data for the ten instruments noted above.

2. You can create a free demo account at [https://demo-login.dukascopy.com/binary/](https://demo-login.dukascopy.com/binary/). Add your username and password [here](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/blob/master/tmp/config.cfg).

3. You'll need ChromeDriver. You can download it [here](https://chromedriver.chromium.org/downloads). Once you've download, update the [config file](https://github.com/ScrapeWithYuri/RNN-stocks-prediction/blob/master/tmp/config.cfg) with its location.
