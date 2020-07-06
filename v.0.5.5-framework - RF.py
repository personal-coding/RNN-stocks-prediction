from gevent.pool import Pool
from gevent import monkey

monkey.patch_all()

import requests, calendar, codecs, numpy as np
import pandas as pd
import _pickle as cPickle
import datetime, time, talib

from tkinter import *
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from sklearn.preprocessing import MinMaxScaler
import configparser

config = configparser.RawConfigParser()
config.read('tmp/config.cfg')

#app parameters
CHROME_PATH = config.get('app','CHROME_PATH')
URL = config.get('app','URL')
USR = config.get('app','USR')
PSW = config.get('app','PSW')

class DukascopyBinary(object):
    """docstring for ClassName"""
    def __init__(self, URL):
        self.URL = URL
        self.options = webdriver.ChromeOptions()
        #self.options.add_argument("--start-maximized")
        #self.prefs = {"profile.managed_default_content_settings.images": 2}
        #self.options.add_experimental_option("prefs", self.prefs)
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(CHROME_PATH, chrome_options=self.options)
        self.driver.implicitly_wait(60)
        self.driver.get(URL)
        
    def set_type(self, ASK=0, BID=1):
        global USR
        global PSW

        time.sleep(3)
        #logging in
        self.login(USR,PSW)
        time.sleep(3)
        self.driver.find_element_by_xpath("//span[contains(@id,'button') and text()='Close']").click()
        time.sleep(2)
        #1 minute time frame
        self.driver.find_element_by_xpath("//div[@title='Select time period' and @role='button']").click()
        time.sleep(2)
        self.driver.find_element_by_xpath("//div[contains(@id,'1M')]//div[contains(text(),'Minute')]").click()
        time.sleep(2)
        #detached OHLC view
        self.driver.find_element_by_xpath("//div[@title='Select OHLC type' and @role='button']").click()
        time.sleep(2)
        self.driver.find_element_by_xpath("//span[text()='Detached OHLC Index']").click()
        time.sleep(2)
        #amount 10
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-amountfield') and @name='amount']").send_keys("0")
        time.sleep(2)
        #duration 6 hours
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-numberfield') and @name='hours']").send_keys(Keys.BACKSPACE+"6")
        time.sleep(2)
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-numberfield') and @name='minutes']").send_keys(Keys.BACKSPACE+"0")
        time.sleep(2)
        
        if(ASK):
            self.type = "ASK"
            self.driver.find_element_by_xpath("//div[@title='Select offer side' and @role='listbox']").click()
            time.sleep(2)
            self.driver.find_element_by_xpath("//div[@role='menuitemradio']//div[text()='ASK']").click()
            time.sleep(2)
            
        elif(BID):
            self.type = "BID"

        #self.driver.find_element_by_xpath("//span[text()='Active Options ']").click()
        
        '''time.sleep(2)
        for i in range(1):
            self.driver.find_element_by_xpath("//div[@title='Auto shift']").click()
            time.sleep(0.02)'''
    
    def login(self,USR,PSW):
        self.driver.find_element_by_xpath("//input[@name='login']").send_keys(USR)
        self.driver.find_element_by_xpath("//input[@name='password']").send_keys(PSW)
        time.sleep(1)
        self.driver.find_element_by_xpath("//span[text()='Log in']").click()
    
    def call(self):
        self.driver.find_element_by_xpath("//div[contains(@class,'call')]").click()
        #time.sleep(0.2)
        self.driver.find_element_by_xpath("//span[contains(@id,'button') and text()='Yes']").click()
    
    def put(self):
        self.driver.find_element_by_xpath("//div[contains(@class,'put')]").click()
        #time.sleep(0.2)
        self.driver.find_element_by_xpath("//span[contains(@id,'button') and text()='Yes']").click()

class App(object):
    def __init__(self):
        global URL

        self.ask = DukascopyBinary(URL)
        self.ask.set_type(ASK=1,BID=0)
        self.ask.driver.implicitly_wait(60)
        self.running = False

        self.ignored_exceptions = (NoSuchElementException,StaleElementReferenceException,)

        self.testing = False
        self.servertime = " 04:23:44 GMT"
        self.BASE_URL = 'https://freeserv.dukascopy.com/2.0/index.php?path=chart%2Fjson3&instrument={0}&offer_side={1}&interval=1MIN' \
                   '&splits=true&limit={2}&time_direction=P&timestamp={3}&jsonp="'

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Referer': 'https://demo-login.dukascopy.com/binary/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        self.bid_ask = ['A', 'B']
        self.tickers = ['EUR/USD', 'EUR/JPY', 'EUR/GBP', 'EUR/CAD', 'EUR/CHF', 'XAU/USD', 'XAG/USD', 'EUR/HKD', 'USD/CNH', 'EUR/SGD']

        self.bid_ask2 = ['Ask', 'Bid']
        self.tickers2 = ['EURUSD', 'EURJPY', 'EURGBP', 'EURCAD', 'EURCHF', 'XAU', 'XAG', 'EURHKD', 'USDCNH', 'EURSGD']

        self.num_worker_threads = len(self.bid_ask) * len(self.tickers)
        self.pool = Pool(self.num_worker_threads)

        self.A = []
        self.now_now = 0
        self.most_recent = 0
        self.TIMESTEP = 30
        self.limit = 180 + self.TIMESTEP*3
        self.last = 1591651080000
        self.most_recent = self.last
        self.BREAK = 840000
        self.concat_list = []
        self.csv_filepath = 'data/MAIN_1min_onlyFX_min_diff_more_data - Copy Copy.csv'
        self.rf_0_pickle_path = 'model_1min_onlyFX_no_regime_more_data_balanced.pickle'

        self.df = pd.read_csv(self.csv_filepath)[:self.BREAK]
        self.df = self.df.drop(columns=['Datetime','Min Diff','Hour'], axis=1)

        self.scaler = MinMaxScaler()
        self.MAX, self.MAX_TEST = 0, 0

        for b_a in self.bid_ask2:
            for ticker in self.tickers2:
                high, low, close, open, volume = self.df[ticker + ' ' + b_a + ' High'], \
                                                 self.df[ticker + ' ' + b_a + ' Low'], \
                                                 self.df[ticker + ' ' + b_a], \
                                                 self.df[ticker + ' ' + b_a + ' Open'], \
                                                 self.df[ticker + ' ' + b_a + ' Volume']

                # overlap ind
                self.df[ticker + b_a + 'midprice'] = talib.MIDPRICE(high, low, timeperiod=self.TIMESTEP)

                # volume ind
                self.df[ticker + b_a + 'volume'] = volume

                # volatitility ind
                self.df[ticker + b_a + 'natr'] = talib.NATR(high, low, close, timeperiod=self.TIMESTEP)

                # momentum ind
                self.df[ticker + b_a + 'bop'] = talib.BOP(open, high, low, close)
                self.df[ticker + b_a + 'adxr'] = talib.ADXR(high, low, close, timeperiod=self.TIMESTEP)

                # stats ind
                self.df[ticker + b_a + 'stddev'] = talib.STDDEV(close, timeperiod=self.TIMESTEP, nbdev=1)
                self.df[ticker + b_a + 'correl'] = talib.CORREL(high, low, timeperiod=self.TIMESTEP)

                # cycle ind
                self.df[ticker + b_a + 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

                self.df = self.df.drop(
                    columns=[ticker + ' ' + b_a + ' High', ticker + ' ' + b_a + ' Low', ticker + ' ' + b_a + ' Open',
                             ticker + ' ' + b_a, ticker + ' ' + b_a + ' Volume'], axis=1)

                self.MAX_TEST = max(
                    self.df[ticker + b_a + 'natr'].isnull().sum(),
                    self.df[ticker + b_a + 'adxr'].isnull().sum(),
                    self.df[ticker + b_a + 'midprice'].isnull().sum(),
                    self.df[ticker + b_a + 'HT_TRENDMODE'].isnull().sum(),
                    self.df[ticker + b_a + 'correl'].isnull().sum(),
                    self.df[ticker + b_a + 'stddev'].isnull().sum(),
                    self.df[ticker + b_a + 'bop'].isnull().sum()
                )

                if self.MAX_TEST > self.MAX: self.MAX = self.MAX_TEST

                del high, low, close, open, volume

        self.scaler.fit(self.df[self.df.columns][self.MAX:self.BREAK])

        with codecs.open(self.rf_0_pickle_path, mode='rb') as f:
            self.rf_0_model = cPickle.load(f)

        del self.df, self.MAX, self.MAX_TEST, self.bid_ask2, self.tickers2

        print('Model Ready...')

    def test_xpath_ask(self, xpath):
        self.retry = True
        self.returned_value = ''
        while self.retry:
            try:
                self.returned_value = WebDriverWait(app.ask.driver, 5, ignored_exceptions=self.ignored_exceptions) \
                    .until(EC.presence_of_element_located((By.XPATH, xpath))).get_attribute("innerText")
                self.retry = False
                return self.returned_value
            except StaleElementReferenceException as e:
                self.retry = True

    def response(self, site):
        qq = requests.get(site, headers=self.headers, stream=True)
        return qq.text

    def process(self, site, ticker, b_a):
        response_body = self.response(site)[2:-2]
        response_list = eval(response_body)
        self.l[ticker + '_' + b_a] = np.array(response_list[::-1])

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def reshape_vals(self,df):
        nsamples, nx, ny = df.shape
        return df.reshape((nsamples, nx * ny))

    def rf_call_or_put(self,pred,pred2,alfa):
        if np.sum(pred) > 0:
            if pred2[0][0][1]-pred2[2][0][1]>alfa and pred2[0][0][1] > pred2[1][0][1]: #-pred2[2][0][1]
                self.ask.call()
                print ('call executed')

            elif pred2[1][0][1]-pred2[2][0][1]>alfa and pred2[1][0][1] > pred2[0][0][1]: #-pred2[2][0][1]
                self.ask.put()
                print ('put executed')

            else:
                print ('no signal')

    def trading(self):
        if self.running:
            self.servertime = self.test_xpath_ask("//span[text()='Server Time:']/..")
            self.servertime = self.servertime.strip().replace(' GMT', '').replace('Server Time:','')
            self.servertime = self.servertime[-2:]

            if self.testing:
                self.now_now = calendar.timegm(time.gmtime())
                self.l = {}

                for b_a in self.bid_ask:
                    for ticker in self.tickers:
                        self.pool.apply_async(self.process, args=(
                            self.BASE_URL.format(ticker,
                            b_a,
                            str(self.limit),
                            str(self.now_now) + '000'), ticker, b_a))
                self.pool.join()

                self.most_recent = self.l[list(self.l)[0]][-1][0]
                if not all(self.most_recent == self.l[value][-1][0] for value in self.l):
                    self.most_recent = self.last
                elif self.most_recent != self.last:
                    self.df = pd.DataFrame()
                    self.A = []

                    for b_a in self.bid_ask:
                        for ticker in self.tickers:
                            high, low, close, open, volume = self.l[ticker + '_' + b_a][:, 2], \
                                                             self.l[ticker + '_' + b_a][:, 3], \
                                                             self.l[ticker + '_' + b_a][:, 4],\
                                                             self.l[ticker + '_' + b_a][:, 1], \
                                                             self.l[ticker + '_' + b_a][:, 5]

                            self.df[ticker + b_a + 'midprice'] = talib.MIDPRICE(high, low, timeperiod=self.TIMESTEP)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'volume'] = volume[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'natr'] = talib.NATR(high, low, close, timeperiod=self.TIMESTEP)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'bop'] = talib.BOP(open, high, low, close)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'adxr'] = talib.ADXR(high, low, close, timeperiod=self.TIMESTEP)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'stddev'] = talib.STDDEV(close, timeperiod=self.TIMESTEP, nbdev=1)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'correl'] = talib.CORREL(high, low, timeperiod=self.TIMESTEP)[self.TIMESTEP*3:]
                            self.df[ticker + b_a + 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)[self.TIMESTEP*3:]

                            del high, low, close, open, volume

                    self.df[self.df.columns] = self.scaler.transform(self.df[self.df.columns])
                    self.df = self.df.to_numpy()

                    self.A.append(self.df.tolist())
                    self.df = np.array(self.A)
                    self.df = self.reshape_vals(self.df)

                    pred = self.rf_0_model.predict(self.df)
                    pred2 = self.rf_0_model.predict_proba(self.df)
                    self.rf_call_or_put(pred, pred2, 0.55)

                    del self.df, self.A

                    print(pred2)
                    self.last = self.most_recent
                    self.testing = False

            if self.servertime == '58' and not self.testing:
                self.testing = True

        self.root.after(100, self.trading)

    def run(self):
        self.root = Tk()
        self.root.title("v.0.5.5-framework-RF")
        self.root.geometry("300x100")

        app = Frame(self.root)
        app.grid()

        start = Button(app, text="Start", command=self.start)
        stop = Button(app, text="Stop", command=self.stop)
        start.grid(row=0, column=0, padx=(40, 40), pady=(40, 40))
        stop.grid(row=0, column=1, padx=(40, 40), pady=(40, 40))

        start.grid()
        stop.grid()

        self.root.after(1000, self.trading)  # After 1 second, call scanning
        self.root.mainloop()

app = App()
app.run()