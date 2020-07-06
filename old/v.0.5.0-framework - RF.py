from gevent.pool import Pool
from gevent import monkey

monkey.patch_all()
from hmmlearn.hmm import GaussianHMM

import requests, calendar, codecs, numpy as np
import pandas as pd
import _pickle as cPickle
import datetime, time

from tkinter import *
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
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
        #detached OHLC view *id :53 is variable attention
        '''self.driver.find_element_by_xpath("//div[@title='Select OHLC type' and @role='button']").click()
        time.sleep(2)
        self.driver.find_element_by_xpath("//span[text()='Detached OHLC Index']").click()
        time.sleep(2)'''
        #amount 100
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-amountfield') and @name='amount']").send_keys("00")
        time.sleep(2)
        #duration 45
        '''self.driver.find_element_by_xpath("//input[contains(@id,'bp-numberfield') and @name='minutes']").send_keys(Keys.BACKSPACE+"45")
        time.sleep(2)'''
        
        if(ASK):
            self.type = "ASK"
            self.driver.find_element_by_xpath("//div[@title='Select offer side' and @role='listbox']").click()
            time.sleep(2)
            self.driver.find_element_by_xpath("//div[@role='menuitemradio']//div[text()='ASK']").click()
            time.sleep(2)
            
        elif(BID):
            self.type = "BID"

        #self.driver.find_element_by_xpath("//span[text()='Active Options ']").click()
        
        time.sleep(2)
        for i in range(100):
            self.driver.find_element_by_xpath("//div[@title='Auto shift']").click()
            time.sleep(0.02)
    
    def login(self,USR,PSW):
        self.driver.find_element_by_xpath("//input[@name='login']").send_keys(USR)
        self.driver.find_element_by_xpath("//input[@name='password']").send_keys(PSW)
        time.sleep(1)
        self.driver.find_element_by_xpath("//span[text()='Log in']").click()
    
    def call(self):
        self.driver.find_element_by_xpath("//div[contains(@class,'call')]").click()
        time.sleep(0.2)
        self.driver.find_element_by_xpath("//span[contains(@id,'button') and text()='Yes']").click()
    
    def put(self):
        self.driver.find_element_by_xpath("//div[contains(@class,'put')]").click()
        time.sleep(0.2)
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
        self.hmm_predict = 0
        self.BASE_URL = 'https://freeserv.dukascopy.com/2.0/index.php?path=chart%2Fjson3&instrument={0}&offer_side={1}&interval=1MIN' \
                   '&splits=true&limit={2}&time_direction=P&timestamp={3}&jsonp="'

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Referer': 'https://demo-login.dukascopy.com/binary/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        self.bid_ask = ['A', 'B']
        self.tickers = ['EUR/USD', 'EUR/JPY', 'EUR/CAD', 'EUR/GBP']

        self.num_worker_threads = len(self.bid_ask) * len(self.tickers)
        self.pool = Pool(self.num_worker_threads)

        self.A = []
        self.now_now = 0
        self.most_recent = 0
        self.limit = 120
        self.last = 1591651080000
        self.most_recent = self.last
        self.BREAK = 10000
        self.concat_list = []
        self.csv_filepath = "data/MAIN.csv"
        self.hmm_pickle_path = 'hmm_model/model_Close_x_4_None_False_GaussianHMM.pkl'
        self.rf_0_pickle_path = 'rf_model_0.pickle'
        self.rf_1_pickle_path = 'rf_model_1.pickle'
        self.rf_2_pickle_path = 'rf_model_2.pickle'
        self.rf_3_pickle_path = 'rf_model_3.pickle'

        self.df = pd.read_csv(self.csv_filepath)[:self.BREAK]
        self.df = self.df.drop(columns=['Datetime', 'X_regime_0', 'Y_regime_0'], axis=1)
        self.df['Spread_close'] = self.df['Close_x'] - self.df['Close_y']

        self.means = []
        self.stds = []

        for column in self.df:
            self.means.append(np.mean(self.df[column][:self.BREAK]))
            self.stds.append(np.std(self.df[column][:self.BREAK]))

        with codecs.open(self.hmm_pickle_path, mode='rb') as f:
            self.hmm_model = cPickle.load(f)

        with codecs.open(self.rf_0_pickle_path, mode='rb') as f:
            self.rf_0_model = cPickle.load(f)

        with codecs.open(self.rf_1_pickle_path, mode='rb') as f:
            self.rf_1_model = cPickle.load(f)

        with codecs.open(self.rf_2_pickle_path, mode='rb') as f:
            self.rf_2_model = cPickle.load(f)

        with codecs.open(self.rf_3_pickle_path, mode='rb') as f:
            self.rf_3_model = cPickle.load(f)

        del self.df

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
            if pred2[0][0][1]>alfa and pred2[0][0][1] > pred2[1][0][1]: #-pred2[2][0][1]
                self.ask.call()
                print ('call executed')

            elif pred2[1][0][1]>alfa and pred2[1][0][1] > pred2[0][0][1]: #-pred2[2][0][1]
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
                    self.hmm_predict = self.hmm_model.predict([[self.l['EUR/USD_A'][-1][-2]]])[0]

                    print('HMM Predicted:', self.hmm_predict)

                    self.concat_list = []
                    for z, a, b, z_b, a_b, b_b, x, y in zip(self.l['EUR/JPY_A'], self.l['EUR/GBP_A'], self.l['EUR/CAD_A'],
                                                            self.l['EUR/JPY_B'], self.l['EUR/GBP_B'], self.l['EUR/CAD_B'],
                                                            self.l['EUR/USD_A'], self.l['EUR/USD_B']):
                        self.concat_list.append([z[-2], a[-2], b[-2], z[-1], a[-1], b[-1],
                                            z_b[-2], a_b[-2], b_b[-2], z_b[-1], a_b[-1], b_b[-1], x[-2] - y[-2]])

                    self.concat_list = np.concatenate((self.l['EUR/USD_A'][:, -5:], self.l['EUR/USD_B'][:, -5:], self.concat_list), 1)
                    self.concat_list = (self.concat_list - self.means) / self.stds
                    self.A.append(self.concat_list.tolist())
                    self.concat_list = np.array(self.A)
                    self.concat_list = self.reshape_vals(self.concat_list)

                    if self.hmm_predict == 0:
                        pred = self.rf_0_model.predict(self.concat_list)
                        pred2 = self.rf_0_model.predict_proba(self.concat_list)
                        self.rf_call_or_put(pred, pred2, 0.55)
                    elif self.hmm_predict == 1:
                        pred = self.rf_1_model.predict(self.concat_list)
                        pred2 = self.rf_1_model.predict_proba(self.concat_list)
                        self.rf_call_or_put(pred, pred2, 0.60)
                    elif self.hmm_predict == 2:
                        pred = self.rf_2_model.predict(self.concat_list)
                        pred2 = self.rf_2_model.predict_proba(self.concat_list)
                        self.rf_call_or_put(pred, pred2, 0.65)
                    elif self.hmm_predict == 3:
                        pred = self.rf_3_model.predict(self.concat_list)
                        pred2 = self.rf_3_model.predict_proba(self.concat_list)
                        self.rf_call_or_put(pred, pred2, 0.55)

                    del self.concat_list

                    print(pred2)
                    self.last = self.most_recent
                    self.testing = False

            if self.servertime == '59' and not self.testing:
                self.testing = True
                self.A = []

        self.root.after(100, self.trading)

    def run(self):
        self.root = Tk()
        self.root.title("v.0.5.0-framework-RF")
        self.root.geometry("300x100")

        self.root2 = Tk()
        self.text = Text(self.root2, width=40, height=40, font=("Helvetica", 16))
        self.text.pack()
        self.text.insert(1.0, "PREDICTION vals:")

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
        self.root2.mainloop()

app = App()
app.run()