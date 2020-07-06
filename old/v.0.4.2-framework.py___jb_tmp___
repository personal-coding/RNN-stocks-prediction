#v.0.4.x FULL IMPLEMENTATION (NN MODEL AND TRADING APP)
#v.0.4.1 portabilty (logging+config for exe)
#v.0.4.2 support for DukascopyBinary 1.10.26

import logging
import logging.config

logerrors = 0

if(logerrors):
    logging.basicConfig(filename='tmp/error.log', level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
else:
    print ('\nconsole printing\n')



#try:
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import datetime
import locale
import sys

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

import ConfigParser
config = ConfigParser.RawConfigParser()
config.read('tmp/config.cfg')
    
'''except Exception as e:
    if(logerrors):
        logger.error(e)
    else:
        print (e)'''
        

#training parameters
ASK_FNAME = "rtf/eurusd002-ask.rtf"
BID_FNAME = "rtf/eurusd002-BID.rtf"
WINDOW=90
FORECAST=45
EMB_SIZE=11
STEP=1 
TRAIN_TEST_PERCENTAGE=0.9
SAVE_NAME = "classification_model.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV = "classification"
TRAINING = 1
TESTING = 0
NUMBER_EPOCHS = 10
TRADING_DAYS = 14

#execution parameters
try:
    LOAD_NAME = config.get('model','LOAD_NAME')
except Exception as e:
    if(logerrors):
        logger.error(e)

def ternary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV):
    count=0
    X,Y = [],[]
    i=0
    try:
        count=count+1
        try:
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

            x_i = np.column_stack((ao,ah,al,ac,av,bo,bh,bl,bc,bv, s))
                
        except Exception as e:
            print (e)
            pass

    except Exception as e:
        print (e)
        pass
    return x_i

def format_data(df,ternary=1,binary=0):
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
    
    #print(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC)
    
    if(ternary==1):
        return ternary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV)

    elif(binary==1):
        return binary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV)


def spread(Y):
    still = 0
    for vec in Y:
        if vec[2]==1:
            still=still+1
    spread =still*100/len(Y)
    print (spread,"%")
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

try:
    #MODEL DEFINITION
    print ('initializing model..')
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

except Exception as e:
    if(logerrors):
        logger.error(e)
    else:
        print (e)


#try:
import httplib
import json
from Tkinter import *
import datetime
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from pymouse import PyMouse
from threading import Thread
    
'''except Exception as e:
    if(logerrors):
        logger.error(e)
    else:
        print (e)'''

#app parameters
try:
    CHROME_PATH = config.get('app','CHROME_PATH')
    URL = config.get('app','URL')
    USR = config.get('app','USR')
    PSW = config.get('app','PSW')
except Exception as e:
    logger.error(e)

class DukascopyBinary(object):
    """docstring for ClassName"""
    def __init__(self, URL):
        self.URL = URL
        self.options = webdriver.ChromeOptions()
        #self.options.add_argument("--start-maximized")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(CHROME_PATH, chrome_options=self.options)
        self.driver.implicitly_wait(60)
        self.driver.get(URL)
        #datafeed values
        self.minute=60
        self.vec=[]
        self.data=[]
        
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
        self.driver.find_element_by_xpath("//div[@title='Select OHLC type' and @role='button']").click()
        time.sleep(2)
        self.driver.find_element_by_xpath("//span[text()='Detached OHLC Index']").click()
        time.sleep(2)
        #amount 100
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-amountfield') and @name='amount']").send_keys("00")
        time.sleep(2)
        #duration 45
        self.driver.find_element_by_xpath("//input[contains(@id,'bp-numberfield') and @name='minutes']").send_keys(Keys.BACKSPACE+"45")
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
        
        time.sleep(2)
        for i in xrange(100):
            self.driver.find_element_by_xpath("//div[@title='Auto shift']").click()
            time.sleep(0.02)
            
    def set_mouse(self,m):
        print ('put mouse in',self.type,' position.. (5s)')
        time.sleep(5)
        self.x,self.y=m.position()
        print ('-> position captured')
        
    
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
        global model
        model.load_weights(LOAD_NAME)
        
        global URL
        self.ask = DukascopyBinary(URL)
        self.ask.set_type(ASK=1,BID=0)
        self.bid = DukascopyBinary(URL)
        self.bid.set_type(ASK=0,BID=1)
        self.ask.driver.implicitly_wait(60)
        self.bid.driver.implicitly_wait(60)
        self.running = False
        self.m = PyMouse()

        self.ignored_exceptions = (NoSuchElementException,StaleElementReferenceException,)
        
        self.data = []
        self.temp = ""
        self.testing = True
        self.first = True
        
        self.datetime1,self.datetime2 = "",""
        self.servertime = " 04:23:44 GMT"
        
        self.aOpen,self.aHigh,self.aLow,self.aClose,self.aVolume=0,0,0,0,0
        self.bOpen,self.bHigh,self.bLow,self.bClose,self.bVolume=0,0,0,0,0
        self.aVolume2, self.bVolume2 = 0.01, 0.01
    
    def test_xpath_ask(self, xpath):
        self.retry = True
        self.returned_value = ""
        while self.retry:
            try:
                self.returned_value = WebDriverWait(app.ask.driver, 5, ignored_exceptions=self.ignored_exceptions) \
                    .until(EC.presence_of_element_located((By.XPATH, xpath))).get_attribute("innerText")
                self.retry = False
                return self.returned_value
            except StaleElementReferenceException as e:
                self.retry = True
    
    def test_xpath_bid(self, xpath):
        self.retry = True
        self.returned_value = ""
        while self.retry:
            try:
                self.returned_value = WebDriverWait(app.bid.driver, 5, ignored_exceptions=self.ignored_exceptions) \
                    .until(EC.presence_of_element_located((By.XPATH, xpath))).get_attribute("innerText")
                self.retry = False
                return self.returned_value
            except StaleElementReferenceException as e:
                self.retry = True
    
    def mouse_config(self):
        self.ask.set_mouse(self.m)
        self.bid.set_mouse(self.m)
        
    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def a_datetime1(self):
        self.datetime1 = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Date:')]/div").replace(',','')

    def a_datetime2(self):
        self.datetime2 = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Time:')]/div").replace(',','')

    def a_open(self):
        self.aOpen = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Open:')]/div").replace(',','')

    def a_high(self):
        self.aHigh = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'High:')]/div").replace(',','')

    def a_low(self):
        self.aLow = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Low:')]/div").replace(',','')

    def a_close(self):
        self.aClose = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Close:')]/div").replace(',','')

    def a_volume(self):
        self.aVolume = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div").replace(',','')

    def a_volume2(self):
        self.aVolume2 = self.test_xpath_ask("//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div").replace(',','')

    def b_open(self):
        self.bOpen = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'Open:')]/div").replace(',','')

    def b_high(self):
        self.bHigh = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'High:')]/div").replace(',','')

    def b_low(self):
        self.bLow = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'Low:')]/div").replace(',','')

    def b_close(self):
        self.bClose = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'Close:')]/div").replace(',','')

    def b_volume(self):
        self.bVolume = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div").replace(',','')

    def b_volume2(self):
        self.bVolume2 = self.test_xpath_bid("//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div").replace(',','')
    
    def __tensor__(self):
        print ('checking tensor')
        if(len(self.data)>WINDOW):
            df = pd.DataFrame(self.data[-WINDOW:])
            df = df.rename(columns={ df.columns[0] : 'Datetime', df.columns[1] : 'Open_x', df.columns[2] : 'High_x', df.columns[3] : 'Low_x', df.columns[4]: 'Close_x', df.columns[5] : 'Volume_x', df.columns[6] : 'Open_y', df.columns[7] : 'High_y', df.columns[8] : 'Low_y', df.columns[9]: 'Close_y', df.columns[10] : 'Volume_y' })
            self.X = format_data(df,1,0)
            self.X = np.array(self.X)
            return 1
        else: 
            return 0
        
    def __alg08__(self,pred,alfa):
        if pred[0][0]-pred[0][2]>alfa:
            self.ask.call()
            print ('call executed')
        
        elif pred[0][1]-pred[0][2]>alfa:
            self.ask.put()
            print ('put executed')
        
        else:
            print ('no signal')
    
    def trading(self):
        if self.running:
            #print ('RUNNING')

            #try:

            self.retry = True
            while self.retry:
                try:
                    now = WebDriverWait(app.ask.driver, 5, ignored_exceptions=self.ignored_exceptions) \
                        .until(EC.presence_of_element_located((By.XPATH, "//span[text()='OHLC Index']/../..//div[contains(text(),'Date:')]/div"))).get_attribute("innerHTML")+ \
                          " " + WebDriverWait(app.ask.driver, 5, ignored_exceptions=self.ignored_exceptions) \
                          .until(EC.presence_of_element_located((By.XPATH, "//span[text()='OHLC Index']/../..//div[contains(text(),'Time:')]/div"))).get_attribute("innerHTML")
                    self.retry = False
                except StaleElementReferenceException as e:
                    self.retry = True

            self.servertime = self.test_xpath_bid("//span[text()='Server Time:']/..")
            self.servertime = self.servertime.strip().replace(' GMT', '').replace('Server Time:','')
            #now = self.servertime[1:6]

            if self.first:
                self.temp = now
                self.first = False

            self.servertime = self.servertime[-2:]

            t1 = Thread(target=self.a_volume2)
            t2 = Thread(target=self.b_volume2)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            if (float(self.aVolume2) >= float(self.aVolume)) and (float(self.bVolume2) >= float(self.bVolume)):
                t1 = Thread(target=self.a_datetime1)
                t2 = Thread(target=self.a_datetime2)
                t3 = Thread(target=self.a_open)
                t4 = Thread(target=self.a_high)
                t5 = Thread(target=self.a_low)
                t6 = Thread(target=self.a_close)
                t7 = Thread(target=self.b_open)
                t8 = Thread(target=self.b_high)
                t9 = Thread(target=self.b_low)
                t10 = Thread(target=self.b_close)

                t1.start()
                t2.start()
                t3.start()
                t4.start()
                t5.start()
                t6.start()
                t7.start()
                t8.start()
                t9.start()
                t10.start()

                t1.join()
                t2.join()
                t3.join()
                t4.join()
                t5.join()
                t6.join()
                t7.join()
                t8.join()
                t9.join()
                t10.join()

                self.aVolume = self.aVolume2
                self.bVolume = self.bVolume2

            if (self.servertime == '00' or float(self.aVolume2) < float(self.aVolume) or float(self.bVolume2) < float(self.bVolume)) and self.testing:
                #print ('NEW VALUE')

                try:
                    print (self.aOpen,self.aHigh,self.aLow,self.aClose,self.aVolume,self.bOpen,self.bHigh,self.bLow,self.bClose,self.bVolume)
                    #print ('NEW VALUE APPENDED',self.aClose)

                    self.testing = False

                    _time = datetime.datetime.strptime(app.datetime1+" "+app.datetime2,"%Y-%m-%d %H:%M:%S.%f")

                    self.data.append([_time,float(self.aOpen),float(self.aHigh),float(self.aLow),float(self.aClose),float(self.aVolume),float(self.bOpen),float(self.bHigh),float(self.bLow),float(self.bClose),float(self.bVolume)])

                    if(self.__tensor__()):
                        pred =model.predict(np.reshape(self.X, (1,self.X.shape[0],self.X.shape[1])))
                        self.text.insert(1.0, pred)
                        self.__alg08__(pred, 0.47) #0.495

                    self.m.move(self.ask.x, self.ask.y)
                    time.sleep(1)
                    self.m.move(self.bid.x, self.bid.y)
                    self.ask.driver.find_element_by_xpath("//div[@title='Auto shift']").click()
                    self.bid.driver.find_element_by_xpath("//div[@title='Auto shift']").click()


                except Exception as e:
                    print ('inner loop exception:',str(e))
                    pass

            if now != self.temp and self.servertime != '00':
                self.testing = True
                self.temp = now
                self.aVolume, self.bVolume, self.aVolume2, self.bVolume2 = -1, -1, 0, 0

            #print ('START SCANNING', self.aClose)
            '''self.datetime1 = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Date:')]/div")
            self.datetime2 = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Time:')]/div")
            self.aOpen = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Open:')]/div")
            self.aHigh = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'High:')]/div")
            self.aLow = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Low:')]/div")
            self.aClose = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Close:')]/div")
            self.aVolume = self.test_xpath(app.ask.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div")
            self.bOpen = self.test_xpath(app.bid.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Open:')]/div")
            self.bHigh = self.test_xpath(app.bid.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'High:')]/div")
            self.bLow = self.test_xpath(app.bid.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Low:')]/div")
            self.bClose = self.test_xpath(app.bid.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Close:')]/div")
            self.bVolume = self.test_xpath(app.bid.driver, "//span[text()='OHLC Index']/../..//div[contains(text(),'Volume')]/div")'''

            #print ('FINISH SCANNING', self.aClose)

            #except Exception as e:

                #print ('outer loop exception:',str(e))
                #pass
            
        self.root.after(300, self.trading)

    def run(self):
        self.root = Tk()
        self.root.title("v.0.4.2-framework")
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


df = pd.read_csv('data/MAIN.csv')
df['Spread_close'] = df['Close_x'] - df['Close_y']
df2 = df[:50000]

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

del df
del df2

"""
to solve:

sync between ask and bid windows (when update time is not sync)
increase frequency of scanning?
ALWAYS double check with official DUKASCOPY data
"""

if(logerrors):
    try:
        app = App()
    except Exception as e: 
        logger.error(e)
else:
    app = App()


try:
    app.mouse_config()
except Exception as e: 
    if(logerrors):
        logger.error(e)
    else:
        print e

try:
    app.run()
except Exception as e:
    if(logerrors):
        logger.error(e)
    else:
        print e