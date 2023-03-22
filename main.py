from binance.client import Client
import pandas as pd
import talib
import pandas as pd
import numpy as np
import gym
import os

api_key = ''
api_secret = ''
client = Client(api_key, api_secret)
filename = 'btcusdt_2018_2020_1h.csv'
if not os.path.isfile(filename):
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2018", "1 Mar, 2020")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    df.to_csv(filename)
else:
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)    
print(df.tail())

# Рассчитываем индикаторы
df['rsi'] = talib.RSI(df['close'], timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'] = macd
df['macd_signal'] = macd_signal
df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)


class BinanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, window_size=40):
        super(BinanceEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.action_space = gym.spaces.Discrete(3)
        #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, window_size+1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9, window_size), dtype=np.float32)
        self.initial_balance = 1000.0
        self.current_balance = self.initial_balance
        self.position_size = 0.0
        self.position_value = 0.0
        self.trade_history = []
        self.current_step = self.window_size
        self.equity = self.current_balance + self.position_value

    def _calculate_profit(self, new_price):
        old_price = self.trade_history[-1][1]
        position_size = self.trade_history[-1][2]

        if position_size > 0:
            return (new_price - old_price) * position_size
        else:
            return 0    

    def _next_observation(self):
        frame = np.zeros((9, self.window_size))
        
        def pad_and_slice(data):
            data_slice = data[self.current_step : self.current_step + self.window_size]
            padded_data = np.pad(data_slice, (0, self.window_size - len(data_slice)), 'constant')
            return padded_data
        
        frame[0] = pad_and_slice(self.data['open'])
        frame[1] = pad_and_slice(self.data['high'])
        frame[2] = pad_and_slice(self.data['low'])
        frame[3] = pad_and_slice(self.data['close'])
        frame[4] = pad_and_slice(self.data['volume'])
        
        if self.current_step + self.window_size < len(self.data['close']):
            frame[5] = pad_and_slice(self.data['close'][self.current_step+1:])
        else:
            frame[5] = np.zeros((self.window_size,))
        frame[6] = pad_and_slice(self.data['rsi'])
        frame[7] = pad_and_slice(self.data['macd'] - self.data['macd_signal'])
        frame[8] = pad_and_slice((self.data['close'] - self.data['lower_band']) / (self.data['upper_band'] - self.data['lower_band']))


        return frame




    def reset(self):
        self.current_step = 0
        self.balance = 1000.0
        self.equity = 0.0
        self.bought_price = 0.0
        self.position_size = 0.0
        self.trade_history = []
        self.profit = 0.0
        return self._next_observation()

    def step(self, action):
        assert self.action_space.contains(action)
        current_price = self.data['close'][self.current_step]
        reward = 0

        if action == 0 and self.current_balance > 0: # BUY
            position_size = self.current_balance / current_price
            self.position_size = position_size
            self.position_value = self.position_size * current_price
            self.current_balance = 0
            self.trade_history.append(('BUY', current_price, self.position_size))
        elif action == 1 and self.position_size > 0: # SELL
            reward += self._calculate_profit(current_price)
            self.current_balance = self.position_size * current_price
            self.position_size = 0
            self.position_value = 0
            self.trade_history.append(('SELL', current_price, self.position_size))
        elif action == 2: # HOLD
            self.equity = self.position_size * current_price

        self.current_step += 1

        new_equity = self.current_balance + self.position_value
        if self.equity == 0:
            reward = 0
        else:
            reward += (new_equity - self.equity) / self.equity

        self.equity = new_equity

        obs = self._next_observation()

        done = self.current_step == len(self.data) - 1

        return obs, reward, done, {}


from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env

env = BinanceEnv(df)

# Создание векторизованной среды
env = make_vec_env(lambda: env, n_envs=1)

# Создание модели алгоритма DQN
model = DQN('MlpPolicy', env, verbose=1)

# Обучение модели
model.learn(total_timesteps=10000)

# Сохранение модели
model.save('binance_dqn_model')

# Загрузка обученной модели
model = DQN.load('binance_dqn_model')

# Создание новой среды для тестирования
test_env = BinanceEnv(df)

# Выполнение тестирования
obs = test_env.reset()
done = False
df_equity = pd.DataFrame(columns=['Test Equity'])
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    df_equity = pd.concat([df_equity, pd.DataFrame({'Test Equity': [test_env.equity]})], ignore_index=True)
    #print('Action:', action, 'Reward:', reward, 'Equity:', test_env.equity)


# Обновление данных
filename = 'btcusdt_2020_2023_1h.csv'
if not os.path.isfile(filename):
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "19 Jan, 2020", "19 Mar, 2023")
    updated_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'], unit='ms')
    updated_df.set_index('timestamp', inplace=True)
    updated_df = updated_df.astype(float)
    updated_df.to_csv(filename)
else:
    updated_df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)    

print(updated_df.tail())
# Рассчитываем индикаторы updated_df
updated_df['rsi'] = talib.RSI(updated_df['close'], timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(updated_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
updated_df['macd'] = macd
updated_df['macd_signal'] = macd_signal
updated_df['upper_band'], updated_df['middle_band'], updated_df['lower_band'] = talib.BBANDS(updated_df['close'], timeperiod=20)


# Создание новой среды с обновленными данными
updated_env = BinanceEnv(updated_df)

# Создание векторизованной среды
updated_env = make_vec_env(lambda: updated_env, n_envs=1)

# Создание новой модели алгоритма DQN
updated_model = DQN('MlpPolicy', updated_env, verbose=1)

# Обучение модели на обновленных данных
updated_model.learn(total_timesteps=10000)

# Сохранение обновленной модели
updated_model.save('updated_binance_dqn_model')

# Загрузка обновленной модели
updated_model = DQN.load('updated_binance_dqn_model')

# Создание новой среды для тестирования с обновленными данными
updated_test_env = BinanceEnv(updated_df)

# Выполнение тестирования с обновленной моделью
obs = updated_test_env.reset()
done = False
df_equity_updated = pd.DataFrame(columns=['updated Equity'])
while not done:
    action, _ = updated_model.predict(obs)
    obs, reward, done, info = updated_test_env.step(action)
    df_equity_updated = pd.concat([df_equity_updated, pd.DataFrame({'updated Equity': [updated_test_env.equity]})], ignore_index=True)
    #print('Action:', action, 'Reward:', reward, 'Equity:', updated_test_env.equity)


import numpy as np

# Загрузка тестовых данных
filename = 'btcusdt_2019_2023_1h.csv'
if not os.path.isfile(filename):
    test_klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2019", "1 Mar, 2023")
    test_df = pd.DataFrame(test_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='ms')
    test_df.set_index('timestamp', inplace=True)
    test_df = test_df.astype(float)
    test_df.to_csv(filename)
else:
    test_df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)        

# Рассчитываем индикаторы test_df
test_df['rsi'] = talib.RSI(test_df['close'], timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(test_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
test_df['macd'] = macd
test_df['macd_signal'] = macd_signal
test_df['upper_band'], test_df['middle_band'], test_df['lower_band'] = talib.BBANDS(test_df['close'], timeperiod=20)

# Создание среды для тестирования с обновленными данными
test_env = BinanceEnv(test_df)

# Создание векторизованной среды
test_env = make_vec_env(lambda: test_env, n_envs=1)

# Загрузка обученной модели
model = DQN.load('updated_binance_dqn_model')



# Выполнение тестирования
cumulative_profit = 0
episode_rewards = []
df_equity_last = pd.DataFrame(columns=['Last test Equity'])

obs = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    episode_rewards.append(reward)
    df_equity_last = pd.concat([df_equity_last, pd.DataFrame({'Last test Equity': [test_env.get_attr('equity')[0]]})], ignore_index=True)
    #print('Action:', action, 'Reward:', reward, 'Equity:', test_env.get_attr('equity')[0])

cumulative_profit = test_env.get_attr('equity')[0] - test_env.get_attr('initial_balance')[0]
print('Cumulative profit:', cumulative_profit)

import matplotlib.pyplot as plt
# вывод графика
plt.plot(df_equity['Test Equity'], label='Test Equity')
plt.plot(df_equity_updated['updated Equity'], label='updated Equity')
plt.plot(df_equity_last['Last test Equity'], label='Last test Equity')
plt.legend()
plt.show()


import time
from datetime import datetime



def real_time_trading(model):
    buy_price = 1
    sell = False
    btc = 0
    usdt = 1000
    sold_price = 1
    time_buy = str(datetime.datetime.now().date())
    time_to_buy = True
    
    while True:
        # Получение текущего времени
        current_time = datetime.datetime.now().time()

        # Получение свежих данных
        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2023", str(datetime.now().date()))
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        # Рассчитываем индикаторы
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)

        # Создание новой среды для тестирования с обновленными данными
        real_time_env = BinanceEnv(df)

        # Получение текущих наблюдений
        obs = real_time_env.reset()

        # Предсказание действий агента
        action, _ = model.predict(obs)

        # Получение текущей цены
        current_price = df['close'].iloc[-1]
        if(action == 0 and sell == False and time_to_buy == True ):
                btc = 100 / float(current_price)
                buy_price = current_price
                sold_price = current_price
                usdt = usdt-100
                sell = True 
                time_to_buy = False
        
        elif(action == 1 and sell == True and (current_price >= sold_price)):
                usdt = usdt + current_price * btc
                btc = 0
                sell = False
        
        if(current_price > sold_price):
            sold_price = current_price 

        # Вывод информации о текущей цене и предсказанном действии
        print(f"Current price: {current_price}, predicted action: {action},  sold_price: {sold_price}, % : {buy_price + buy_price*0.02}")

        # Открываем возможность для покупки каждый час
        if current_time.minute == 0 and current_time.second == 0 and usdt > 0:
            # Изменение флага time_to_buy на True
            time_to_buy = True
            print("Открытие для покупки")
            
        # Задержка перед следующим обновлением данных (например, 60 секунд)
        time.sleep(60)


real_time_trading(model)

