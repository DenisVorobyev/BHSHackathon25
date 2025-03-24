import backtrader as bt
from binance.client import Client
import requests
import json
import time
import pandas as pd
import numpy as np
from backtrader.indicators import MACD
from backtrader.strategy import Strategy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
from scipy import stats
import threading
import queue
import websocket
import json

class StatisticalArbitrageStrategy(Strategy):
    params = (
        ('pair1', 'BTCUSDT'),
        ('pair2', 'ETHUSDT'),
        ('lookback_period', 60),  # Number of periods for calculating statistics
        ('zscore_threshold', 2.0),  # Z-score threshold for trading
        ('position_size', 0.1),  # Position size as fraction of portfolio
        ('stop_loss', 0.05),  # Stop loss percentage
        ('take_profit', 0.1),  # Take profit percentage
    )

    def __init__(self):
        self.pair1 = self.datas[0]
        self.pair2 = self.datas[1]
        self.order = None
        self.position_size = 0
        self.entry_price = 0
        self.spread_history = []
        self.zscore = 0
        self.cointegration = False

    def calculate_spread(self):
        # Calculate the spread between the two assets
        return self.pair1[0] - self.pair2[0]

    def calculate_zscore(self):
        if len(self.spread_history) < self.p.lookback_period:
            return 0
        
        spread_array = np.array(self.spread_history[-self.p.lookback_period:])
        mean = np.mean(spread_array)
        std = np.std(spread_array)
        return (self.spread_history[-1] - mean) / std

    def check_cointegration(self):
        if len(self.spread_history) < self.p.lookback_period:
            return False
        
        # Perform cointegration test
        _, pvalue, _ = coint(
            self.pair1.get(size=self.p.lookback_period),
            self.pair2.get(size=self.p.lookback_period)
        )
        return pvalue < 0.05  # 95% confidence level

    def calculate_position_size(self):
        # Calculate position size based on volatility
        volatility = np.std(self.spread_history[-self.p.lookback_period:])
        return self.p.position_size * (1 / volatility)

    def next(self):
        # Update spread history
        current_spread = self.calculate_spread()
        self.spread_history.append(current_spread)
        
        # Keep only the lookback period
        if len(self.spread_history) > self.p.lookback_period:
            self.spread_history.pop(0)

        # Calculate z-score and check cointegration
        self.zscore = self.calculate_zscore()
        self.cointegration = self.check_cointegration()

        # Trading logic
        if self.order:
            return

        if self.cointegration:
            if abs(self.zscore) > self.p.zscore_threshold:
                # Calculate position size
                self.position_size = self.calculate_position_size()
                
                if self.zscore > self.p.zscore_threshold:
                    # Short pair1, long pair2
                    self.order = self.sell(data=self.pair1, size=self.position_size)
                    self.order = self.buy(data=self.pair2, size=self.position_size)
                    self.entry_price = current_spread
                
                elif self.zscore < -self.p.zscore_threshold:
                    # Long pair1, short pair2
                    self.order = self.buy(data=self.pair1, size=self.position_size)
                    self.order = self.sell(data=self.pair2, size=self.position_size)
                    self.entry_price = current_spread

        # Check for exit conditions
        if self.position:
            profit_pct = (current_spread - self.entry_price) / self.entry_price
            
            if profit_pct >= self.p.take_profit or profit_pct <= -self.p.stop_loss:
                self.close()
                self.order = None
                self.position_size = 0
                self.entry_price = 0

class RealTimeDataHandler:
    def __init__(self, symbol1, symbol2):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.data_queue = queue.Queue()
        self.ws = None
        self.running = False

    def start(self):
        self.running = True
        self.ws = websocket.WebSocketApp(
            f"wss://testnet.binance.vision/ws/{self.symbol1.lower()}@trade/{self.symbol2.lower()}@trade",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def on_message(self, ws, message):
        data = json.loads(message)
        self.data_queue.put(data)

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket Connection Closed")
        if self.running:
            # Attempt to reconnect
            time.sleep(5)
            self.start()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

def get_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    # Fetch recent tweets with query related to ticker
    fetched_tweets = []  # Implement tweet fetching logic here
    sentiments = []
    for tweet in fetched_tweets:
        score = analyzer.polarity_scores(tweet)
        sentiments.append(score['compound'])
    return np.array(sentiments)

def get_historical_data(symbol, interval, limit):
    # Initialize the Binance client with testnet
    client = Client(
        api_key='',  # Testnet doesn't require API keys for public endpoints
        api_secret='',
        testnet=True  # Enable testnet
    )
    
    try:
        # Calculate start time (1 year ago)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        
        # Convert to milliseconds timestamp
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Get historical klines/candlestick data
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_str=start_ts,
            end_str=end_ts
        )
        
        # Convert the data to the required format
        data = []
        for kline in klines:
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(kline[0] / 1000)
            data.append({
                'datetime': dt,
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        # Sort data by time to ensure chronological order
        data.sort(key=lambda x: x['datetime'])
        return data
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
        return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def run_backtest():
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(StatisticalArbitrageStrategy)
    
    # Get historical data
    symbol1 = 'BTCUSDT'
    symbol2 = 'ETHUSDT'
    interval = '1m'  # Use 1-minute data for more frequent trading
    
    print("Fetching historical data...")
    data1 = get_historical_data(symbol1, interval, 1000)
    data2 = get_historical_data(symbol2, interval, 1000)
    
    if not data1 or not data2:
        print("Failed to fetch historical data")
        return
    
    # Convert to pandas DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Set datetime as index
    df1.set_index('datetime', inplace=True)
    df2.set_index('datetime', inplace=True)
    
    # Create data feeds
    data1_feed = bt.feeds.PandasData(
        dataname=df1,
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    data2_feed = bt.feeds.PandasData(
        dataname=df2,
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # Add data feeds
    cerebro.adddata(data1_feed, name='pair1')
    cerebro.adddata(data2_feed, name='pair2')
    
    # Set initial capital
    cerebro.broker.setcash(100000.0)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # Run backtest
    print("Running backtest...")
    results = cerebro.run()
    
    # Print results
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print(f'Sharpe Ratio: {results[0].analyzers.sharpe.get_analysis()}')
    print(f'Max Drawdown: {results[0].analyzers.drawdown.get_analysis()}')

if __name__ == "__main__":
    try:
        # Run backtest first
        print("Starting backtest...")
        run_backtest()
        
        # Initialize real-time data handler
        print("\nInitializing real-time trading...")
        data_handler = RealTimeDataHandler('BTCUSDT', 'ETHUSDT')
        data_handler.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping the program...")
        if 'data_handler' in locals():
            data_handler.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'data_handler' in locals():
            data_handler.stop()
