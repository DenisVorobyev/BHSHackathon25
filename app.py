from flask import Flask, render_template, jsonify, request, session
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import os
from dotenv import load_dotenv
import logging
from company_analyzer import CompanyAnalyzer
from cryptoData import getCryptoInfo
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug environment and .env file location
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
logger.info(f"Current directory: {current_dir}")
logger.info(f"Looking for .env file at: {env_path}")
logger.info(f".env file exists: {os.path.exists(env_path)}")

# Load environment variables
load_dotenv(env_path)  # Explicitly specify the path to .env

# Validate Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Log credential validation (without exposing the actual keys)
logger.info(f"API Key length: {len(ALPACA_API_KEY) if ALPACA_API_KEY else 0}")
logger.info(f"Secret Key length: {len(ALPACA_SECRET_KEY) if ALPACA_SECRET_KEY else 0}")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("Alpaca API credentials not found in environment variables")
    raise ValueError("Alpaca API credentials not configured. Please check your .env file")

# Get the absolute path to the templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
logger.debug(f"Template directory: {template_dir}")
logger.debug(f"Template directory exists: {os.path.exists(template_dir)}")
logger.debug(f"Template file exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.urandom(24)

# Initialize Alpaca clients with error handling
try:
    trading_client = TradingClient(
        paper=True,  # This automatically sets the correct paper trading endpoint
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY
    )
    logger.info("Successfully initialized Alpaca trading client")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca trading client: {str(e)}")
    raise

try:
    data_client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY
    )
    logger.info("Successfully initialized Alpaca data client")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca data client: {str(e)}")
    raise

def calculate_spread(pair1_data, pair2_data):
    return pair1_data - pair2_data

def calculate_zscore(spread_history, lookback_period=60):
    if len(spread_history) < lookback_period:
        return 0
    spread_array = np.array(spread_history[-lookback_period:])
    mean = np.mean(spread_array)
    std = np.std(spread_array)
    return (spread_history[-1] - mean) / std

def check_cointegration(pair1_data, pair2_data, lookback_period=60):
    if len(pair1_data) < lookback_period or len(pair2_data) < lookback_period:
        return False
    _, pvalue, _ = coint(
        pair1_data[-lookback_period:],
        pair2_data[-lookback_period:]
    )
    return pvalue < 0.05

@app.route('/')
def index():
    """Render the user view (crypto-focused dashboard)"""
    return render_template('index.html')

@app.route('/business')
def business():
    """Render the business view (stock buyback analysis + crypto)"""
    return render_template('business.html')

@app.route('/api/account')
def get_account():
    account = trading_client.get_account()
    return jsonify({
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'buying_power': float(account.buying_power),
        'equity': float(account.equity)
    })

@app.route('/api/positions')
def get_positions():
    try:
        positions = trading_client.get_all_positions()
        return jsonify([{
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'avg_entry_price': float(pos.avg_entry_price),
            'current_price': float(pos.current_price),
            'market_value': float(pos.market_value),
            'unrealized_pl': float(pos.unrealized_pl)
        } for pos in positions])
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch positions. Please check your Alpaca API credentials.'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_pairs():
    data = request.json
    symbol1 = data.get('symbol1', 'AAPL')
    symbol2 = data.get('symbol2', 'MSFT')
    
    # Get historical data
    end = datetime.now()
    start = end - timedelta(days=365)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol1, symbol2],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )
    
    bars = data_client.get_stock_bars(request_params)
    
    # Convert to pandas DataFrames
    df1 = bars.df.xs(symbol1, level=1)
    df2 = bars.df.xs(symbol2, level=1)
    
    # Calculate spread and statistics
    spread = calculate_spread(df1['close'], df2['close'])
    zscore = calculate_zscore(spread.tolist())
    is_cointegrated = check_cointegration(df1['close'].values, df2['close'].values)
    
    return jsonify({
        'zscore': float(zscore),
        'is_cointegrated': bool(is_cointegrated),
        'spread': float(spread.iloc[-1]),
        'pair1_price': float(df1['close'].iloc[-1]),
        'pair2_price': float(df2['close'].iloc[-1])
    })

@app.route('/api/trade', methods=['POST'])
def execute_trade():
    data = request.json
    symbol = data.get('symbol')
    side = data.get('side')
    qty = float(data.get('qty', 1))
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    
    try:
        order = trading_client.submit_order(order_data)
        return jsonify({
            'status': 'success',
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'qty': float(order.qty),
            'filled_qty': float(order.filled_qty),
            'status': order.status
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/crypto/market')
def get_crypto_market_data():
    """Get overall crypto market data"""
    try:
        # Get BTC data for market cap and dominance
        btc = yf.Ticker("BTC-USD")
        info = btc.info
        
        return jsonify({
            'total_market_cap': info.get('totalMarketCap', 0),
            'total_volume': info.get('volume24h', 0),
            'btc_dominance': info.get('marketCapDominance', 0)
        })
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return jsonify({
            'total_market_cap': 0,
            'total_volume': 0,
            'btc_dominance': 0
        })

@app.route('/api/crypto/<symbol>')
def get_crypto_data(symbol):
    """Get detailed data for a specific cryptocurrency"""
    try:
        # Map common crypto names to Yahoo Finance symbols
        symbol_map = {
            'bitcoin': 'BTC-USD',
            'btc': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'eth': 'ETH-USD',
            'solana': 'SOL-USD',
            'sol': 'SOL-USD'
        }
        
        # Get the correct symbol
        yf_symbol = symbol_map.get(symbol.lower(), f"{symbol.upper()}-USD")
        logger.info(f"Using Yahoo Finance symbol: {yf_symbol}")
        
        # Get price data from yfinance
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        
        if not info:
            raise ValueError(f"No data available for {yf_symbol}")
            
        # Get historical data
        history = ticker.history(period="1d", interval="1h")
        
        if history.empty:
            raise ValueError(f"No price history available for {yf_symbol}")
        
        # Get sentiment data from our analysis
        crypto_info = getCryptoInfo(symbol.upper())
        
        # Prepare price history data
        price_history = {
            'labels': history.index.strftime('%H:%M').tolist(),
            'prices': history['Close'].tolist()
        }
        
        return jsonify({
            'price_change_24h': info.get('priceChange24h', 0),
            'sentiment': {
                'positive': crypto_info.SentimentScore['pos'],
                'negative': crypto_info.SentimentScore['neg'],
                'neutral': crypto_info.SentimentScore['neu']
            },
            'reasoning': crypto_info.Reasoning,
            'price_history': price_history
        })
    except Exception as e:
        logger.error(f"Error getting {symbol} data: {str(e)}")
        return jsonify({
            'price_change_24h': 0,
            'sentiment': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            },
            'reasoning': f"Error analyzing {symbol}: {str(e)}",
            'price_history': {'labels': [], 'prices': []}
        })

@app.route('/api/analyze_company', methods=['POST'])
def analyze_company():
    """Analyze company for buyback potential"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
            
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({
                'status': 'error',
                'message': 'Symbol is required'
            }), 400
        
        logger.debug(f"Starting company analysis for symbol: {symbol}")
        
        # Validate symbol format
        if not symbol.isalpha():
            return jsonify({
                'status': 'error',
                'message': 'Invalid symbol format. Please use only letters (e.g., AAPL, MSFT)'
            }), 400
            
        try:
            analyzer = CompanyAnalyzer(symbol)
            recommendation = analyzer.get_recommendation()
            
            if not recommendation:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not analyze company. Please ensure the symbol is valid and the company has sufficient public data available.'
                }), 500
                
            return jsonify({
                'status': 'success',
                'data': recommendation
            })
            
        except ValueError as ve:
            logger.error(f"Validation error for {symbol}: {str(ve)}")
            return jsonify({
                'status': 'error',
                'message': str(ve)
            }), 400
            
    except Exception as e:
        logger.error(f"Error analyzing company: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing company: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 