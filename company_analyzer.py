import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import random
from functools import lru_cache
import json
from typing import Optional, Dict, Any, List
from queue import Queue
from threading import Lock
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.max_retries = 5
        self.base_delay = 30
        self._window_start = time.time()
        self._max_requests_per_window = 10
        self._window_size = 60
        self._request_queue = Queue()
        self._rate_limit_lock = Lock()
        self._consecutive_failures = 0
        self._last_request_time = 0
        self._min_request_interval = 5
        
        # Initialize yfinance Ticker with validation
        self.company = self._initialize_ticker()
        self._validate_ticker()
        logger.info(f"Initialized analyzer for {self.symbol}")

    def _initialize_ticker(self):
        """Initialize yfinance Ticker with rate limiting"""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                logger.info(f"Initializing Ticker for {self.symbol} (attempt {attempt + 1}/{self.max_retries})")
                
                # Initialize the Ticker
                ticker = yf.Ticker(self.symbol)
                
                # Test if we can get basic info
                logger.debug(f"Checking stock data for {self.symbol}")
                info = ticker.info
                if not info:
                    logger.error(f"No info data available for {self.symbol}")
                    raise ValueError("No info data available")
                
                logger.info(f"Successfully initialized Ticker for {self.symbol}")
                return ticker
                
            except Exception as e:
                logger.error(f"Initialization error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = random.uniform(2, 5)
                    logger.info(f"Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                else:
                    raise ValueError(f"Failed to initialize {self.symbol} after {self.max_retries} attempts: {str(e)}")

    def _validate_ticker(self):
        """Validate the ticker using info data"""
        logger.info(f"Validating ticker for {self.symbol}")
        try:
            if self.company.info:
                logger.info("Validation successful using info data")
                return
        except Exception as e:
            logger.error(f"Error in info validation: {str(e)}")
        
        raise ValueError(f"Invalid symbol: {self.symbol} - Info data not available")

    def _rate_limit(self):
        """Improved rate limiting with dynamic backoff"""
        with self._rate_limit_lock:
            now = time.time()
            
            # Ensure minimum time between requests
            time_since_last = now - self._last_request_time
            if time_since_last < self._min_request_interval:
                time.sleep(self._min_request_interval - time_since_last)
            
            # Check window limits
            elapsed = now - self._window_start
            if elapsed > self._window_size:
                self._window_start = now
                self._consecutive_failures = 0
                return
            
            # Calculate adaptive delay based on consecutive failures
            delay = max(
                self.base_delay * (2 ** self._consecutive_failures),
                random.uniform(1, 3)  # Minimum jitter
            )
            time.sleep(delay)
            self._last_request_time = time.time()

    @lru_cache(maxsize=100)
    def _cached_request(self, method_name, *args, **kwargs):
        """Generic cached request handler"""
        try:
            self._rate_limit()
            method = getattr(self.company, method_name)
            result = method(*args, **kwargs)
            
            # Validate result
            if not result or self.symbol not in result:
                raise ValueError(f"Empty result from {method_name}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in {method_name}: {str(e)}")
            self._consecutive_failures += 1
            raise

    def get_company_info(self) -> Dict[str, Any]:
        """Get basic company information"""
        try:
            info = self.company.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            logger.error(f"Error getting company info: {str(e)}")
            return {}

    def get_financial_metrics(self) -> Dict[str, float]:
        """Get key financial metrics"""
        try:
            info = self.company.info
            return {
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0)
            }
        except Exception as e:
            logger.error(f"Error getting financial metrics: {str(e)}")
            return {}

    def get_stock_price_data(self, period: str = "1y") -> pd.DataFrame:
        """Get historical stock price data"""
        try:
            # Get historical data
            history = self.company.history(period=period)
            
            # Get current price from the most recent data point
            current_price = history['Close'].iloc[-1] if not history.empty else 0
            
            # Add current price to the DataFrame
            history['current_price'] = current_price
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting stock price data: {str(e)}")
            # Return a minimal DataFrame with current price
            return pd.DataFrame({
                'date': [datetime.now()],
                'close': [0],
                'current_price': [0]
            })

    def analyze_buyback_potential(self) -> Dict[str, Any]:
        """Analyze company's buyback potential"""
        try:
            info = self.company.info
            market_cap = info.get('marketCap', 0)
            total_cash = info.get('totalCash', 0)
            free_cash_flow = info.get('freeCashflow', 0)
            
            if market_cap == 0 or total_cash == 0 or free_cash_flow == 0:
                raise ValueError(f"Missing financial data - Market Cap: {market_cap}, Total Cash: {total_cash}, FCF: {free_cash_flow}")
            
            # Calculate buyback metrics
            cash_to_market_cap = total_cash / market_cap if market_cap > 0 else 0
            fcf_to_market_cap = free_cash_flow / market_cap if market_cap > 0 else 0
            
            return {
                'market_cap': market_cap,
                'total_cash': total_cash,
                'free_cash_flow': free_cash_flow,
                'cash_to_market_cap_ratio': cash_to_market_cap,
                'fcf_to_market_cap_ratio': fcf_to_market_cap,
                'buyback_potential': 'High' if cash_to_market_cap > 0.1 or fcf_to_market_cap > 0.05 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing buyback potential: {str(e)}", exc_info=True)
            return {}

    def get_recommendation(self) -> Dict[str, Any]:
        """Get buyback recommendation with detailed analysis"""
        try:
            # Get all required data
            analysis = self.analyze_buyback_potential()
            if not analysis:
                return {'error': 'Analysis failed - insufficient data available'}

            # Calculate recommendation score (0-100)
            score = 0
            
            # Cash position (max 40 points)
            cash_ratio = analysis.get('cash_to_market_cap_ratio', 0)
            score += min(40, cash_ratio * 400)  # 10% = 40 points
            
            # FCF yield (max 30 points)
            fcf_ratio = analysis.get('fcf_to_market_cap_ratio', 0)
            score += min(30, fcf_ratio * 300)  # 10% = 30 points
            
            # Get financial metrics for additional scoring
            metrics = self.get_financial_metrics()
            
            # Profitability (max 15 points)
            profit_margin = metrics.get('profit_margin', 0)
            score += min(15, profit_margin * 150)  # 10% = 15 points
            
            # Financial health (max 15 points)
            debt_ratio = metrics.get('debt_to_equity', 0)
            if debt_ratio < 0.5:  # Low debt
                score += 15
            elif debt_ratio < 1.0:  # Moderate debt
                score += 10
            elif debt_ratio < 2.0:  # High debt
                score += 5
            
            # Get recommendation text based on score
            recommendation = self._get_recommendation_text(score)
            
            return {
                'recommendation': recommendation,
                'score': min(100, int(score)),
                'metrics': analysis,
                'financial_health': {
                    'cash_position': 'Strong' if cash_ratio > 0.1 else 'Moderate' if cash_ratio > 0.05 else 'Weak',
                    'cash_flow': 'Strong' if fcf_ratio > 0.05 else 'Moderate' if fcf_ratio > 0.02 else 'Weak',
                    'profitability': 'Strong' if profit_margin > 0.15 else 'Moderate' if profit_margin > 0.1 else 'Weak',
                    'debt_level': 'Low' if debt_ratio < 0.5 else 'Moderate' if debt_ratio < 1.0 else 'High'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendation: {str(e)}")
            return {'error': f'Failed to generate recommendation: {str(e)}'}

    def _get_recommendation_text(self, score: float) -> str:
        """Get recommendation text based on score"""
        if score >= 80:
            return "Strong Buyback Candidate - Excellent financial position for buybacks"
        elif score >= 60:
            return "Favorable Buyback Conditions - Good financial position for buybacks"
        elif score >= 40:
            return "Neutral - Monitor Cash Position - Consider buybacks if conditions improve"
        else:
            return "Not Recommended for Buybacks - Focus on strengthening financial position"

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting company analysis...")
        analyzer = CompanyAnalyzer("AAPL")
        logger.info("Successfully initialized analyzer")
        
        logger.info("Getting company info...")
        company_info = analyzer.get_company_info()
        logger.info(f"Company info: {json.dumps(company_info, indent=2)}")
        
        logger.info("Getting financial metrics...")
        financial_metrics = analyzer.get_financial_metrics()
        logger.info(f"Financial metrics: {json.dumps(financial_metrics, indent=2)}")
        
        logger.info("Getting buyback analysis...")
        buyback_analysis = analyzer.analyze_buyback_potential()
        logger.info(f"Buyback analysis: {json.dumps(buyback_analysis, indent=2)}")
        
        logger.info("Getting final recommendation...")
        result = analyzer.get_recommendation()
        logger.info(f"Final recommendation: {json.dumps(result, indent=2)}")
        
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Failed to analyze company: {str(e)}", exc_info=True)
        print(f"Error analyzing company: {str(e)}")