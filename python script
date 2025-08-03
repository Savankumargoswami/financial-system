"""
Autonomous Financial Risk Management System - Budget Version
Optimized for DigitalOcean deployment with Yahoo Finance
"""

import os
import sys
import time
import logging
import asyncio
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

# Third-party imports
import numpy as np
import pandas as pd
import yfinance as yf
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://trader:password@localhost:5432/financial_system')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Trading
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    SYMBOLS = os.getenv('SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN,TSLA').split(',')
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    STOP_LOSS_THRESHOLD = float(os.getenv('STOP_LOSS_THRESHOLD', '-0.15'))
    
    # Intervals (seconds)
    DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', '300'))
    DECISION_INTERVAL = int(os.getenv('DECISION_INTERVAL', '300'))
    RISK_CHECK_INTERVAL = int(os.getenv('RISK_CHECK_INTERVAL', '60'))

# Data structures
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    rsi: float = 50.0
    macd: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0

@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    confidence: float
    reasoning: str

# Database manager
class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL, pool_size=10, max_overflow=20)
        
    def get_connection(self):
        return psycopg2.connect(Config.DATABASE_URL, cursor_factory=RealDictCursor)
        
    def insert_market_data(self, data: MarketData):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO market_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, adjusted_close)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    adjusted_close = EXCLUDED.adjusted_close
                """, (
                    data.symbol, data.timestamp, data.open_price, data.high_price,
                    data.low_price, data.close_price, data.volume, data.close_price
                ))
            conn.commit()
    
    def get_latest_prices(self) -> Dict[str, float]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (symbol) symbol, close_price
                    FROM market_data
                    ORDER BY symbol, timestamp DESC
                """)
                return {row['symbol']: float(row['close_price']) for row in cur.fetchall()}
    
    def get_portfolio_positions(self) -> List[Position]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM portfolio_positions")
                positions = []
                for row in cur.fetchall():
                    positions.append(Position(
                        symbol=row['symbol'],
                        quantity=float(row['quantity']),
                        average_price=float(row['average_price']),
                        current_price=float(row['current_price'] or 0),
                        market_value=float(row['market_value'] or 0),
                        unrealized_pnl=float(row['unrealized_pnl'] or 0),
                        weight=float(row['weight'] or 0)
                    ))
                return positions

# Data collector using Yahoo Finance
class DataCollector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        
    def collect_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Collect real-time market data from Yahoo Finance"""
        market_data = {}
        
        try:
            # Download data for all symbols at once (more efficient)
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    
                    # Get latest data
                    hist = ticker.history(period='2d', interval='1m')
                    if hist.empty:
                        logger.warning(f"No data for {symbol}")
                        continue
                        
                    latest = hist.iloc[-1]
                    timestamp = hist.index[-1].to_pydatetime()
                    
                    # Calculate technical indicators
                    rsi = self.calculate_rsi(hist['Close'].values)
                    macd = self.calculate_macd(hist['Close'].values)
                    bb_upper, bb_lower = self.calculate_bollinger_bands(hist['Close'].values)
                    
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(latest['Open']),
                        high_price=float(latest['High']),
                        low_price=float(latest['Low']),
                        close_price=float(latest['Close']),
                        volume=int(latest['Volume']),
                        rsi=rsi,
                        macd=macd,
                        bb_upper=bb_upper,
                        bb_lower=bb_lower
                    )
                    
                    # Store in database
                    self.db_manager.insert_market_data(market_data[symbol])
                    
                    # Cache in Redis for quick access
                    self.redis_client.setex(
                        f"market_data:{symbol}",
                        300,  # 5 minute expiry
                        json.dumps(asdict(market_data[symbol]), default=str)
                    )
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            
        logger.info(f"Collected data for {len(market_data)} symbols")
        return market_data
    
    def calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI technical indicator"""
        if len(prices) < window + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD technical indicator"""
        if len(prices) < slow:
            return 0.0
            
        exp_fast = pd.Series(prices).ewm(span=fast).mean()
        exp_slow = pd.Series(prices).ewm(span=slow).mean()
        macd = exp_fast.iloc[-1] - exp_slow.iloc[-1]
        return float(macd)
    
    def calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < window:
            price = prices[-1]
            return float(price * 1.02), float(price * 0.98)
            
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return float(upper_band), float(lower_band)

# Simple Neural Network for predictions
class SimplePredictor(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        return self.network(x)

# Strategy Agent
class StrategyAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        self.predictor = SimplePredictor()
        
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """Generate trading signals based on market data"""
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # Simple strategy combining multiple indicators
                signal = self.analyze_symbol(data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                
        return signals
    
    def analyze_symbol(self, data: MarketData) -> Optional[TradingSignal]:
        """Analyze individual symbol and generate signal"""
        # Multi-factor analysis
        factors = []
        reasoning = []
        
        # RSI Analysis
        if data.rsi < 30:
            factors.append(0.3)  # Oversold - bullish
            reasoning.append("RSI oversold")
        elif data.rsi > 70:
            factors.append(-0.3)  # Overbought - bearish
            reasoning.append("RSI overbought")
        else:
            factors.append(0)
            
        # MACD Analysis
        if data.macd > 0:
            factors.append(0.2)  # Positive momentum
            reasoning.append("MACD positive")
        else:
            factors.append(-0.2)  # Negative momentum
            reasoning.append("MACD negative")
            
        # Bollinger Bands Analysis
        if data.close_price < data.bb_lower:
            factors.append(0.2)  # Below lower band - bullish
            reasoning.append("Below Bollinger lower band")
        elif data.close_price > data.bb_upper:
            factors.append(-0.2)  # Above upper band - bearish
            reasoning.append("Above Bollinger upper band")
        else:
            factors.append(0)
            
        # Volume Analysis (simplified)
        if data.volume > 1000000:  # High volume
            factors.append(0.1)
            reasoning.append("High volume")
        else:
            factors.append(-0.1)
            
        # Calculate final signal
        signal_strength = sum(factors)
        confidence = min(abs(signal_strength), 1.0)
        
        # Only generate signals with minimum confidence
        if confidence < 0.3:
            return None
            
        # Determine action
        if signal_strength > 0.3:
            action = "BUY"
            quantity = min(confidence * Config.MAX_POSITION_SIZE * Config.INITIAL_CAPITAL / data.close_price, 100)
        elif signal_strength < -0.3:
            action = "SELL"
            quantity = 50  # Fixed sell quantity for now
        else:
            return None
            
        return TradingSignal(
            symbol=data.symbol,
            action=action,
            quantity=quantity,
            confidence=confidence,
            reasoning="; ".join(reasoning)
        )

# Risk Manager
class RiskManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def check_risk(self, signals: List[TradingSignal], positions: List[Position]) -> List[TradingSignal]:
        """Filter signals based on risk criteria"""
        filtered_signals = []
        
        current_positions = {pos.symbol: pos for pos in positions}
        total_portfolio_value = sum(pos.market_value for pos in positions) + Config.INITIAL_CAPITAL
        
        for signal in signals:
            # Check position size limits
            if signal.action == "BUY":
                position_value = signal.quantity * self.get_current_price(signal.symbol)
                position_weight = position_value / total_portfolio_value
                
                if position_weight > Config.MAX_POSITION_SIZE:
                    logger.warning(f"Position size too large for {signal.symbol}: {position_weight:.2%}")
                    continue
                    
            # Check stop loss for existing positions
            if signal.symbol in current_positions:
                position = current_positions[signal.symbol]
                pnl_percent = (position.current_price - position.average_price) / position.average_price
                
                if pnl_percent < Config.STOP_LOSS_THRESHOLD:
                    # Force sell signal due to stop loss
                    signal.action = "SELL"
                    signal.quantity = position.quantity
                    signal.reasoning = f"Stop loss triggered: {pnl_percent:.2%}"
                    
            filtered_signals.append(signal)
            
        return filtered_signals
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        prices = self.db_manager.get_latest_prices()
        return prices.get(symbol, 100.0)  # Default fallback

# Portfolio Manager
class PortfolioManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def execute_signals(self, signals: List[TradingSignal]):
        """Execute trading signals (paper trading)"""
        for signal in signals:
            try:
                if Config.PAPER_TRADING:
                    self.execute_paper_trade(signal)
                else:
                    self.execute_real_trade(signal)
                    
                logger.info(f"Executed {signal.action} {signal.quantity} {signal.symbol}")
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def execute_paper_trade(self, signal: TradingSignal):
        """Execute paper trade (simulation)"""
        current_price = self.get_current_price(signal.symbol)
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Record the trade order
                cur.execute("""
                    INSERT INTO trade_orders 
                    (symbol, order_type, quantity, price, executed_price, status, executed_at)
                    VALUES (%s, %s, %s, %s, %s, 'EXECUTED', %s)
                """, (
                    signal.symbol, signal.action, signal.quantity, 
                    current_price, current_price, datetime.now()
                ))
                
                # Update portfolio position
                if signal.action == "BUY":
                    cur.execute("""
                        INSERT INTO portfolio_positions 
                        (symbol, quantity, average_price, current_price, market_value, unrealized_pnl, weight)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                        quantity = portfolio_positions.quantity + EXCLUDED.quantity,
                        average_price = (portfolio_positions.average_price * portfolio_positions.quantity + 
                                       EXCLUDED.average_price * EXCLUDED.quantity) / 
                                       (portfolio_positions.quantity + EXCLUDED.quantity),
                        current_price = EXCLUDED.current_price,
                        market_value = (portfolio_positions.quantity + EXCLUDED.quantity) * EXCLUDED.current_price,
                        updated_at = CURRENT_TIMESTAMP
                    """, (
                        signal.symbol, signal.quantity, current_price, current_price,
                        signal.quantity * current_price, 0.0, 0.1
                    ))
                elif signal.action == "SELL":
                    cur.execute("""
                        UPDATE portfolio_positions 
                        SET quantity = GREATEST(0, quantity - %s),
                            current_price = %s,
                            market_value = GREATEST(0, quantity - %s) * %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = %s
                    """, (signal.quantity, current_price, signal.quantity, current_price, signal.symbol))
                    
                    # Remove position if quantity becomes 0
                    cur.execute("DELETE FROM portfolio_positions WHERE symbol = %s AND quantity <= 0", (signal.symbol,))
            
            conn.commit()
    
    def execute_real_trade(self, signal: TradingSignal):
        """Execute real trade (placeholder for broker integration)"""
        logger.info(f"Real trading not implemented yet: {signal.action} {signal.quantity} {signal.symbol}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        prices = self.db_manager.get_latest_prices()
        return prices.get(symbol, 100.0)
    
    def update_portfolio_values(self):
        """Update current portfolio values"""
        latest_prices = self.db_manager.get_latest_prices()
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                for symbol, price in latest_prices.items():
                    cur.execute("""
                        UPDATE portfolio_positions 
                        SET current_price = %s,
                            market_value = quantity * %s,
                            unrealized_pnl = (quantity * %s) - (quantity * average_price),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = %s
                    """, (price, price, price, symbol))
                
                # Update portfolio weights
                cur.execute("SELECT SUM(market_value) FROM portfolio_positions")
                total_value = cur.fetchone()[0] or 1
                
                cur.execute("""
                    UPDATE portfolio_positions 
                    SET weight = market_value / %s
                    WHERE market_value > 0
                """, (total_value,))
            
            conn.commit()

# Main System Controller
class FinancialSystem:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_collector = DataCollector(self.db_manager)
        self.strategy_agent = StrategyAgent(self.db_manager)
        self.risk_manager = RiskManager(self.db_manager)
        self.portfolio_manager = PortfolioManager(self.db_manager)
        
        self.running = False
        self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        
    def start(self):
        """Start the financial system"""
        logger.info("Starting Autonomous Financial System...")
        self.running = True
        
        # Start background threads
        threading.Thread(target=self.data_collection_loop, daemon=True).start()
        threading.Thread(target=self.trading_loop, daemon=True).start()
        threading.Thread(target=self.risk_monitoring_loop, daemon=True).start()
        
        logger.info("System started successfully!")
    
    def stop(self):
        """Stop the financial system"""
        logger.info("Stopping system...")
        self.running = False
    
    def data_collection_loop(self):
        """Continuous data collection loop"""
        while self.running:
            try:
                market_data = self.data_collector.collect_market_data(Config.SYMBOLS)
                self.redis_client.setex("last_data_update", 3600, datetime.now().isoformat())
                
                time.sleep(Config.DATA_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def trading_loop(self):
        """Main trading decision loop"""
        while self.running:
            try:
                # Get latest market data
                market_data = {}
                for symbol in Config.SYMBOLS:
                    cached_data = self.redis_client.get(f"market_data:{symbol}")
                    if cached_data:
                        data_dict = json.loads(cached_data)
                        data_dict['timestamp'] = datetime.fromisoformat(data_dict['timestamp'])
                        market_data[symbol] = MarketData(**data_dict)
                
                if not market_data:
                    logger.warning("No market data available for trading decisions")
                    time.sleep(Config.DECISION_INTERVAL)
                    continue
                
                # Generate trading signals
                signals = self.strategy_agent.generate_signals(market_data)
                logger.info(f"Generated {len(signals)} trading signals")
                
                # Apply risk management
                positions = self.db_manager.get_portfolio_positions()
                filtered_signals = self.risk_manager.check_risk(signals, positions)
                logger.info(f"Risk-filtered signals: {len(filtered_signals)}")
                
                # Execute trades
                if filtered_signals:
                    self.portfolio_manager.execute_signals(filtered_signals)
                
                # Update portfolio values
                self.portfolio_manager.update_portfolio_values()
                
                # Store system metrics
                self.record_system_metrics()
                
                time.sleep(Config.DECISION_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def risk_monitoring_loop(self):
        """Continuous risk monitoring"""
        while self.running:
            try:
                positions = self.db_manager.get_portfolio_positions()
                
                # Calculate portfolio metrics
                total_value = sum(pos.market_value for pos in positions)
                total_pnl = sum(pos.unrealized_pnl for pos in positions)
                
                # Check for risk violations
                if total_pnl / Config.INITIAL_CAPITAL < -0.20:  # 20% loss limit
                    logger.critical("EMERGENCY: Portfolio loss exceeds 20%!")
                    self.send_alert("Emergency stop - portfolio loss > 20%")
                
                # Log current status
                logger.info(f"Portfolio value: ${total_value:,.2f}, PnL: ${total_pnl:,.2f}")
                
                time.sleep(Config.RISK_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                time.sleep(60)
    
    def record_system_metrics(self):
        """Record system performance metrics"""
        try:
            positions = self.db_manager.get_portfolio_positions()
            total_value = sum(pos.market_value for pos in positions)
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            cash_balance = Config.INITIAL_CAPITAL - (total_value - total_pnl)
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO risk_metrics 
                        (portfolio_value, cash_balance, total_pnl, daily_pnl, num_positions)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (total_value, cash_balance, total_pnl, 0, len(positions)))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    def send_alert(self, message: str):
        """Send alert notification"""
        logger.critical(f"ALERT: {message}")
        # Email/SMS implementation would go here
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        positions = self.db_manager.get_portfolio_positions()
        latest_prices = self.db_manager.get_latest_prices()
        
        return {
            "running": self.running,
            "positions": len(positions),
            "portfolio_value": sum(pos.market_value for pos in positions),
            "total_pnl": sum(pos.unrealized_pnl for pos in positions),
            "latest_prices": latest_prices,
            "last_update": datetime.now().isoformat()
        }

# Web Dashboard using FastAPI
app = FastAPI(title="Autonomous Financial System", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Global system instance
financial_system = None

@app.on_event("startup")
async def startup_event():
    global financial_system
    financial_system = FinancialSystem()
    financial_system.start()

@app.on_event("shutdown")
async def shutdown_event():
    if financial_system:
        financial_system.stop()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def get_status():
    """Get system status API"""
    if financial_system:
        return financial_system.get_system_status()
    return {"error": "System not initialized"}

@app.get("/api/positions")
async def get_positions():
    """Get current portfolio positions"""
    if financial_system:
        positions = financial_system.db_manager.get_portfolio_positions()
        return [asdict(pos) for pos in positions]
    return []

@app.get("/api/prices")
async def get_prices():
    """Get latest market prices"""
    if financial_system:
        return financial_system.db_manager.get_latest_prices()
    return {}

@app.post("/api/emergency_stop")
async def emergency_stop():
    """Emergency stop endpoint"""
    if financial_system:
        financial_system.stop()
        return {"message": "Emergency stop activated"}
    return {"error": "System not running"}

if __name__ == "__main__":
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        if financial_system:
            financial_system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ensure directories exist
    os.makedirs("/app/logs", exist_ok=True)
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/templates", exist_ok=True)
    
    # Start the web server
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
