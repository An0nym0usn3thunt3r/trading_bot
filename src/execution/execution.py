"""
Execution module for Nasdaq-100 E-mini futures trading bot.
This module handles order execution, smart order routing, and broker integration.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import datetime
import time
import uuid
import threading
import queue
from enum import Enum
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enum for order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Enum for order sides."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Enum for order statuses."""
    CREATED = "CREATED"
    PENDING = "PENDING"
    SENT = "SENT"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class TimeInForce(Enum):
    """Enum for time in force options."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class Order:
    """
    Class representing a trading order.
    """
    
    def __init__(self, symbol, order_type, side, quantity, price=None, stop_price=None,
                 time_in_force=TimeInForce.DAY, strategy_id=None, order_id=None):
        """
        Initialize an order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        order_type : OrderType
            Type of order.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Order quantity.
        price : float, optional
            Limit price for limit orders.
        stop_price : float, optional
            Stop price for stop orders.
        time_in_force : TimeInForce, optional
            Time in force for the order.
        strategy_id : str, optional
            ID of the strategy that generated the order.
        order_id : str, optional
            Unique order ID. If None, a UUID will be generated.
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.strategy_id = strategy_id
        self.order_id = order_id if order_id else str(uuid.uuid4())
        
        self.status = OrderStatus.CREATED
        self.filled_quantity = 0
        self.average_fill_price = 0.0
        self.commission = 0.0
        self.creation_time = datetime.datetime.now()
        self.update_time = self.creation_time
        self.broker_order_id = None
        self.error_message = None
        
        # Validate order
        self._validate()
        
        logger.info(f"Created order: {self}")
    
    def _validate(self):
        """Validate the order parameters."""
        # Check required fields
        if not self.symbol:
            raise ValueError("Symbol is required")
        
        if not isinstance(self.order_type, OrderType):
            raise ValueError("order_type must be an OrderType")
        
        if not isinstance(self.side, OrderSide):
            raise ValueError("side must be an OrderSide")
        
        if not isinstance(self.time_in_force, TimeInForce):
            raise ValueError("time_in_force must be a TimeInForce")
        
        # Check quantity
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Check price for limit orders
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError("Price is required for limit orders")
        
        # Check stop price for stop orders
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop price is required for stop orders")
    
    def update(self, status=None, filled_quantity=None, average_fill_price=None,
               commission=None, broker_order_id=None, error_message=None):
        """
        Update the order with new information.
        
        Parameters:
        -----------
        status : OrderStatus, optional
            New order status.
        filled_quantity : int, optional
            Filled quantity.
        average_fill_price : float, optional
            Average fill price.
        commission : float, optional
            Commission paid.
        broker_order_id : str, optional
            Broker's order ID.
        error_message : str, optional
            Error message if any.
        """
        if status:
            self.status = status
        
        if filled_quantity is not None:
            self.filled_quantity = filled_quantity
        
        if average_fill_price is not None:
            self.average_fill_price = average_fill_price
        
        if commission is not None:
            self.commission = commission
        
        if broker_order_id:
            self.broker_order_id = broker_order_id
        
        if error_message:
            self.error_message = error_message
        
        self.update_time = datetime.datetime.now()
        
        logger.info(f"Updated order {self.order_id}: status={self.status}, "
                   f"filled={self.filled_quantity}/{self.quantity}")
    
    def is_complete(self):
        """
        Check if the order is complete (filled, canceled, rejected, or error).
        
        Returns:
        --------
        bool
            True if the order is complete, False otherwise.
        """
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.ERROR]
    
    def to_dict(self):
        """
        Convert the order to a dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of the order.
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'strategy_id': self.strategy_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'creation_time': self.creation_time.isoformat(),
            'update_time': self.update_time.isoformat(),
            'broker_order_id': self.broker_order_id,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, order_dict):
        """
        Create an order from a dictionary.
        
        Parameters:
        -----------
        order_dict : dict
            Dictionary representation of the order.
        
        Returns:
        --------
        Order
            Created order.
        """
        order = cls(
            symbol=order_dict['symbol'],
            order_type=OrderType(order_dict['order_type']),
            side=OrderSide(order_dict['side']),
            quantity=order_dict['quantity'],
            price=order_dict.get('price'),
            stop_price=order_dict.get('stop_price'),
            time_in_force=TimeInForce(order_dict.get('time_in_force', 'DAY')),
            strategy_id=order_dict.get('strategy_id'),
            order_id=order_dict.get('order_id')
        )
        
        # Update other fields
        order.status = OrderStatus(order_dict.get('status', 'CREATED'))
        order.filled_quantity = order_dict.get('filled_quantity', 0)
        order.average_fill_price = order_dict.get('average_fill_price', 0.0)
        order.commission = order_dict.get('commission', 0.0)
        order.creation_time = datetime.datetime.fromisoformat(order_dict.get('creation_time', datetime.datetime.now().isoformat()))
        order.update_time = datetime.datetime.fromisoformat(order_dict.get('update_time', datetime.datetime.now().isoformat()))
        order.broker_order_id = order_dict.get('broker_order_id')
        order.error_message = order_dict.get('error_message')
        
        return order
    
    def __str__(self):
        """String representation of the order."""
        return (f"Order(id={self.order_id}, symbol={self.symbol}, type={self.order_type.value}, "
                f"side={self.side.value}, quantity={self.quantity}, price={self.price}, "
                f"status={self.status.value}, filled={self.filled_quantity})")


class Fill:
    """
    Class representing a trade fill.
    """
    
    def __init__(self, order_id, symbol, side, quantity, price, timestamp=None, commission=0.0, fill_id=None):
        """
        Initialize a fill.
        
        Parameters:
        -----------
        order_id : str
            ID of the order that was filled.
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of the fill (buy or sell).
        quantity : int
            Fill quantity.
        price : float
            Fill price.
        timestamp : datetime.datetime, optional
            Fill timestamp. If None, current time is used.
        commission : float, optional
            Commission paid.
        fill_id : str, optional
            Unique fill ID. If None, a UUID will be generated.
        """
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp if timestamp else datetime.datetime.now()
        self.commission = commission
        self.fill_id = fill_id if fill_id else str(uuid.uuid4())
        
        logger.info(f"Created fill: {self}")
    
    def to_dict(self):
        """
        Convert the fill to a dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of the fill.
        """
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, OrderSide) else self.side,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission
        }
    
    @classmethod
    def from_dict(cls, fill_dict):
        """
        Create a fill from a dictionary.
        
        Parameters:
        -----------
        fill_dict : dict
            Dictionary representation of the fill.
        
        Returns:
        --------
        Fill
            Created fill.
        """
        side = fill_dict['side']
        if not isinstance(side, OrderSide):
            side = OrderSide(side)
        
        return cls(
            order_id=fill_dict['order_id'],
            symbol=fill_dict['symbol'],
            side=side,
            quantity=fill_dict['quantity'],
            price=fill_dict['price'],
            timestamp=datetime.datetime.fromisoformat(fill_dict['timestamp']),
            commission=fill_dict.get('commission', 0.0),
            fill_id=fill_dict.get('fill_id')
        )
    
    def __str__(self):
        """String representation of the fill."""
        side_value = self.side.value if isinstance(self.side, OrderSide) else self.side
        return (f"Fill(id={self.fill_id}, order_id={self.order_id}, symbol={self.symbol}, "
                f"side={side_value}, quantity={self.quantity}, price={self.price})")


class BrokerBase(ABC):
    """
    Abstract base class for broker implementations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the broker.
        
        Parameters:
        -----------
        config : dict, optional
            Broker configuration.
        """
        self.config = config or {}
        self.connected = False
        self.orders = {}  # Dictionary of orders by order_id
        self.fills = []  # List of fills
        
        # Callbacks
        self.on_order_update = None
        self.on_fill = None
        self.on_account_update = None
        self.on_position_update = None
        self.on_error = None
    
    @abstractmethod
    def connect(self):
        """Connect to the broker."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def place_order(self, order):
        """
        Place an order with the broker.
        
        Parameters:
        -----------
        order : Order
            Order to place.
        
        Returns:
        --------
        bool
            True if the order was successfully placed, False otherwise.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
        --------
        dict
            Account information.
        """
        pass
    
    @abstractmethod
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
        --------
        dict
            Dictionary of positions by symbol.
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id):
        """
        Get the status of an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order.
        
        Returns:
        --------
        OrderStatus
            Status of the order.
        """
        pass
    
    def register_callbacks(self, on_order_update=None, on_fill=None, on_account_update=None,
                          on_position_update=None, on_error=None):
        """
        Register callbacks for broker events.
        
        Parameters:
        -----------
        on_order_update : callable, optional
            Callback for order updates.
        on_fill : callable, optional
            Callback for fills.
        on_account_update : callable, optional
            Callback for account updates.
        on_position_update : callable, optional
            Callback for position updates.
        on_error : callable, optional
            Callback for errors.
        """
        self.on_order_update = on_order_update
        self.on_fill = on_fill
        self.on_account_update = on_account_update
        self.on_position_update = on_position_update
        self.on_error = on_error
    
    def _handle_order_update(self, order):
        """
        Handle an order update.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        # Store the order
        self.orders[order.order_id] = order
        
        # Call the callback if registered
        if self.on_order_update:
            self.on_order_update(order)
    
    def _handle_fill(self, fill):
        """
        Handle a fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Store the fill
        self.fills.append(fill)
        
        # Update the order
        if fill.order_id in self.orders:
            order = self.orders[fill.order_id]
            
            # Update filled quantity
            new_filled_quantity = order.filled_quantity + fill.quantity
            
            # Calculate new average fill price
            if new_filled_quantity > 0:
                new_avg_price = ((order.filled_quantity * order.average_fill_price) + 
                                (fill.quantity * fill.price)) / new_filled_quantity
            else:
                new_avg_price = 0.0
            
            # Update order
            order.update(
                filled_quantity=new_filled_quantity,
                average_fill_price=new_avg_price,
                commission=order.commission + fill.commission
            )
            
            # Check if order is fully filled
            if order.filled_quantity >= order.quantity:
                order.update(status=OrderStatus.FILLED)
        
        # Call the callback if registered
        if self.on_fill:
            self.on_fill(fill)
    
    def _handle_error(self, error_msg, order_id=None):
        """
        Handle an error.
        
        Parameters:
        -----------
        error_msg : str
            Error message.
        order_id : str, optional
            ID of the order that caused the error.
        """
        logger.error(f"Broker error: {error_msg}")
        
        # Update the order if provided
        if order_id and order_id in self.orders:
            order = self.orders[order_id]
            order.update(status=OrderStatus.ERROR, error_message=error_msg)
        
        # Call the callback if registered
        if self.on_error:
            self.on_error(error_msg, order_id)


class SimulatedBroker(BrokerBase):
    """
    Simulated broker for backtesting and development.
    """
    
    def __init__(self, config=None):
        """
        Initialize the simulated broker.
        
        Parameters:
        -----------
        config : dict, optional
            Broker configuration.
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault('initial_balance', 100000.0)
        self.config.setdefault('commission_rate', 0.0001)  # 0.01%
        self.config.setdefault('slippage', 0.0001)  # 0.01%
        self.config.setdefault('delay_mean', 0.1)  # 100ms
        self.config.setdefault('delay_std', 0.05)  # 50ms
        self.config.setdefault('fill_probability', 0.95)  # 95% fill probability
        self.config.setdefault('partial_fill_probability', 0.2)  # 20% partial fill probability
        
        # Initialize account and positions
        self.account = {
            'balance': self.config['initial_balance'],
            'equity': self.config['initial_balance'],
            'margin_used': 0.0,
            'margin_available': self.config['initial_balance']
        }
        
        self.positions = {}  # Dictionary of positions by symbol
        
        # Market data
        self.market_data = {}  # Dictionary of market data by symbol
        
        # Order processing thread
        self.order_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
    
    def connect(self):
        """Connect to the simulated broker."""
        if self.connected:
            logger.warning("Already connected to simulated broker")
            return True
        
        self.connected = True
        self.running = True
        
        # Start order processing thread
        self.processing_thread = threading.Thread(target=self._process_orders)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Connected to simulated broker")
        return True
    
    def disconnect(self):
        """Disconnect from the simulated broker."""
        if not self.connected:
            logger.warning("Not connected to simulated broker")
            return
        
        self.running = False
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.connected = False
        logger.info("Disconnected from simulated broker")
    
    def place_order(self, order):
        """
        Place an order with the simulated broker.
        
        Parameters:
        -----------
        order : Order
            Order to place.
        
        Returns:
        --------
        bool
            True if the order was successfully placed, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Update order status
        order.update(status=OrderStatus.PENDING)
        
        # Add to processing queue
        self.order_queue.put(order)
        
        logger.info(f"Placed order: {order}")
        return True
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        # Check if order can be canceled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.ERROR]:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False
        
        # Update order status
        order.update(status=OrderStatus.CANCELED)
        
        logger.info(f"Canceled order: {order}")
        return True
    
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
        --------
        dict
            Account information.
        """
        return self.account.copy()
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
        --------
        dict
            Dictionary of positions by symbol.
        """
        return self.positions.copy()
    
    def get_order_status(self, order_id):
        """
        Get the status of an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order.
        
        Returns:
        --------
        OrderStatus
            Status of the order.
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return None
        
        return self.orders[order_id].status
    
    def update_market_data(self, symbol, data):
        """
        Update market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        data : dict
            Market data.
        """
        self.market_data[symbol] = data
    
    def _process_orders(self):
        """Process orders in the queue."""
        while self.running:
            try:
                # Get an order from the queue
                try:
                    order = self.order_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Simulate processing delay
                delay = np.random.normal(self.config['delay_mean'], self.config['delay_std'])
                delay = max(0.01, delay)  # Ensure positive delay
                time.sleep(delay)
                
                # Check if order was canceled
                if order.status == OrderStatus.CANCELED:
                    self.order_queue.task_done()
                    continue
                
                # Update order status
                order.update(status=OrderStatus.SENT)
                
                # Get market data for the symbol
                market_data = self.market_data.get(order.symbol)
                
                if not market_data:
                    # Reject order if no market data
                    order.update(status=OrderStatus.REJECTED, error_message="No market data available")
                    self._handle_order_update(order)
                    self.order_queue.task_done()
                    continue
                
                # Check if order can be filled
                can_fill, fill_price = self._check_fill_conditions(order, market_data)
                
                if not can_fill:
                    # Reject order
                    order.update(status=OrderStatus.REJECTED, error_message="Cannot fill order")
                    self._handle_order_update(order)
                    self.order_queue.task_done()
                    continue
                
                # Determine if order will be filled
                if np.random.random() > self.config['fill_probability']:
                    # Reject order
                    order.update(status=OrderStatus.REJECTED, error_message="Order not filled")
                    self._handle_order_update(order)
                    self.order_queue.task_done()
                    continue
                
                # Determine if partial fill
                is_partial = np.random.random() < self.config['partial_fill_probability']
                
                if is_partial:
                    # Partial fill
                    fill_quantity = max(1, int(order.quantity * np.random.uniform(0.1, 0.9)))
                else:
                    # Full fill
                    fill_quantity = order.quantity
                
                # Calculate commission
                commission = fill_price * fill_quantity * self.config['commission_rate']
                
                # Create fill
                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_quantity,
                    price=fill_price,
                    commission=commission
                )
                
                # Handle fill
                self._handle_fill(fill)
                
                # Update order status
                if fill_quantity < order.quantity:
                    order.update(status=OrderStatus.PARTIALLY_FILLED)
                else:
                    order.update(status=OrderStatus.FILLED)
                
                self._handle_order_update(order)
                
                # Update account and positions
                self._update_account_and_positions(fill)
                
                self.order_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error processing orders: {e}")
    
    def _check_fill_conditions(self, order, market_data):
        """
        Check if an order can be filled based on market conditions.
        
        Parameters:
        -----------
        order : Order
            Order to check.
        market_data : dict
            Market data for the symbol.
        
        Returns:
        --------
        tuple
            (can_fill, fill_price)
        """
        # Get current price
        current_price = market_data.get('price', 0.0)
        
        if current_price <= 0:
            return False, 0.0
        
        # Apply slippage
        slippage = current_price * self.config['slippage']
        
        if order.side == OrderSide.BUY:
            fill_price = current_price + slippage
        else:  # SELL
            fill_price = current_price - slippage
        
        # Check order type
        if order.order_type == OrderType.MARKET:
            # Market orders always fill
            return True, fill_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price is favorable
            if order.side == OrderSide.BUY and fill_price <= order.price:
                return True, min(fill_price, order.price)
            elif order.side == OrderSide.SELL and fill_price >= order.price:
                return True, max(fill_price, order.price)
            else:
                return False, 0.0
        
        elif order.order_type == OrderType.STOP:
            # Stop orders fill if price crosses stop price
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return True, fill_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                return True, fill_price
            else:
                return False, 0.0
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit orders fill if price crosses stop price and limit price is favorable
            if order.side == OrderSide.BUY and current_price >= order.stop_price and fill_price <= order.price:
                return True, min(fill_price, order.price)
            elif order.side == OrderSide.SELL and current_price <= order.stop_price and fill_price >= order.price:
                return True, max(fill_price, order.price)
            else:
                return False, 0.0
        
        return False, 0.0
    
    def _update_account_and_positions(self, fill):
        """
        Update account and positions based on a fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Calculate fill value
        fill_value = fill.quantity * fill.price
        
        # Update positions
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = {
                'quantity': 0,
                'average_price': 0.0,
                'market_value': 0.0
            }
        
        position = self.positions[fill.symbol]
        
        if fill.side == OrderSide.BUY:
            # Buy increases position
            new_quantity = position['quantity'] + fill.quantity
            
            if new_quantity > 0:
                # Calculate new average price
                if position['quantity'] > 0:
                    # Adding to long position
                    position['average_price'] = ((position['quantity'] * position['average_price']) + 
                                               (fill.quantity * fill.price)) / new_quantity
                else:
                    # Covering short position
                    position['average_price'] = fill.price
            
            position['quantity'] = new_quantity
            
            # Update account
            self.account['balance'] -= (fill_value + fill.commission)
        
        else:  # SELL
            # Sell decreases position
            new_quantity = position['quantity'] - fill.quantity
            
            if new_quantity < 0:
                # Calculate new average price
                if position['quantity'] < 0:
                    # Adding to short position
                    position['average_price'] = ((abs(position['quantity']) * position['average_price']) + 
                                               (fill.quantity * fill.price)) / abs(new_quantity)
                else:
                    # Opening short position
                    position['average_price'] = fill.price
            
            position['quantity'] = new_quantity
            
            # Update account
            self.account['balance'] += (fill_value - fill.commission)
        
        # Update position market value
        position['market_value'] = position['quantity'] * fill.price
        
        # Update account equity
        self.account['equity'] = self.account['balance']
        for pos in self.positions.values():
            self.account['equity'] += pos['market_value']
        
        # Update margin
        self.account['margin_used'] = 0.0
        for pos in self.positions.values():
            if pos['quantity'] != 0:
                # Simplified margin calculation
                self.account['margin_used'] += abs(pos['market_value']) * 0.1  # 10% margin requirement
        
        self.account['margin_available'] = self.account['equity'] - self.account['margin_used']
        
        # Call callbacks
        if self.on_account_update:
            self.on_account_update(self.account)
        
        if self.on_position_update:
            self.on_position_update(self.positions)


class InteractiveBrokersBroker(BrokerBase):
    """
    Interactive Brokers implementation.
    
    Note: This is a placeholder implementation. In a real system, this would
    use the Interactive Brokers API to interact with the broker.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Interactive Brokers broker.
        
        Parameters:
        -----------
        config : dict, optional
            Broker configuration.
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault('host', '127.0.0.1')
        self.config.setdefault('port', 7496)  # TWS port
        self.config.setdefault('client_id', 1)
        
        # IB API client (placeholder)
        self.ib_client = None
    
    def connect(self):
        """Connect to Interactive Brokers."""
        if self.connected:
            logger.warning("Already connected to Interactive Brokers")
            return True
        
        logger.info("Connecting to Interactive Brokers (placeholder)")
        
        # In a real implementation, this would connect to the IB API
        # self.ib_client = IBClient(self.config['host'], self.config['port'], self.config['client_id'])
        # self.ib_client.connect()
        
        self.connected = True
        logger.info("Connected to Interactive Brokers")
        return True
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if not self.connected:
            logger.warning("Not connected to Interactive Brokers")
            return
        
        logger.info("Disconnecting from Interactive Brokers (placeholder)")
        
        # In a real implementation, this would disconnect from the IB API
        # self.ib_client.disconnect()
        
        self.connected = False
        logger.info("Disconnected from Interactive Brokers")
    
    def place_order(self, order):
        """
        Place an order with Interactive Brokers.
        
        Parameters:
        -----------
        order : Order
            Order to place.
        
        Returns:
        --------
        bool
            True if the order was successfully placed, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        logger.info(f"Placing order with Interactive Brokers (placeholder): {order}")
        
        # In a real implementation, this would place an order with the IB API
        # ib_order = self._convert_to_ib_order(order)
        # self.ib_client.place_order(ib_order)
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Update order status (simulated)
        order.update(status=OrderStatus.SENT, broker_order_id=f"IB_{order.order_id}")
        
        # Simulate a fill (for demonstration purposes)
        self._simulate_fill(order)
        
        return True
    
    def cancel_order(self, order_id):
        """
        Cancel an order with Interactive Brokers.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        logger.info(f"Canceling order with Interactive Brokers (placeholder): {order}")
        
        # In a real implementation, this would cancel an order with the IB API
        # self.ib_client.cancel_order(order.broker_order_id)
        
        # Update order status (simulated)
        order.update(status=OrderStatus.CANCELED)
        
        return True
    
    def get_account_info(self):
        """
        Get account information from Interactive Brokers.
        
        Returns:
        --------
        dict
            Account information.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return {}
        
        logger.info("Getting account info from Interactive Brokers (placeholder)")
        
        # In a real implementation, this would get account info from the IB API
        # account_info = self.ib_client.get_account_info()
        
        # Simulated account info
        account_info = {
            'balance': 100000.0,
            'equity': 100000.0,
            'margin_used': 0.0,
            'margin_available': 100000.0
        }
        
        return account_info
    
    def get_positions(self):
        """
        Get current positions from Interactive Brokers.
        
        Returns:
        --------
        dict
            Dictionary of positions by symbol.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return {}
        
        logger.info("Getting positions from Interactive Brokers (placeholder)")
        
        # In a real implementation, this would get positions from the IB API
        # positions = self.ib_client.get_positions()
        
        # Simulated positions
        positions = {}
        
        return positions
    
    def get_order_status(self, order_id):
        """
        Get the status of an order from Interactive Brokers.
        
        Parameters:
        -----------
        order_id : str
            ID of the order.
        
        Returns:
        --------
        OrderStatus
            Status of the order.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return None
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return None
        
        logger.info(f"Getting order status from Interactive Brokers (placeholder): {order_id}")
        
        # In a real implementation, this would get order status from the IB API
        # status = self.ib_client.get_order_status(self.orders[order_id].broker_order_id)
        
        # Return stored status
        return self.orders[order_id].status
    
    def _simulate_fill(self, order):
        """
        Simulate a fill for demonstration purposes.
        
        Parameters:
        -----------
        order : Order
            Order to fill.
        """
        # Simulate a delay
        threading.Timer(0.5, self._delayed_fill, args=[order]).start()
    
    def _delayed_fill(self, order):
        """
        Process a delayed fill.
        
        Parameters:
        -----------
        order : Order
            Order to fill.
        """
        # Create a fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price if order.price else 100.0,  # Simulated price
            commission=1.0  # Simulated commission
        )
        
        # Handle the fill
        self._handle_fill(fill)
        
        # Update order status
        order.update(status=OrderStatus.FILLED)
        
        # Handle order update
        self._handle_order_update(order)


class TDAmeritradeBroker(BrokerBase):
    """
    TD Ameritrade implementation.
    
    Note: This is a placeholder implementation. In a real system, this would
    use the TD Ameritrade API to interact with the broker.
    """
    
    def __init__(self, config=None):
        """
        Initialize the TD Ameritrade broker.
        
        Parameters:
        -----------
        config : dict, optional
            Broker configuration.
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault('api_key', '')
        self.config.setdefault('redirect_uri', '')
        self.config.setdefault('token_path', '')
        
        # TD Ameritrade API client (placeholder)
        self.td_client = None
    
    def connect(self):
        """Connect to TD Ameritrade."""
        if self.connected:
            logger.warning("Already connected to TD Ameritrade")
            return True
        
        logger.info("Connecting to TD Ameritrade (placeholder)")
        
        # In a real implementation, this would connect to the TD Ameritrade API
        # self.td_client = TDClient(
        #     client_id=self.config['api_key'],
        #     redirect_uri=self.config['redirect_uri'],
        #     token_path=self.config['token_path']
        # )
        # self.td_client.login()
        
        self.connected = True
        logger.info("Connected to TD Ameritrade")
        return True
    
    def disconnect(self):
        """Disconnect from TD Ameritrade."""
        if not self.connected:
            logger.warning("Not connected to TD Ameritrade")
            return
        
        logger.info("Disconnecting from TD Ameritrade (placeholder)")
        
        # In a real implementation, this would disconnect from the TD Ameritrade API
        # self.td_client = None
        
        self.connected = False
        logger.info("Disconnected from TD Ameritrade")
    
    def place_order(self, order):
        """
        Place an order with TD Ameritrade.
        
        Parameters:
        -----------
        order : Order
            Order to place.
        
        Returns:
        --------
        bool
            True if the order was successfully placed, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        logger.info(f"Placing order with TD Ameritrade (placeholder): {order}")
        
        # In a real implementation, this would place an order with the TD Ameritrade API
        # td_order = self._convert_to_td_order(order)
        # response = self.td_client.place_order(order.symbol, td_order)
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Update order status (simulated)
        order.update(status=OrderStatus.SENT, broker_order_id=f"TD_{order.order_id}")
        
        # Simulate a fill (for demonstration purposes)
        self._simulate_fill(order)
        
        return True
    
    def cancel_order(self, order_id):
        """
        Cancel an order with TD Ameritrade.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return False
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        logger.info(f"Canceling order with TD Ameritrade (placeholder): {order}")
        
        # In a real implementation, this would cancel an order with the TD Ameritrade API
        # self.td_client.cancel_order(order.broker_order_id)
        
        # Update order status (simulated)
        order.update(status=OrderStatus.CANCELED)
        
        return True
    
    def get_account_info(self):
        """
        Get account information from TD Ameritrade.
        
        Returns:
        --------
        dict
            Account information.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return {}
        
        logger.info("Getting account info from TD Ameritrade (placeholder)")
        
        # In a real implementation, this would get account info from the TD Ameritrade API
        # account_info = self.td_client.get_accounts()
        
        # Simulated account info
        account_info = {
            'balance': 100000.0,
            'equity': 100000.0,
            'margin_used': 0.0,
            'margin_available': 100000.0
        }
        
        return account_info
    
    def get_positions(self):
        """
        Get current positions from TD Ameritrade.
        
        Returns:
        --------
        dict
            Dictionary of positions by symbol.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return {}
        
        logger.info("Getting positions from TD Ameritrade (placeholder)")
        
        # In a real implementation, this would get positions from the TD Ameritrade API
        # positions = self.td_client.get_positions()
        
        # Simulated positions
        positions = {}
        
        return positions
    
    def get_order_status(self, order_id):
        """
        Get the status of an order from TD Ameritrade.
        
        Parameters:
        -----------
        order_id : str
            ID of the order.
        
        Returns:
        --------
        OrderStatus
            Status of the order.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return None
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return None
        
        logger.info(f"Getting order status from TD Ameritrade (placeholder): {order_id}")
        
        # In a real implementation, this would get order status from the TD Ameritrade API
        # status = self.td_client.get_order(self.orders[order_id].broker_order_id)
        
        # Return stored status
        return self.orders[order_id].status
    
    def _simulate_fill(self, order):
        """
        Simulate a fill for demonstration purposes.
        
        Parameters:
        -----------
        order : Order
            Order to fill.
        """
        # Simulate a delay
        threading.Timer(0.5, self._delayed_fill, args=[order]).start()
    
    def _delayed_fill(self, order):
        """
        Process a delayed fill.
        
        Parameters:
        -----------
        order : Order
            Order to fill.
        """
        # Create a fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price if order.price else 100.0,  # Simulated price
            commission=1.0  # Simulated commission
        )
        
        # Handle the fill
        self._handle_fill(fill)
        
        # Update order status
        order.update(status=OrderStatus.FILLED)
        
        # Handle order update
        self._handle_order_update(order)


class ExecutionHandler:
    """
    Execution handler for managing order execution.
    """
    
    def __init__(self, broker, risk_manager=None):
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        broker : BrokerBase
            Broker implementation.
        risk_manager : RiskManager, optional
            Risk manager for checking risk limits.
        """
        self.broker = broker
        self.risk_manager = risk_manager
        
        # Register callbacks
        self.broker.register_callbacks(
            on_order_update=self._on_order_update,
            on_fill=self._on_fill,
            on_account_update=self._on_account_update,
            on_position_update=self._on_position_update,
            on_error=self._on_error
        )
        
        # Initialize state
        self.orders = {}  # Dictionary of orders by order_id
        self.fills = []  # List of fills
        self.account_info = {}  # Account information
        self.positions = {}  # Dictionary of positions by symbol
        
        # Callbacks
        self.on_order_update = None
        self.on_fill = None
        self.on_account_update = None
        self.on_position_update = None
        self.on_error = None
        
        logger.info("Execution handler initialized")
    
    def connect(self):
        """Connect to the broker."""
        return self.broker.connect()
    
    def disconnect(self):
        """Disconnect from the broker."""
        self.broker.disconnect()
    
    def place_order(self, order):
        """
        Place an order.
        
        Parameters:
        -----------
        order : Order
            Order to place.
        
        Returns:
        --------
        bool
            True if the order was successfully placed, False otherwise.
        """
        # Check risk limits if risk manager is available
        if self.risk_manager:
            order_dict = order.to_dict()
            is_allowed, reason, modified_order_dict = self.risk_manager.check_risk_limits(order_dict)
            
            if not is_allowed:
                logger.warning(f"Order rejected by risk manager: {reason}")
                
                # Update order status
                order.update(status=OrderStatus.REJECTED, error_message=reason)
                
                # Store the order
                self.orders[order.order_id] = order
                
                # Call callback
                if self.on_order_update:
                    self.on_order_update(order)
                
                return False
            
            if modified_order_dict and modified_order_dict != order_dict:
                logger.info("Order modified by risk manager")
                
                # Update order with modified values
                if 'quantity' in modified_order_dict:
                    order.quantity = modified_order_dict['quantity']
                
                if 'price' in modified_order_dict:
                    order.price = modified_order_dict['price']
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Place the order with the broker
        result = self.broker.place_order(order)
        
        return result
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        return self.broker.cancel_order(order_id)
    
    def get_order(self, order_id):
        """
        Get an order by ID.
        
        Parameters:
        -----------
        order_id : str
            ID of the order.
        
        Returns:
        --------
        Order
            Order object.
        """
        return self.orders.get(order_id)
    
    def get_orders(self, symbol=None, status=None):
        """
        Get orders filtered by symbol and/or status.
        
        Parameters:
        -----------
        symbol : str, optional
            Trading symbol.
        status : OrderStatus, optional
            Order status.
        
        Returns:
        --------
        list
            List of orders.
        """
        result = []
        
        for order in self.orders.values():
            if symbol and order.symbol != symbol:
                continue
            
            if status and order.status != status:
                continue
            
            result.append(order)
        
        return result
    
    def get_open_orders(self, symbol=None):
        """
        Get open orders.
        
        Parameters:
        -----------
        symbol : str, optional
            Trading symbol.
        
        Returns:
        --------
        list
            List of open orders.
        """
        open_statuses = [OrderStatus.CREATED, OrderStatus.PENDING, OrderStatus.SENT, OrderStatus.PARTIALLY_FILLED]
        
        result = []
        
        for order in self.orders.values():
            if symbol and order.symbol != symbol:
                continue
            
            if order.status in open_statuses:
                result.append(order)
        
        return result
    
    def get_fills(self, symbol=None, start_time=None, end_time=None):
        """
        Get fills filtered by symbol and/or time range.
        
        Parameters:
        -----------
        symbol : str, optional
            Trading symbol.
        start_time : datetime.datetime, optional
            Start time.
        end_time : datetime.datetime, optional
            End time.
        
        Returns:
        --------
        list
            List of fills.
        """
        result = []
        
        for fill in self.fills:
            if symbol and fill.symbol != symbol:
                continue
            
            if start_time and fill.timestamp < start_time:
                continue
            
            if end_time and fill.timestamp > end_time:
                continue
            
            result.append(fill)
        
        return result
    
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
        --------
        dict
            Account information.
        """
        # Get latest account info from broker
        self.account_info = self.broker.get_account_info()
        return self.account_info
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
        --------
        dict
            Dictionary of positions by symbol.
        """
        # Get latest positions from broker
        self.positions = self.broker.get_positions()
        return self.positions
    
    def get_position(self, symbol):
        """
        Get position for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        
        Returns:
        --------
        dict
            Position information.
        """
        # Get latest positions from broker
        self.positions = self.broker.get_positions()
        return self.positions.get(symbol)
    
    def register_callbacks(self, on_order_update=None, on_fill=None, on_account_update=None,
                          on_position_update=None, on_error=None):
        """
        Register callbacks for execution events.
        
        Parameters:
        -----------
        on_order_update : callable, optional
            Callback for order updates.
        on_fill : callable, optional
            Callback for fills.
        on_account_update : callable, optional
            Callback for account updates.
        on_position_update : callable, optional
            Callback for position updates.
        on_error : callable, optional
            Callback for errors.
        """
        self.on_order_update = on_order_update
        self.on_fill = on_fill
        self.on_account_update = on_account_update
        self.on_position_update = on_position_update
        self.on_error = on_error
    
    def _on_order_update(self, order):
        """
        Handle order update from broker.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        # Store the order
        self.orders[order.order_id] = order
        
        # Call callback
        if self.on_order_update:
            self.on_order_update(order)
    
    def _on_fill(self, fill):
        """
        Handle fill from broker.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Store the fill
        self.fills.append(fill)
        
        # Call callback
        if self.on_fill:
            self.on_fill(fill)
        
        # Update risk manager if available
        if self.risk_manager:
            # Create account update
            account_data = {
                'balance': self.account_info.get('balance', 0.0),
                'position': self.positions.get(fill.symbol, {}).get('quantity', 0),
                'daily_pnl': 0.0,  # Would calculate from fills
                'trades_today': len(self.fills),
                'last_trade_time': datetime.datetime.now()
            }
            
            self.risk_manager.update_account_state(account_data)
    
    def _on_account_update(self, account_info):
        """
        Handle account update from broker.
        
        Parameters:
        -----------
        account_info : dict
            Account information.
        """
        # Store account info
        self.account_info = account_info
        
        # Call callback
        if self.on_account_update:
            self.on_account_update(account_info)
        
        # Update risk manager if available
        if self.risk_manager:
            account_data = {
                'balance': account_info.get('balance', 0.0),
                'daily_pnl': 0.0  # Would calculate from account history
            }
            
            self.risk_manager.update_account_state(account_data)
    
    def _on_position_update(self, positions):
        """
        Handle position update from broker.
        
        Parameters:
        -----------
        positions : dict
            Dictionary of positions by symbol.
        """
        # Store positions
        self.positions = positions
        
        # Call callback
        if self.on_position_update:
            self.on_position_update(positions)
        
        # Update risk manager if available
        if self.risk_manager:
            for symbol, position in positions.items():
                account_data = {
                    'position': position.get('quantity', 0)
                }
                
                self.risk_manager.update_account_state(account_data)
    
    def _on_error(self, error_msg, order_id=None):
        """
        Handle error from broker.
        
        Parameters:
        -----------
        error_msg : str
            Error message.
        order_id : str, optional
            ID of the order that caused the error.
        """
        logger.error(f"Broker error: {error_msg}")
        
        # Call callback
        if self.on_error:
            self.on_error(error_msg, order_id)


class OrderManager:
    """
    Order manager for handling order lifecycle.
    """
    
    def __init__(self, execution_handler):
        """
        Initialize the order manager.
        
        Parameters:
        -----------
        execution_handler : ExecutionHandler
            Execution handler for placing orders.
        """
        self.execution_handler = execution_handler
        
        # Register callbacks
        self.execution_handler.register_callbacks(
            on_order_update=self._on_order_update,
            on_fill=self._on_fill
        )
        
        # Initialize state
        self.active_orders = {}  # Dictionary of active orders by order_id
        self.order_strategies = {}  # Dictionary of order strategies by order_id
        
        logger.info("Order manager initialized")
    
    def create_market_order(self, symbol, side, quantity, strategy_id=None):
        """
        Create a market order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Order quantity.
        strategy_id : str, optional
            ID of the strategy that generated the order.
        
        Returns:
        --------
        Order
            Created order.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            strategy_id=strategy_id
        )
        
        return order
    
    def create_limit_order(self, symbol, side, quantity, price, strategy_id=None):
        """
        Create a limit order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Order quantity.
        price : float
            Limit price.
        strategy_id : str, optional
            ID of the strategy that generated the order.
        
        Returns:
        --------
        Order
            Created order.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            price=price,
            strategy_id=strategy_id
        )
        
        return order
    
    def create_stop_order(self, symbol, side, quantity, stop_price, strategy_id=None):
        """
        Create a stop order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Order quantity.
        stop_price : float
            Stop price.
        strategy_id : str, optional
            ID of the strategy that generated the order.
        
        Returns:
        --------
        Order
            Created order.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.STOP,
            side=side,
            quantity=quantity,
            stop_price=stop_price,
            strategy_id=strategy_id
        )
        
        return order
    
    def create_stop_limit_order(self, symbol, side, quantity, stop_price, limit_price, strategy_id=None):
        """
        Create a stop-limit order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Order quantity.
        stop_price : float
            Stop price.
        limit_price : float
            Limit price.
        strategy_id : str, optional
            ID of the strategy that generated the order.
        
        Returns:
        --------
        Order
            Created order.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.STOP_LIMIT,
            side=side,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            strategy_id=strategy_id
        )
        
        return order
    
    def submit_order(self, order, order_strategy=None):
        """
        Submit an order for execution.
        
        Parameters:
        -----------
        order : Order
            Order to submit.
        order_strategy : OrderStrategy, optional
            Strategy for managing the order.
        
        Returns:
        --------
        bool
            True if the order was successfully submitted, False otherwise.
        """
        # Store order strategy if provided
        if order_strategy:
            self.order_strategies[order.order_id] = order_strategy
        
        # Store active order
        self.active_orders[order.order_id] = order
        
        # Submit order for execution
        result = self.execution_handler.place_order(order)
        
        if not result:
            # Remove from active orders if submission failed
            self.active_orders.pop(order.order_id, None)
            self.order_strategies.pop(order.order_id, None)
        
        return result
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel.
        
        Returns:
        --------
        bool
            True if the order was successfully canceled, False otherwise.
        """
        # Check if order is active
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} is not active")
            return False
        
        # Cancel order
        result = self.execution_handler.cancel_order(order_id)
        
        if result:
            # Remove from active orders if cancellation succeeded
            self.active_orders.pop(order_id, None)
            self.order_strategies.pop(order_id, None)
        
        return result
    
    def cancel_all_orders(self, symbol=None):
        """
        Cancel all active orders.
        
        Parameters:
        -----------
        symbol : str, optional
            Trading symbol. If provided, only cancels orders for this symbol.
        
        Returns:
        --------
        int
            Number of orders canceled.
        """
        canceled_count = 0
        
        # Get list of order IDs to cancel
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            order = self.active_orders[order_id]
            
            if symbol and order.symbol != symbol:
                continue
            
            if self.cancel_order(order_id):
                canceled_count += 1
        
        return canceled_count
    
    def get_active_orders(self, symbol=None):
        """
        Get active orders.
        
        Parameters:
        -----------
        symbol : str, optional
            Trading symbol. If provided, only returns orders for this symbol.
        
        Returns:
        --------
        list
            List of active orders.
        """
        if not symbol:
            return list(self.active_orders.values())
        
        return [order for order in self.active_orders.values() if order.symbol == symbol]
    
    def _on_order_update(self, order):
        """
        Handle order update.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        # Check if order is complete
        if order.is_complete():
            # Remove from active orders
            self.active_orders.pop(order.order_id, None)
            
            # Notify order strategy if exists
            if order.order_id in self.order_strategies:
                strategy = self.order_strategies[order.order_id]
                strategy.on_order_complete(order)
                
                # Remove strategy if order is complete
                self.order_strategies.pop(order.order_id, None)
        else:
            # Update active order
            self.active_orders[order.order_id] = order
            
            # Notify order strategy if exists
            if order.order_id in self.order_strategies:
                strategy = self.order_strategies[order.order_id]
                strategy.on_order_update(order)
    
    def _on_fill(self, fill):
        """
        Handle fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Notify order strategy if exists
        if fill.order_id in self.order_strategies:
            strategy = self.order_strategies[fill.order_id]
            strategy.on_fill(fill)


class OrderStrategy(ABC):
    """
    Abstract base class for order strategies.
    """
    
    def __init__(self, order_manager):
        """
        Initialize the order strategy.
        
        Parameters:
        -----------
        order_manager : OrderManager
            Order manager for submitting orders.
        """
        self.order_manager = order_manager
        self.orders = {}  # Dictionary of orders by order_id
        self.fills = []  # List of fills
    
    @abstractmethod
    def on_order_update(self, order):
        """
        Handle order update.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        pass
    
    @abstractmethod
    def on_fill(self, fill):
        """
        Handle fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        pass
    
    @abstractmethod
    def on_order_complete(self, order):
        """
        Handle order completion.
        
        Parameters:
        -----------
        order : Order
            Completed order.
        """
        pass


class TWAPStrategy(OrderStrategy):
    """
    Time-Weighted Average Price (TWAP) order strategy.
    """
    
    def __init__(self, order_manager, symbol, side, total_quantity, duration, num_slices):
        """
        Initialize the TWAP strategy.
        
        Parameters:
        -----------
        order_manager : OrderManager
            Order manager for submitting orders.
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        total_quantity : int
            Total order quantity.
        duration : int
            Duration in seconds.
        num_slices : int
            Number of slices to divide the order into.
        """
        super().__init__(order_manager)
        
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.duration = duration
        self.num_slices = num_slices
        
        self.slice_quantity = total_quantity // num_slices
        self.remaining_quantity = total_quantity
        self.executed_quantity = 0
        self.slice_interval = duration / num_slices
        
        self.start_time = None
        self.end_time = None
        self.next_slice_time = None
        
        self.active = False
        self.complete = False
        
        self.timer = None
        
        logger.info(f"Created TWAP strategy for {symbol}: {total_quantity} {side.value} over {duration}s in {num_slices} slices")
    
    def start(self):
        """Start the TWAP strategy."""
        if self.active:
            logger.warning("TWAP strategy already active")
            return
        
        self.active = True
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=self.duration)
        self.next_slice_time = self.start_time
        
        logger.info(f"Starting TWAP strategy for {self.symbol}: {self.total_quantity} {self.side.value}")
        
        # Submit first slice
        self._submit_next_slice()
    
    def stop(self):
        """Stop the TWAP strategy."""
        if not self.active:
            logger.warning("TWAP strategy not active")
            return
        
        self.active = False
        
        # Cancel timer if exists
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
        # Cancel all active orders
        for order_id in list(self.orders.keys()):
            self.order_manager.cancel_order(order_id)
        
        logger.info(f"Stopped TWAP strategy for {self.symbol}")
    
    def on_order_update(self, order):
        """
        Handle order update.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        # Store the order
        self.orders[order.order_id] = order
    
    def on_fill(self, fill):
        """
        Handle fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Store the fill
        self.fills.append(fill)
        
        # Update executed quantity
        self.executed_quantity += fill.quantity
        self.remaining_quantity -= fill.quantity
        
        logger.info(f"TWAP fill: {fill.quantity} at {fill.price}, "
                   f"executed: {self.executed_quantity}/{self.total_quantity}")
        
        # Check if strategy is complete
        if self.executed_quantity >= self.total_quantity:
            self.complete = True
            self.active = False
            
            # Cancel timer if exists
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            logger.info(f"TWAP strategy complete for {self.symbol}")
    
    def on_order_complete(self, order):
        """
        Handle order completion.
        
        Parameters:
        -----------
        order : Order
            Completed order.
        """
        # Remove from orders
        self.orders.pop(order.order_id, None)
        
        # Check if we need to submit next slice
        if self.active and not self.complete:
            current_time = datetime.datetime.now()
            
            if current_time >= self.next_slice_time:
                self._submit_next_slice()
    
    def _submit_next_slice(self):
        """Submit the next slice of the TWAP order."""
        if not self.active or self.complete:
            return
        
        current_time = datetime.datetime.now()
        
        # Check if we've reached the end time
        if current_time >= self.end_time:
            # Submit remaining quantity
            if self.remaining_quantity > 0:
                self._submit_slice(self.remaining_quantity)
            
            self.active = False
            return
        
        # Calculate quantity for this slice
        quantity = min(self.slice_quantity, self.remaining_quantity)
        
        if quantity <= 0:
            self.complete = True
            self.active = False
            return
        
        # Submit slice
        self._submit_slice(quantity)
        
        # Schedule next slice
        self.next_slice_time = current_time + datetime.timedelta(seconds=self.slice_interval)
        
        # Set timer for next slice
        self.timer = threading.Timer(self.slice_interval, self._submit_next_slice)
        self.timer.daemon = True
        self.timer.start()
    
    def _submit_slice(self, quantity):
        """
        Submit a slice of the TWAP order.
        
        Parameters:
        -----------
        quantity : int
            Quantity for this slice.
        """
        # Create market order
        order = self.order_manager.create_market_order(
            symbol=self.symbol,
            side=self.side,
            quantity=quantity,
            strategy_id="TWAP"
        )
        
        # Submit order
        self.order_manager.submit_order(order, self)
        
        logger.info(f"Submitted TWAP slice: {quantity} {self.side.value} {self.symbol}")


class VWAPStrategy(OrderStrategy):
    """
    Volume-Weighted Average Price (VWAP) order strategy.
    """
    
    def __init__(self, order_manager, symbol, side, total_quantity, duration, volume_profile):
        """
        Initialize the VWAP strategy.
        
        Parameters:
        -----------
        order_manager : OrderManager
            Order manager for submitting orders.
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        total_quantity : int
            Total order quantity.
        duration : int
            Duration in seconds.
        volume_profile : list
            List of volume percentages for each time slice.
        """
        super().__init__(order_manager)
        
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.duration = duration
        self.volume_profile = volume_profile
        
        self.num_slices = len(volume_profile)
        self.remaining_quantity = total_quantity
        self.executed_quantity = 0
        self.slice_interval = duration / self.num_slices
        
        self.start_time = None
        self.end_time = None
        self.next_slice_time = None
        self.current_slice = 0
        
        self.active = False
        self.complete = False
        
        self.timer = None
        
        logger.info(f"Created VWAP strategy for {symbol}: {total_quantity} {side.value} over {duration}s with {self.num_slices} slices")
    
    def start(self):
        """Start the VWAP strategy."""
        if self.active:
            logger.warning("VWAP strategy already active")
            return
        
        self.active = True
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=self.duration)
        self.next_slice_time = self.start_time
        
        logger.info(f"Starting VWAP strategy for {self.symbol}: {self.total_quantity} {self.side.value}")
        
        # Submit first slice
        self._submit_next_slice()
    
    def stop(self):
        """Stop the VWAP strategy."""
        if not self.active:
            logger.warning("VWAP strategy not active")
            return
        
        self.active = False
        
        # Cancel timer if exists
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
        # Cancel all active orders
        for order_id in list(self.orders.keys()):
            self.order_manager.cancel_order(order_id)
        
        logger.info(f"Stopped VWAP strategy for {self.symbol}")
    
    def on_order_update(self, order):
        """
        Handle order update.
        
        Parameters:
        -----------
        order : Order
            Updated order.
        """
        # Store the order
        self.orders[order.order_id] = order
    
    def on_fill(self, fill):
        """
        Handle fill.
        
        Parameters:
        -----------
        fill : Fill
            Fill information.
        """
        # Store the fill
        self.fills.append(fill)
        
        # Update executed quantity
        self.executed_quantity += fill.quantity
        self.remaining_quantity -= fill.quantity
        
        logger.info(f"VWAP fill: {fill.quantity} at {fill.price}, "
                   f"executed: {self.executed_quantity}/{self.total_quantity}")
        
        # Check if strategy is complete
        if self.executed_quantity >= self.total_quantity:
            self.complete = True
            self.active = False
            
            # Cancel timer if exists
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            logger.info(f"VWAP strategy complete for {self.symbol}")
    
    def on_order_complete(self, order):
        """
        Handle order completion.
        
        Parameters:
        -----------
        order : Order
            Completed order.
        """
        # Remove from orders
        self.orders.pop(order.order_id, None)
        
        # Check if we need to submit next slice
        if self.active and not self.complete:
            current_time = datetime.datetime.now()
            
            if current_time >= self.next_slice_time:
                self._submit_next_slice()
    
    def _submit_next_slice(self):
        """Submit the next slice of the VWAP order."""
        if not self.active or self.complete:
            return
        
        current_time = datetime.datetime.now()
        
        # Check if we've reached the end time
        if current_time >= self.end_time or self.current_slice >= self.num_slices:
            # Submit remaining quantity
            if self.remaining_quantity > 0:
                self._submit_slice(self.remaining_quantity)
            
            self.active = False
            return
        
        # Calculate quantity for this slice
        volume_percent = self.volume_profile[self.current_slice]
        quantity = int(self.total_quantity * volume_percent)
        quantity = min(quantity, self.remaining_quantity)
        
        if quantity <= 0:
            self.current_slice += 1
            self.next_slice_time = current_time + datetime.timedelta(seconds=self.slice_interval)
            
            # Set timer for next slice
            self.timer = threading.Timer(self.slice_interval, self._submit_next_slice)
            self.timer.daemon = True
            self.timer.start()
            return
        
        # Submit slice
        self._submit_slice(quantity)
        
        # Move to next slice
        self.current_slice += 1
        self.next_slice_time = current_time + datetime.timedelta(seconds=self.slice_interval)
        
        # Set timer for next slice
        self.timer = threading.Timer(self.slice_interval, self._submit_next_slice)
        self.timer.daemon = True
        self.timer.start()
    
    def _submit_slice(self, quantity):
        """
        Submit a slice of the VWAP order.
        
        Parameters:
        -----------
        quantity : int
            Quantity for this slice.
        """
        # Create market order
        order = self.order_manager.create_market_order(
            symbol=self.symbol,
            side=self.side,
            quantity=quantity,
            strategy_id="VWAP"
        )
        
        # Submit order
        self.order_manager.submit_order(order, self)
        
        logger.info(f"Submitted VWAP slice: {quantity} {self.side.value} {self.symbol}")


class SmartOrderRouter:
    """
    Smart order router for optimizing order execution.
    """
    
    def __init__(self, execution_handler):
        """
        Initialize the smart order router.
        
        Parameters:
        -----------
        execution_handler : ExecutionHandler
            Execution handler for placing orders.
        """
        self.execution_handler = execution_handler
        
        # Initialize state
        self.market_data = {}  # Dictionary of market data by symbol
        
        logger.info("Smart order router initialized")
    
    def update_market_data(self, symbol, data):
        """
        Update market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        data : dict
            Market data.
        """
        self.market_data[symbol] = data
    
    def route_order(self, order):
        """
        Route an order to the optimal execution venue.
        
        Parameters:
        -----------
        order : Order
            Order to route.
        
        Returns:
        --------
        bool
            True if the order was successfully routed, False otherwise.
        """
        # In a real implementation, this would analyze market data and choose the best venue
        # For now, just pass the order to the execution handler
        
        logger.info(f"Routing order: {order}")
        
        return self.execution_handler.place_order(order)
    
    def split_order(self, order, num_parts):
        """
        Split an order into multiple smaller orders.
        
        Parameters:
        -----------
        order : Order
            Order to split.
        num_parts : int
            Number of parts to split the order into.
        
        Returns:
        --------
        list
            List of split orders.
        """
        if num_parts <= 1:
            return [order]
        
        # Calculate base quantity per part
        base_quantity = order.quantity // num_parts
        
        # Calculate remainder
        remainder = order.quantity % num_parts
        
        # Create split orders
        split_orders = []
        
        for i in range(num_parts):
            # Add remainder to first parts
            part_quantity = base_quantity + (1 if i < remainder else 0)
            
            if part_quantity <= 0:
                continue
            
            # Create new order
            split_order = Order(
                symbol=order.symbol,
                order_type=order.order_type,
                side=order.side,
                quantity=part_quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                strategy_id=order.strategy_id
            )
            
            split_orders.append(split_order)
        
        logger.info(f"Split order {order.order_id} into {len(split_orders)} parts")
        
        return split_orders
    
    def create_twap_strategy(self, symbol, side, quantity, duration, num_slices):
        """
        Create a TWAP order strategy.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Total order quantity.
        duration : int
            Duration in seconds.
        num_slices : int
            Number of slices to divide the order into.
        
        Returns:
        --------
        TWAPStrategy
            Created TWAP strategy.
        """
        # Create order manager if not already available
        order_manager = OrderManager(self.execution_handler)
        
        # Create TWAP strategy
        twap = TWAPStrategy(
            order_manager=order_manager,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            duration=duration,
            num_slices=num_slices
        )
        
        return twap
    
    def create_vwap_strategy(self, symbol, side, quantity, duration, volume_profile=None):
        """
        Create a VWAP order strategy.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Side of order (buy or sell).
        quantity : int
            Total order quantity.
        duration : int
            Duration in seconds.
        volume_profile : list, optional
            List of volume percentages for each time slice. If None, uses a default profile.
        
        Returns:
        --------
        VWAPStrategy
            Created VWAP strategy.
        """
        # Create order manager if not already available
        order_manager = OrderManager(self.execution_handler)
        
        # Use default volume profile if not provided
        if volume_profile is None:
            # Simple U-shaped volume profile (higher volume at open and close)
            volume_profile = [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
        
        # Create VWAP strategy
        vwap = VWAPStrategy(
            order_manager=order_manager,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            duration=duration,
            volume_profile=volume_profile
        )
        
        return vwap


if __name__ == "__main__":
    # Example usage
    print("Execution module ready for use")
