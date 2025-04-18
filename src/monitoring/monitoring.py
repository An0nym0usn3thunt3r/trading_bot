"""
Monitoring and Reporting System for Nasdaq-100 E-mini futures trading bot.
This module implements various monitoring, logging, alerting, and reporting components.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import datetime
import time
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import telegram
import requests
import socket
import psutil
import traceback
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    Main monitoring system class that coordinates various monitoring components.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the monitoring system.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the monitoring configuration file.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.log_manager = LogManager(self.config)
        
        # Initialize state
        self.is_running = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # seconds
        
        # Initialize data storage
        self.data_dir = self.config.get('data_dir', 'logs')
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Monitoring system initialized")
    
    def _load_config(self, config_path):
        """
        Load monitoring configuration from file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        
        Returns:
        --------
        dict
            Monitoring configuration.
        """
        default_config = {
            'monitoring_interval': 60,  # seconds
            'data_dir': 'logs',
            'performance_metrics': ['pnl', 'sharpe_ratio', 'drawdown', 'win_rate'],
            'system_metrics': ['cpu', 'memory', 'disk', 'network'],
            'alert_thresholds': {
                'drawdown': 0.05,  # 5%
                'daily_loss': 1000.0,  # $1000
                'cpu_usage': 80.0,  # 80%
                'memory_usage': 80.0,  # 80%
                'disk_usage': 80.0,  # 80%
                'error_rate': 0.01  # 1%
            },
            'alert_channels': ['email', 'telegram', 'log'],
            'email_settings': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': '',
                'to_emails': []
            },
            'telegram_settings': {
                'bot_token': '',
                'chat_id': ''
            },
            'report_schedule': {
                'daily': '17:00',  # 5:00 PM
                'weekly': 'Friday 17:00',  # Friday 5:00 PM
                'monthly': '1 17:00'  # 1st day of month 5:00 PM
            },
            'log_settings': {
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_file': 'trading_bot.log',
                'max_log_size': 10485760,  # 10 MB
                'backup_count': 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded monitoring configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading monitoring configuration: {e}")
        
        return default_config
    
    def start(self):
        """Start the monitoring system."""
        if self.is_running:
            logger.warning("Monitoring system already running")
            return
        
        self.is_running = True
        
        # Start log manager
        self.log_manager.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start report generator
        self.report_generator.start()
        
        logger.info("Monitoring system started")
    
    def stop(self):
        """Stop the monitoring system."""
        if not self.is_running:
            logger.warning("Monitoring system not running")
            return
        
        self.is_running = False
        
        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop report generator
        self.report_generator.stop()
        
        # Stop log manager
        self.log_manager.stop()
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect performance metrics
                performance_metrics = self.performance_monitor.collect_metrics()
                
                # Collect system metrics
                system_metrics = self.system_monitor.collect_metrics()
                
                # Check for alerts
                self._check_alerts(performance_metrics, system_metrics)
                
                # Save metrics
                self._save_metrics(performance_metrics, system_metrics)
                
                # Sleep until next monitoring interval
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)  # Sleep for a short time before retrying
    
    def _check_alerts(self, performance_metrics, system_metrics):
        """
        Check for alert conditions.
        
        Parameters:
        -----------
        performance_metrics : dict
            Performance metrics.
        system_metrics : dict
            System metrics.
        """
        # Check performance alerts
        if 'drawdown' in performance_metrics:
            drawdown = performance_metrics['drawdown']
            threshold = self.config['alert_thresholds']['drawdown']
            
            if drawdown >= threshold:
                self.alert_manager.send_alert(
                    'Drawdown Alert',
                    f'Drawdown has reached {drawdown:.2%}, which exceeds the threshold of {threshold:.2%}',
                    'high'
                )
        
        if 'daily_pnl' in performance_metrics:
            daily_pnl = performance_metrics['daily_pnl']
            threshold = -self.config['alert_thresholds']['daily_loss']
            
            if daily_pnl <= threshold:
                self.alert_manager.send_alert(
                    'Daily Loss Alert',
                    f'Daily P&L is ${daily_pnl:.2f}, which exceeds the threshold of ${-threshold:.2f}',
                    'high'
                )
        
        # Check system alerts
        if 'cpu_usage' in system_metrics:
            cpu_usage = system_metrics['cpu_usage']
            threshold = self.config['alert_thresholds']['cpu_usage']
            
            if cpu_usage >= threshold:
                self.alert_manager.send_alert(
                    'CPU Usage Alert',
                    f'CPU usage is {cpu_usage:.1f}%, which exceeds the threshold of {threshold:.1f}%',
                    'medium'
                )
        
        if 'memory_usage' in system_metrics:
            memory_usage = system_metrics['memory_usage']
            threshold = self.config['alert_thresholds']['memory_usage']
            
            if memory_usage >= threshold:
                self.alert_manager.send_alert(
                    'Memory Usage Alert',
                    f'Memory usage is {memory_usage:.1f}%, which exceeds the threshold of {threshold:.1f}%',
                    'medium'
                )
        
        if 'disk_usage' in system_metrics:
            disk_usage = system_metrics['disk_usage']
            threshold = self.config['alert_thresholds']['disk_usage']
            
            if disk_usage >= threshold:
                self.alert_manager.send_alert(
                    'Disk Usage Alert',
                    f'Disk usage is {disk_usage:.1f}%, which exceeds the threshold of {threshold:.1f}%',
                    'medium'
                )
        
        if 'error_rate' in system_metrics:
            error_rate = system_metrics['error_rate']
            threshold = self.config['alert_thresholds']['error_rate']
            
            if error_rate >= threshold:
                self.alert_manager.send_alert(
                    'Error Rate Alert',
                    f'Error rate is {error_rate:.2%}, which exceeds the threshold of {threshold:.2%}',
                    'high'
                )
    
    def _save_metrics(self, performance_metrics, system_metrics):
        """
        Save metrics to disk.
        
        Parameters:
        -----------
        performance_metrics : dict
            Performance metrics.
        system_metrics : dict
            System metrics.
        """
        # Create timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Combine metrics
        metrics = {
            'timestamp': timestamp,
            'performance': performance_metrics,
            'system': system_metrics
        }
        
        # Save to JSON file
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        metrics_file = os.path.join(self.data_dir, f'metrics_{date_str}.json')
        
        try:
            # Append to file
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def update_trading_data(self, data):
        """
        Update trading data for monitoring.
        
        Parameters:
        -----------
        data : dict
            Trading data.
        """
        self.performance_monitor.update_trading_data(data)
    
    def update_strategy_data(self, strategy_id, data):
        """
        Update strategy data for monitoring.
        
        Parameters:
        -----------
        strategy_id : str
            Strategy ID.
        data : dict
            Strategy data.
        """
        self.performance_monitor.update_strategy_data(strategy_id, data)
    
    def log_order(self, order):
        """
        Log an order.
        
        Parameters:
        -----------
        order : dict
            Order information.
        """
        self.log_manager.log_order(order)
    
    def log_fill(self, fill):
        """
        Log a fill.
        
        Parameters:
        -----------
        fill : dict
            Fill information.
        """
        self.log_manager.log_fill(fill)
    
    def log_error(self, error_msg, context=None):
        """
        Log an error.
        
        Parameters:
        -----------
        error_msg : str
            Error message.
        context : dict, optional
            Error context.
        """
        self.log_manager.log_error(error_msg, context)
        self.system_monitor.increment_error_count()
    
    def generate_report(self, report_type='daily'):
        """
        Generate a report.
        
        Parameters:
        -----------
        report_type : str
            Type of report ('daily', 'weekly', or 'monthly').
        
        Returns:
        --------
        str
            Path to the generated report.
        """
        return self.report_generator.generate_report(report_type)


class PerformanceMonitor:
    """
    Monitor trading performance metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the performance monitor.
        
        Parameters:
        -----------
        config : dict
            Monitoring configuration.
        """
        self.config = config
        
        # Initialize metrics
        self.metrics = {
            'pnl': 0.0,
            'daily_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'drawdown': 0.0,
            'max_drawdown': 0.0,
            'equity': 100000.0,  # Initial equity
            'peak_equity': 100000.0,  # Initial peak equity
            'positions': {}
        }
        
        # Initialize trading data
        self.trading_data = {
            'orders': [],
            'fills': [],
            'positions': {},
            'account': {
                'balance': 100000.0,
                'equity': 100000.0
            },
            'daily_pnl_history': [],
            'equity_history': []
        }
        
        # Initialize strategy data
        self.strategy_data = {}
        
        # Initialize equity curve
        self.equity_curve = [(datetime.datetime.now(), self.metrics['equity'])]
        
        logger.info("Performance monitor initialized")
    
    def collect_metrics(self):
        """
        Collect performance metrics.
        
        Returns:
        --------
        dict
            Performance metrics.
        """
        # Update metrics based on trading data
        self._update_metrics()
        
        return self.metrics.copy()
    
    def update_trading_data(self, data):
        """
        Update trading data.
        
        Parameters:
        -----------
        data : dict
            Trading data.
        """
        # Update orders
        if 'orders' in data:
            self.trading_data['orders'].extend(data['orders'])
        
        # Update fills
        if 'fills' in data:
            self.trading_data['fills'].extend(data['fills'])
            
            # Update total trades
            self.metrics['total_trades'] = len(self.trading_data['fills'])
        
        # Update positions
        if 'positions' in data:
            self.trading_data['positions'] = data['positions']
            self.metrics['positions'] = data['positions']
        
        # Update account
        if 'account' in data:
            self.trading_data['account'] = data['account']
            
            # Update equity
            self.metrics['equity'] = data['account'].get('equity', self.metrics['equity'])
            
            # Update peak equity
            if self.metrics['equity'] > self.metrics['peak_equity']:
                self.metrics['peak_equity'] = self.metrics['equity']
            
            # Update drawdown
            if self.metrics['peak_equity'] > 0:
                self.metrics['drawdown'] = 1.0 - (self.metrics['equity'] / self.metrics['peak_equity'])
                
                # Update max drawdown
                if self.metrics['drawdown'] > self.metrics['max_drawdown']:
                    self.metrics['max_drawdown'] = self.metrics['drawdown']
            
            # Add to equity history
            self.trading_data['equity_history'].append((datetime.datetime.now(), self.metrics['equity']))
            self.equity_curve.append((datetime.datetime.now(), self.metrics['equity']))
        
        # Update daily P&L
        if 'daily_pnl' in data:
            self.metrics['daily_pnl'] = data['daily_pnl']
            self.trading_data['daily_pnl_history'].append((datetime.datetime.now(), data['daily_pnl']))
    
    def update_strategy_data(self, strategy_id, data):
        """
        Update strategy data.
        
        Parameters:
        -----------
        strategy_id : str
            Strategy ID.
        data : dict
            Strategy data.
        """
        if strategy_id not in self.strategy_data:
            self.strategy_data[strategy_id] = {
                'trades': [],
                'pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Update strategy data
        for key, value in data.items():
            self.strategy_data[strategy_id][key] = value
    
    def _update_metrics(self):
        """Update performance metrics based on trading data."""
        # Calculate P&L
        self.metrics['pnl'] = self.metrics['equity'] - 100000.0  # Assuming initial equity of $100,000
        
        # Calculate win rate
        if self.metrics['total_trades'] > 0:
            # Count winning and losing trades
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            
            for fill in self.trading_data['fills']:
                if 'pnl' in fill:
                    if fill['pnl'] > 0:
                        winning_trades += 1
                        total_profit += fill['pnl']
                    elif fill['pnl'] < 0:
                        losing_trades += 1
                        total_loss += abs(fill['pnl'])
            
            self.metrics['winning_trades'] = winning_trades
            self.metrics['losing_trades'] = losing_trades
            
            # Calculate win rate
            self.metrics['win_rate'] = winning_trades / self.metrics['total_trades']
            
            # Calculate average win and loss
            if winning_trades > 0:
                self.metrics['average_win'] = total_profit / winning_trades
            
            if losing_trades > 0:
                self.metrics['average_loss'] = total_loss / losing_trades
            
            # Calculate profit factor
            if total_loss > 0:
                self.metrics['profit_factor'] = total_profit / total_loss
        
        # Calculate Sharpe ratio
        if len(self.trading_data['equity_history']) > 1:
            # Calculate daily returns
            returns = []
            
            for i in range(1, len(self.trading_data['equity_history'])):
                prev_equity = self.trading_data['equity_history'][i-1][1]
                curr_equity = self.trading_data['equity_history'][i][1]
                
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if returns:
                # Calculate Sharpe ratio
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    # Annualized Sharpe ratio (assuming daily returns)
                    self.metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)
    
    def get_equity_curve(self):
        """
        Get the equity curve.
        
        Returns:
        --------
        list
            List of (timestamp, equity) tuples.
        """
        return self.equity_curve.copy()
    
    def get_daily_pnl_history(self):
        """
        Get the daily P&L history.
        
        Returns:
        --------
        list
            List of (timestamp, daily_pnl) tuples.
        """
        return self.trading_data['daily_pnl_history'].copy()
    
    def get_strategy_performance(self, strategy_id=None):
        """
        Get strategy performance.
        
        Parameters:
        -----------
        strategy_id : str, optional
            Strategy ID. If None, returns performance for all strategies.
        
        Returns:
        --------
        dict
            Strategy performance.
        """
        if strategy_id:
            return self.strategy_data.get(strategy_id, {}).copy()
        
        return self.strategy_data.copy()


class SystemMonitor:
    """
    Monitor system metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the system monitor.
        
        Parameters:
        -----------
        config : dict
            Monitoring configuration.
        """
        self.config = config
        
        # Initialize metrics
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_sent': 0,
            'network_recv': 0,
            'error_count': 0,
            'error_rate': 0.0,
            'uptime': 0
        }
        
        # Initialize error tracking
        self.error_count = 0
        self.request_count = 0
        self.start_time = datetime.datetime.now()
        
        logger.info("System monitor initialized")
    
    def collect_metrics(self):
        """
        Collect system metrics.
        
        Returns:
        --------
        dict
            System metrics.
        """
        # Collect CPU usage
        self.metrics['cpu_usage'] = psutil.cpu_percent()
        
        # Collect memory usage
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'] = memory.percent
        
        # Collect disk usage
        disk = psutil.disk_usage('/')
        self.metrics['disk_usage'] = disk.percent
        
        # Collect network usage
        network = psutil.net_io_counters()
        self.metrics['network_sent'] = network.bytes_sent
        self.metrics['network_recv'] = network.bytes_recv
        
        # Calculate error rate
        if self.request_count > 0:
            self.metrics['error_rate'] = self.error_count / self.request_count
        
        # Calculate uptime
        uptime = datetime.datetime.now() - self.start_time
        self.metrics['uptime'] = uptime.total_seconds()
        
        return self.metrics.copy()
    
    def increment_error_count(self):
        """Increment the error count."""
        self.error_count += 1
    
    def increment_request_count(self):
        """Increment the request count."""
        self.request_count += 1
    
    def reset_counters(self):
        """Reset error and request counters."""
        self.error_count = 0
        self.request_count = 0


class AlertManager:
    """
    Manage alerts and notifications.
    """
    
    def __init__(self, config):
        """
        Initialize the alert manager.
        
        Parameters:
        -----------
        config : dict
            Monitoring configuration.
        """
        self.config = config
        
        # Initialize alert channels
        self.channels = self.config.get('alert_channels', ['log'])
        
        # Initialize alert history
        self.alert_history = []
        
        # Initialize rate limiting
        self.rate_limits = {
            'low': 3600,  # 1 hour
            'medium': 1800,  # 30 minutes
            'high': 300  # 5 minutes
        }
        self.last_alert_time = {}
        
        logger.info("Alert manager initialized")
    
    def send_alert(self, title, message, severity='medium'):
        """
        Send an alert.
        
        Parameters:
        -----------
        title : str
            Alert title.
        message : str
            Alert message.
        severity : str
            Alert severity ('low', 'medium', or 'high').
        
        Returns:
        --------
        bool
            True if the alert was sent, False otherwise.
        """
        # Check rate limiting
        if not self._check_rate_limit(title, severity):
            return False
        
        # Create alert
        alert = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send to each channel
        for channel in self.channels:
            try:
                if channel == 'email':
                    self._send_email_alert(alert)
                elif channel == 'telegram':
                    self._send_telegram_alert(alert)
                elif channel == 'log':
                    self._send_log_alert(alert)
            except Exception as e:
                logger.error(f"Error sending alert to {channel}: {e}")
        
        return True
    
    def _check_rate_limit(self, alert_key, severity):
        """
        Check if an alert is rate limited.
        
        Parameters:
        -----------
        alert_key : str
            Alert key for rate limiting.
        severity : str
            Alert severity.
        
        Returns:
        --------
        bool
            True if the alert is not rate limited, False otherwise.
        """
        current_time = datetime.datetime.now()
        
        # Get rate limit for severity
        rate_limit = self.rate_limits.get(severity, 3600)
        
        # Check if alert is rate limited
        if alert_key in self.last_alert_time:
            last_time = self.last_alert_time[alert_key]
            time_diff = (current_time - last_time).total_seconds()
            
            if time_diff < rate_limit:
                return False
        
        # Update last alert time
        self.last_alert_time[alert_key] = current_time
        
        return True
    
    def _send_email_alert(self, alert):
        """
        Send an alert via email.
        
        Parameters:
        -----------
        alert : dict
            Alert information.
        """
        # Get email settings
        email_settings = self.config.get('email_settings', {})
        
        if not email_settings.get('username') or not email_settings.get('to_emails'):
            logger.warning("Email settings not configured")
            return
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = email_settings.get('from_email', email_settings['username'])
        msg['To'] = ', '.join(email_settings['to_emails'])
        msg['Subject'] = f"[{alert['severity'].upper()}] {alert['title']}"
        
        # Add body
        body = f"{alert['message']}\n\nTimestamp: {alert['timestamp']}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            server = smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port'])
            server.starttls()
            server.login(email_settings['username'], email_settings['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent email alert: {alert['title']}")
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_telegram_alert(self, alert):
        """
        Send an alert via Telegram.
        
        Parameters:
        -----------
        alert : dict
            Alert information.
        """
        # Get Telegram settings
        telegram_settings = self.config.get('telegram_settings', {})
        
        if not telegram_settings.get('bot_token') or not telegram_settings.get('chat_id'):
            logger.warning("Telegram settings not configured")
            return
        
        # Create message
        message = f"*[{alert['severity'].upper()}] {alert['title']}*\n\n{alert['message']}\n\nTimestamp: {alert['timestamp']}"
        
        # Send message
        try:
            bot = telegram.Bot(token=telegram_settings['bot_token'])
            bot.send_message(
                chat_id=telegram_settings['chat_id'],
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"Sent Telegram alert: {alert['title']}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def _send_log_alert(self, alert):
        """
        Send an alert to the log.
        
        Parameters:
        -----------
        alert : dict
            Alert information.
        """
        log_message = f"ALERT [{alert['severity'].upper()}] {alert['title']}: {alert['message']}"
        
        if alert['severity'] == 'high':
            logger.critical(log_message)
        elif alert['severity'] == 'medium':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_alert_history(self, severity=None, start_time=None, end_time=None):
        """
        Get alert history.
        
        Parameters:
        -----------
        severity : str, optional
            Filter by severity.
        start_time : datetime.datetime, optional
            Start time.
        end_time : datetime.datetime, optional
            End time.
        
        Returns:
        --------
        list
            List of alerts.
        """
        filtered_alerts = []
        
        for alert in self.alert_history:
            # Filter by severity
            if severity and alert['severity'] != severity:
                continue
            
            # Filter by time range
            alert_time = datetime.datetime.fromisoformat(alert['timestamp'])
            
            if start_time and alert_time < start_time:
                continue
            
            if end_time and alert_time > end_time:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts


class ReportGenerator:
    """
    Generate performance reports.
    """
    
    def __init__(self, config):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        config : dict
            Monitoring configuration.
        """
        self.config = config
        
        # Initialize state
        self.is_running = False
        self.report_thread = None
        
        # Initialize report schedule
        self.report_schedule = self.config.get('report_schedule', {})
        
        # Initialize data directory
        self.data_dir = self.config.get('data_dir', 'logs')
        self.report_dir = os.path.join(self.data_dir, 'reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize performance monitor
        self.performance_monitor = None
        
        logger.info("Report generator initialized")
    
    def start(self):
        """Start the report generator."""
        if self.is_running:
            logger.warning("Report generator already running")
            return
        
        self.is_running = True
        
        # Start report thread
        self.report_thread = threading.Thread(target=self._report_loop)
        self.report_thread.daemon = True
        self.report_thread.start()
        
        logger.info("Report generator started")
    
    def stop(self):
        """Stop the report generator."""
        if not self.is_running:
            logger.warning("Report generator not running")
            return
        
        self.is_running = False
        
        # Stop report thread
        if self.report_thread:
            self.report_thread.join(timeout=5.0)
        
        logger.info("Report generator stopped")
    
    def set_performance_monitor(self, performance_monitor):
        """
        Set the performance monitor.
        
        Parameters:
        -----------
        performance_monitor : PerformanceMonitor
            Performance monitor.
        """
        self.performance_monitor = performance_monitor
    
    def _report_loop(self):
        """Main report loop."""
        while self.is_running:
            try:
                # Get current time
                now = datetime.datetime.now()
                
                # Check daily report
                if 'daily' in self.report_schedule:
                    daily_time = self.report_schedule['daily']
                    daily_hour, daily_minute = map(int, daily_time.split(':'))
                    
                    if now.hour == daily_hour and now.minute == daily_minute:
                        self.generate_report('daily')
                
                # Check weekly report
                if 'weekly' in self.report_schedule:
                    weekly_schedule = self.report_schedule['weekly'].split()
                    weekly_day = weekly_schedule[0]
                    weekly_time = weekly_schedule[1]
                    weekly_hour, weekly_minute = map(int, weekly_time.split(':'))
                    
                    if now.strftime('%A') == weekly_day and now.hour == weekly_hour and now.minute == weekly_minute:
                        self.generate_report('weekly')
                
                # Check monthly report
                if 'monthly' in self.report_schedule:
                    monthly_schedule = self.report_schedule['monthly'].split()
                    monthly_day = int(monthly_schedule[0])
                    monthly_time = monthly_schedule[1]
                    monthly_hour, monthly_minute = map(int, monthly_time.split(':'))
                    
                    if now.day == monthly_day and now.hour == monthly_hour and now.minute == monthly_minute:
                        self.generate_report('monthly')
                
                # Sleep for 1 minute
                time.sleep(60)
            
            except Exception as e:
                logger.error(f"Error in report loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Sleep for a minute before retrying
    
    def generate_report(self, report_type='daily'):
        """
        Generate a report.
        
        Parameters:
        -----------
        report_type : str
            Type of report ('daily', 'weekly', or 'monthly').
        
        Returns:
        --------
        str
            Path to the generated report.
        """
        logger.info(f"Generating {report_type} report")
        
        # Create report directory
        report_date = datetime.datetime.now().strftime('%Y-%m-%d')
        report_dir = os.path.join(self.report_dir, f"{report_type}_{report_date}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report sections
        self._generate_performance_summary(report_dir, report_type)
        self._generate_equity_curve(report_dir, report_type)
        self._generate_trade_analysis(report_dir, report_type)
        self._generate_strategy_performance(report_dir, report_type)
        
        # Generate HTML report
        html_path = self._generate_html_report(report_dir, report_type)
        
        logger.info(f"Generated {report_type} report: {html_path}")
        
        return html_path
    
    def _generate_performance_summary(self, report_dir, report_type):
        """
        Generate performance summary.
        
        Parameters:
        -----------
        report_dir : str
            Report directory.
        report_type : str
            Type of report.
        """
        if not self.performance_monitor:
            logger.warning("Performance monitor not set")
            return
        
        # Get performance metrics
        metrics = self.performance_monitor.collect_metrics()
        
        # Create summary
        summary = {
            'report_type': report_type,
            'report_date': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Save to JSON file
        summary_path = os.path.join(report_dir, 'performance_summary.json')
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary table
        table_data = []
        
        for key, value in metrics.items():
            if key == 'positions':
                continue
            
            if isinstance(value, float):
                if key in ['win_rate', 'drawdown', 'max_drawdown']:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            table_data.append([key, formatted_value])
        
        # Save to CSV file
        table_path = os.path.join(report_dir, 'performance_summary.csv')
        
        with open(table_path, 'w') as f:
            f.write('Metric,Value\n')
            for row in table_data:
                f.write(f"{row[0]},{row[1]}\n")
    
    def _generate_equity_curve(self, report_dir, report_type):
        """
        Generate equity curve.
        
        Parameters:
        -----------
        report_dir : str
            Report directory.
        report_type : str
            Type of report.
        """
        if not self.performance_monitor:
            logger.warning("Performance monitor not set")
            return
        
        # Get equity curve
        equity_curve = self.performance_monitor.get_equity_curve()
        
        if not equity_curve:
            logger.warning("No equity curve data")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        dates = [point[0] for point in equity_curve]
        values = [point[1] for point in equity_curve]
        
        plt.plot(dates, values)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
        
        # Save figure
        equity_path = os.path.join(report_dir, 'equity_curve.png')
        plt.savefig(equity_path)
        plt.close()
        
        # Save data to CSV
        csv_path = os.path.join(report_dir, 'equity_curve.csv')
        
        with open(csv_path, 'w') as f:
            f.write('Date,Equity\n')
            for date, value in equity_curve:
                f.write(f"{date.isoformat()},{value}\n")
    
    def _generate_trade_analysis(self, report_dir, report_type):
        """
        Generate trade analysis.
        
        Parameters:
        -----------
        report_dir : str
            Report directory.
        report_type : str
            Type of report.
        """
        if not self.performance_monitor:
            logger.warning("Performance monitor not set")
            return
        
        # Get trading data
        trading_data = self.performance_monitor.trading_data
        
        if not trading_data.get('fills'):
            logger.warning("No trade data")
            return
        
        # Create trade analysis
        trades = []
        
        for fill in trading_data['fills']:
            if 'timestamp' in fill and 'symbol' in fill and 'side' in fill and 'quantity' in fill and 'price' in fill:
                trade = {
                    'timestamp': fill['timestamp'],
                    'symbol': fill['symbol'],
                    'side': fill['side'],
                    'quantity': fill['quantity'],
                    'price': fill['price'],
                    'pnl': fill.get('pnl', 0.0)
                }
                
                trades.append(trade)
        
        # Save to CSV file
        trades_path = os.path.join(report_dir, 'trades.csv')
        
        with open(trades_path, 'w') as f:
            f.write('Timestamp,Symbol,Side,Quantity,Price,PnL\n')
            for trade in trades:
                f.write(f"{trade['timestamp']},{trade['symbol']},{trade['side']},{trade['quantity']},{trade['price']},{trade['pnl']}\n")
        
        # Create trade distribution chart
        if trades:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot trade P&L distribution
            pnl_values = [trade['pnl'] for trade in trades if 'pnl' in trade]
            
            if pnl_values:
                sns.histplot(pnl_values, kde=True)
                plt.title('Trade P&L Distribution')
                plt.xlabel('P&L')
                plt.ylabel('Frequency')
                plt.grid(True)
                
                # Save figure
                dist_path = os.path.join(report_dir, 'trade_distribution.png')
                plt.savefig(dist_path)
                plt.close()
    
    def _generate_strategy_performance(self, report_dir, report_type):
        """
        Generate strategy performance.
        
        Parameters:
        -----------
        report_dir : str
            Report directory.
        report_type : str
            Type of report.
        """
        if not self.performance_monitor:
            logger.warning("Performance monitor not set")
            return
        
        # Get strategy performance
        strategy_performance = self.performance_monitor.get_strategy_performance()
        
        if not strategy_performance:
            logger.warning("No strategy performance data")
            return
        
        # Create strategy performance table
        table_data = []
        
        for strategy_id, performance in strategy_performance.items():
            row = {
                'strategy_id': strategy_id,
                'pnl': performance.get('pnl', 0.0),
                'win_rate': performance.get('win_rate', 0.0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                'trades': len(performance.get('trades', []))
            }
            
            table_data.append(row)
        
        # Save to CSV file
        strategy_path = os.path.join(report_dir, 'strategy_performance.csv')
        
        with open(strategy_path, 'w') as f:
            f.write('Strategy,PnL,Win Rate,Sharpe Ratio,Trades\n')
            for row in table_data:
                f.write(f"{row['strategy_id']},{row['pnl']:.2f},{row['win_rate']:.2%},{row['sharpe_ratio']:.2f},{row['trades']}\n")
        
        # Create strategy comparison chart
        if table_data:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot strategy P&L
            strategies = [row['strategy_id'] for row in table_data]
            pnl_values = [row['pnl'] for row in table_data]
            
            plt.bar(strategies, pnl_values)
            plt.title('Strategy P&L Comparison')
            plt.xlabel('Strategy')
            plt.ylabel('P&L')
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Format y-axis
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
            
            # Save figure
            strategy_chart_path = os.path.join(report_dir, 'strategy_comparison.png')
            plt.savefig(strategy_chart_path)
            plt.close()
    
    def _generate_html_report(self, report_dir, report_type):
        """
        Generate HTML report.
        
        Parameters:
        -----------
        report_dir : str
            Report directory.
        report_type : str
            Type of report.
        
        Returns:
        --------
        str
            Path to the HTML report.
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_type.capitalize()} Trading Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #0066cc;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>{report_type.capitalize()} Trading Report</h1>
            <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Performance Summary</h2>
        """
        
        # Add performance summary
        summary_path = os.path.join(report_dir, 'performance_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            html_content += "<table>"
            html_content += "<tr><th>Metric</th><th>Value</th></tr>"
            
            for key, value in summary['metrics'].items():
                if key == 'positions':
                    continue
                
                if isinstance(value, float):
                    if key in ['win_rate', 'drawdown', 'max_drawdown']:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                html_content += f"<tr><td>{key}</td><td>{formatted_value}</td></tr>"
            
            html_content += "</table>"
        
        # Add equity curve
        equity_path = 'equity_curve.png'
        if os.path.exists(os.path.join(report_dir, equity_path)):
            html_content += f"""
            <div class="section">
                <h2>Equity Curve</h2>
                <img class="chart" src="{equity_path}" alt="Equity Curve">
            </div>
            """
        
        # Add trade distribution
        dist_path = 'trade_distribution.png'
        if os.path.exists(os.path.join(report_dir, dist_path)):
            html_content += f"""
            <div class="section">
                <h2>Trade P&L Distribution</h2>
                <img class="chart" src="{dist_path}" alt="Trade P&L Distribution">
            </div>
            """
        
        # Add strategy comparison
        strategy_chart_path = 'strategy_comparison.png'
        if os.path.exists(os.path.join(report_dir, strategy_chart_path)):
            html_content += f"""
            <div class="section">
                <h2>Strategy Performance</h2>
                <img class="chart" src="{strategy_chart_path}" alt="Strategy Performance">
            </div>
            """
        
        # Add trades table
        trades_path = os.path.join(report_dir, 'trades.csv')
        if os.path.exists(trades_path):
            html_content += """
            <div class="section">
                <h2>Recent Trades</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                    </tr>
            """
            
            with open(trades_path, 'r') as f:
                # Skip header
                next(f)
                
                # Add up to 20 most recent trades
                lines = list(f)[-20:]
                
                for line in lines:
                    fields = line.strip().split(',')
                    html_content += f"""
                    <tr>
                        <td>{fields[0]}</td>
                        <td>{fields[1]}</td>
                        <td>{fields[2]}</td>
                        <td>{fields[3]}</td>
                        <td>{fields[4]}</td>
                        <td>{fields[5]}</td>
                    </tr>
                    """
            
            html_content += """
                </table>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        html_path = os.path.join(report_dir, 'report.html')
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path


class LogManager:
    """
    Manage logs and log rotation.
    """
    
    def __init__(self, config):
        """
        Initialize the log manager.
        
        Parameters:
        -----------
        config : dict
            Monitoring configuration.
        """
        self.config = config
        
        # Get log settings
        self.log_settings = self.config.get('log_settings', {})
        
        # Initialize log directory
        self.data_dir = self.config.get('data_dir', 'logs')
        self.log_dir = os.path.join(self.data_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log files
        self.log_file = os.path.join(self.log_dir, self.log_settings.get('log_file', 'trading_bot.log'))
        self.order_log_file = os.path.join(self.log_dir, 'orders.log')
        self.fill_log_file = os.path.join(self.log_dir, 'fills.log')
        self.error_log_file = os.path.join(self.log_dir, 'errors.log')
        
        # Initialize log handlers
        self.log_handlers = {}
        
        logger.info("Log manager initialized")
    
    def start(self):
        """Start the log manager."""
        # Configure root logger
        root_logger = logging.getLogger()
        
        # Set log level
        log_level_str = self.log_settings.get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str)
        root_logger.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        log_format = self.log_settings.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger.addHandler(file_handler)
        
        # Store handler
        self.log_handlers['root'] = file_handler
        
        logger.info("Log manager started")
    
    def stop(self):
        """Stop the log manager."""
        # Remove handlers
        root_logger = logging.getLogger()
        
        for handler in self.log_handlers.values():
            root_logger.removeHandler(handler)
        
        self.log_handlers = {}
        
        logger.info("Log manager stopped")
    
    def log_order(self, order):
        """
        Log an order.
        
        Parameters:
        -----------
        order : dict
            Order information.
        """
        # Convert order to string
        order_str = json.dumps(order)
        
        # Log to order log file
        with open(self.order_log_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {order_str}\n")
    
    def log_fill(self, fill):
        """
        Log a fill.
        
        Parameters:
        -----------
        fill : dict
            Fill information.
        """
        # Convert fill to string
        fill_str = json.dumps(fill)
        
        # Log to fill log file
        with open(self.fill_log_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {fill_str}\n")
    
    def log_error(self, error_msg, context=None):
        """
        Log an error.
        
        Parameters:
        -----------
        error_msg : str
            Error message.
        context : dict, optional
            Error context.
        """
        # Create error entry
        error_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'message': error_msg,
            'context': context
        }
        
        # Convert to string
        error_str = json.dumps(error_entry)
        
        # Log to error log file
        with open(self.error_log_file, 'a') as f:
            f.write(f"{error_str}\n")
        
        # Log to main log
        logger.error(f"Error: {error_msg}")
        
        if context:
            logger.error(f"Context: {context}")
    
    def get_logs(self, log_type='main', start_time=None, end_time=None, limit=100):
        """
        Get logs.
        
        Parameters:
        -----------
        log_type : str
            Type of log ('main', 'orders', 'fills', or 'errors').
        start_time : datetime.datetime, optional
            Start time.
        end_time : datetime.datetime, optional
            End time.
        limit : int
            Maximum number of log entries to return.
        
        Returns:
        --------
        list
            List of log entries.
        """
        # Get log file
        if log_type == 'main':
            log_file = self.log_file
        elif log_type == 'orders':
            log_file = self.order_log_file
        elif log_type == 'fills':
            log_file = self.fill_log_file
        elif log_type == 'errors':
            log_file = self.error_log_file
        else:
            logger.error(f"Invalid log type: {log_type}")
            return []
        
        # Check if log file exists
        if not os.path.exists(log_file):
            logger.warning(f"Log file not found: {log_file}")
            return []
        
        # Read log file
        logs = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Parse log entry
                if log_type == 'main':
                    # Parse main log format
                    try:
                        log_parts = line.split(' - ', 3)
                        timestamp_str = log_parts[0]
                        timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        
                        # Filter by time range
                        if start_time and timestamp < start_time:
                            continue
                        
                        if end_time and timestamp > end_time:
                            continue
                        
                        logs.append(line.strip())
                    except Exception:
                        logs.append(line.strip())
                else:
                    # Parse JSON log format
                    try:
                        if log_type == 'errors':
                            # Error logs are JSON objects
                            log_entry = json.loads(line)
                            timestamp = datetime.datetime.fromisoformat(log_entry['timestamp'])
                        else:
                            # Order and fill logs have timestamp prefix
                            timestamp_str, json_str = line.split(' - ', 1)
                            timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        
                        # Filter by time range
                        if start_time and timestamp < start_time:
                            continue
                        
                        if end_time and timestamp > end_time:
                            continue
                        
                        logs.append(line.strip())
                    except Exception:
                        logs.append(line.strip())
        
        # Return limited number of logs (most recent first)
        return logs[-limit:]
    
    def rotate_logs(self):
        """Rotate log files."""
        # Get max log size
        max_log_size = self.log_settings.get('max_log_size', 10485760)  # 10 MB
        
        # Get backup count
        backup_count = self.log_settings.get('backup_count', 5)
        
        # Check and rotate each log file
        for log_file in [self.log_file, self.order_log_file, self.fill_log_file, self.error_log_file]:
            if os.path.exists(log_file) and os.path.getsize(log_file) > max_log_size:
                self._rotate_log_file(log_file, backup_count)
    
    def _rotate_log_file(self, log_file, backup_count):
        """
        Rotate a log file.
        
        Parameters:
        -----------
        log_file : str
            Path to the log file.
        backup_count : int
            Number of backup files to keep.
        """
        # Check if log file exists
        if not os.path.exists(log_file):
            return
        
        # Remove oldest backup
        oldest_backup = f"{log_file}.{backup_count}"
        if os.path.exists(oldest_backup):
            os.remove(oldest_backup)
        
        # Shift existing backups
        for i in range(backup_count - 1, 0, -1):
            backup = f"{log_file}.{i}"
            new_backup = f"{log_file}.{i + 1}"
            
            if os.path.exists(backup):
                os.rename(backup, new_backup)
        
        # Rename current log file
        os.rename(log_file, f"{log_file}.1")
        
        # Create new log file
        open(log_file, 'w').close()
        
        logger.info(f"Rotated log file: {log_file}")


class DashboardServer:
    """
    Web server for the monitoring dashboard.
    """
    
    def __init__(self, monitoring_system, host='0.0.0.0', port=8080):
        """
        Initialize the dashboard server.
        
        Parameters:
        -----------
        monitoring_system : MonitoringSystem
            Monitoring system.
        host : str
            Host to bind to.
        port : int
            Port to bind to.
        """
        self.monitoring_system = monitoring_system
        self.host = host
        self.port = port
        
        # Initialize state
        self.is_running = False
        self.server = None
        self.server_thread = None
        
        logger.info("Dashboard server initialized")
    
    def start(self):
        """Start the dashboard server."""
        if self.is_running:
            logger.warning("Dashboard server already running")
            return
        
        self.is_running = True
        
        # Start server thread
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logger.info(f"Dashboard server started on http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the dashboard server."""
        if not self.is_running:
            logger.warning("Dashboard server not running")
            return
        
        self.is_running = False
        
        # Stop server
        if self.server:
            self.server.shutdown()
        
        # Stop server thread
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        logger.info("Dashboard server stopped")
    
    def _run_server(self):
        """Run the dashboard server."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import urllib.parse
            
            class DashboardHandler(BaseHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    self.monitoring_system = monitoring_system
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    parsed_path = urllib.parse.urlparse(self.path)
                    path = parsed_path.path
                    
                    if path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(self._get_dashboard_html().encode())
                    
                    elif path == '/api/metrics':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        # Get metrics
                        performance_metrics = self.monitoring_system.performance_monitor.collect_metrics()
                        system_metrics = self.monitoring_system.system_monitor.collect_metrics()
                        
                        metrics = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'performance': performance_metrics,
                            'system': system_metrics
                        }
                        
                        self.wfile.write(json.dumps(metrics).encode())
                    
                    elif path == '/api/equity_curve':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        # Get equity curve
                        equity_curve = self.monitoring_system.performance_monitor.get_equity_curve()
                        
                        # Convert to JSON-serializable format
                        equity_data = [
                            {'timestamp': point[0].isoformat(), 'equity': point[1]}
                            for point in equity_curve
                        ]
                        
                        self.wfile.write(json.dumps(equity_data).encode())
                    
                    elif path == '/api/alerts':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        # Get alerts
                        alerts = self.monitoring_system.alert_manager.get_alert_history()
                        
                        self.wfile.write(json.dumps(alerts).encode())
                    
                    elif path.startswith('/reports/'):
                        report_path = path[9:]  # Remove '/reports/'
                        
                        # Get report directory
                        report_dir = os.path.join(self.monitoring_system.report_generator.report_dir, report_path)
                        
                        if os.path.exists(report_dir) and os.path.isdir(report_dir):
                            # List reports
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html')
                            self.end_headers()
                            
                            html = '<html><head><title>Reports</title></head><body>'
                            html += '<h1>Reports</h1>'
                            html += '<ul>'
                            
                            for item in os.listdir(report_dir):
                                item_path = os.path.join(report_dir, item)
                                if os.path.isdir(item_path):
                                    html += f'<li><a href="/reports/{report_path}/{item}">{item}</a></li>'
                                else:
                                    html += f'<li><a href="/report/{report_path}/{item}">{item}</a></li>'
                            
                            html += '</ul></body></html>'
                            
                            self.wfile.write(html.encode())
                        else:
                            self.send_response(404)
                            self.send_header('Content-type', 'text/plain')
                            self.end_headers()
                            self.wfile.write(b'Report not found')
                    
                    elif path.startswith('/report/'):
                        report_path = path[8:]  # Remove '/report/'
                        
                        # Get report file
                        report_file = os.path.join(self.monitoring_system.report_generator.report_dir, report_path)
                        
                        if os.path.exists(report_file) and os.path.isfile(report_file):
                            self.send_response(200)
                            
                            # Set content type based on file extension
                            if report_file.endswith('.html'):
                                self.send_header('Content-type', 'text/html')
                            elif report_file.endswith('.json'):
                                self.send_header('Content-type', 'application/json')
                            elif report_file.endswith('.csv'):
                                self.send_header('Content-type', 'text/csv')
                            elif report_file.endswith('.png'):
                                self.send_header('Content-type', 'image/png')
                            else:
                                self.send_header('Content-type', 'text/plain')
                            
                            self.end_headers()
                            
                            with open(report_file, 'rb') as f:
                                self.wfile.write(f.read())
                        else:
                            self.send_response(404)
                            self.send_header('Content-type', 'text/plain')
                            self.end_headers()
                            self.wfile.write(b'Report file not found')
                    
                    else:
                        self.send_response(404)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b'Not found')
                
                def _get_dashboard_html(self):
                    """Get the dashboard HTML."""
                    return """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Trading Bot Dashboard</title>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <style>
                            body {
                                font-family: Arial, sans-serif;
                                margin: 0;
                                padding: 0;
                                color: #333;
                            }
                            .container {
                                max-width: 1200px;
                                margin: 0 auto;
                                padding: 20px;
                            }
                            .header {
                                background-color: #0066cc;
                                color: white;
                                padding: 10px 20px;
                                margin-bottom: 20px;
                            }
                            .card {
                                background-color: white;
                                border-radius: 5px;
                                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                                padding: 20px;
                                margin-bottom: 20px;
                            }
                            .card h2 {
                                margin-top: 0;
                                color: #0066cc;
                            }
                            .metrics {
                                display: grid;
                                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                                gap: 10px;
                            }
                            .metric {
                                background-color: #f9f9f9;
                                border-radius: 5px;
                                padding: 15px;
                                text-align: center;
                            }
                            .metric h3 {
                                margin-top: 0;
                                font-size: 14px;
                                color: #666;
                            }
                            .metric p {
                                margin-bottom: 0;
                                font-size: 24px;
                                font-weight: bold;
                                color: #0066cc;
                            }
                            .chart-container {
                                height: 300px;
                                margin-bottom: 20px;
                            }
                            table {
                                width: 100%;
                                border-collapse: collapse;
                            }
                            th, td {
                                padding: 8px;
                                text-align: left;
                                border-bottom: 1px solid #ddd;
                            }
                            th {
                                background-color: #f2f2f2;
                            }
                            .alert {
                                padding: 10px;
                                margin-bottom: 10px;
                                border-radius: 5px;
                            }
                            .alert-high {
                                background-color: #f8d7da;
                                color: #721c24;
                            }
                            .alert-medium {
                                background-color: #fff3cd;
                                color: #856404;
                            }
                            .alert-low {
                                background-color: #d1ecf1;
                                color: #0c5460;
                            }
                            .nav {
                                display: flex;
                                margin-bottom: 20px;
                            }
                            .nav a {
                                padding: 10px 15px;
                                text-decoration: none;
                                color: #0066cc;
                                border-bottom: 2px solid transparent;
                            }
                            .nav a.active {
                                border-bottom: 2px solid #0066cc;
                            }
                            .tab-content {
                                display: none;
                            }
                            .tab-content.active {
                                display: block;
                            }
                        </style>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    </head>
                    <body>
                        <div class="header">
                            <h1>Trading Bot Dashboard</h1>
                        </div>
                        
                        <div class="container">
                            <div class="nav">
                                <a href="#" class="tab-link active" data-tab="overview">Overview</a>
                                <a href="#" class="tab-link" data-tab="performance">Performance</a>
                                <a href="#" class="tab-link" data-tab="system">System</a>
                                <a href="#" class="tab-link" data-tab="alerts">Alerts</a>
                                <a href="#" class="tab-link" data-tab="reports">Reports</a>
                            </div>
                            
                            <div id="overview" class="tab-content active">
                                <div class="card">
                                    <h2>Performance Metrics</h2>
                                    <div class="metrics" id="performance-metrics">
                                        <div class="metric">
                                            <h3>P&L</h3>
                                            <p id="metric-pnl">$0.00</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Daily P&L</h3>
                                            <p id="metric-daily-pnl">$0.00</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Win Rate</h3>
                                            <p id="metric-win-rate">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Sharpe Ratio</h3>
                                            <p id="metric-sharpe-ratio">0.00</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Drawdown</h3>
                                            <p id="metric-drawdown">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Total Trades</h3>
                                            <p id="metric-total-trades">0</p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>Equity Curve</h2>
                                    <div class="chart-container">
                                        <canvas id="equity-chart"></canvas>
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>System Health</h2>
                                    <div class="metrics" id="system-metrics">
                                        <div class="metric">
                                            <h3>CPU Usage</h3>
                                            <p id="metric-cpu-usage">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Memory Usage</h3>
                                            <p id="metric-memory-usage">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Disk Usage</h3>
                                            <p id="metric-disk-usage">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Error Rate</h3>
                                            <p id="metric-error-rate">0.0%</p>
                                        </div>
                                        <div class="metric">
                                            <h3>Uptime</h3>
                                            <p id="metric-uptime">0s</p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>Recent Alerts</h2>
                                    <div id="recent-alerts">
                                        <p>No alerts</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="performance" class="tab-content">
                                <div class="card">
                                    <h2>Performance Metrics</h2>
                                    <div class="metrics" id="performance-metrics-full">
                                        <!-- Will be populated by JavaScript -->
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>Equity Curve</h2>
                                    <div class="chart-container">
                                        <canvas id="equity-chart-full"></canvas>
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>Strategy Performance</h2>
                                    <table id="strategy-table">
                                        <tr>
                                            <th>Strategy</th>
                                            <th>P&L</th>
                                            <th>Win Rate</th>
                                            <th>Sharpe Ratio</th>
                                            <th>Trades</th>
                                        </tr>
                                        <!-- Will be populated by JavaScript -->
                                    </table>
                                </div>
                            </div>
                            
                            <div id="system" class="tab-content">
                                <div class="card">
                                    <h2>System Metrics</h2>
                                    <div class="metrics" id="system-metrics-full">
                                        <!-- Will be populated by JavaScript -->
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>CPU Usage</h2>
                                    <div class="chart-container">
                                        <canvas id="cpu-chart"></canvas>
                                    </div>
                                </div>
                                
                                <div class="card">
                                    <h2>Memory Usage</h2>
                                    <div class="chart-container">
                                        <canvas id="memory-chart"></canvas>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="alerts" class="tab-content">
                                <div class="card">
                                    <h2>Alerts</h2>
                                    <div id="alerts-list">
                                        <!-- Will be populated by JavaScript -->
                                    </div>
                                </div>
                            </div>
                            
                            <div id="reports" class="tab-content">
                                <div class="card">
                                    <h2>Reports</h2>
                                    <h3>Daily Reports</h3>
                                    <div id="daily-reports">
                                        <p>Loading...</p>
                                    </div>
                                    
                                    <h3>Weekly Reports</h3>
                                    <div id="weekly-reports">
                                        <p>Loading...</p>
                                    </div>
                                    
                                    <h3>Monthly Reports</h3>
                                    <div id="monthly-reports">
                                        <p>Loading...</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <script>
                            // Tab navigation
                            document.querySelectorAll('.tab-link').forEach(link => {
                                link.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    
                                    // Remove active class from all tabs
                                    document.querySelectorAll('.tab-link').forEach(l => {
                                        l.classList.remove('active');
                                    });
                                    
                                    // Add active class to clicked tab
                                    this.classList.add('active');
                                    
                                    // Hide all tab content
                                    document.querySelectorAll('.tab-content').forEach(content => {
                                        content.classList.remove('active');
                                    });
                                    
                                    // Show selected tab content
                                    const tabId = this.getAttribute('data-tab');
                                    document.getElementById(tabId).classList.add('active');
                                });
                            });
                            
                            // Charts
                            let equityChart = null;
                            let equityChartFull = null;
                            let cpuChart = null;
                            let memoryChart = null;
                            
                            // Initialize charts
                            function initCharts() {
                                // Equity chart
                                const equityCtx = document.getElementById('equity-chart').getContext('2d');
                                equityChart = new Chart(equityCtx, {
                                    type: 'line',
                                    data: {
                                        labels: [],
                                        datasets: [{
                                            label: 'Equity',
                                            data: [],
                                            borderColor: '#0066cc',
                                            backgroundColor: 'rgba(0, 102, 204, 0.1)',
                                            fill: true
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        scales: {
                                            x: {
                                                type: 'time',
                                                time: {
                                                    unit: 'day'
                                                }
                                            },
                                            y: {
                                                beginAtZero: false
                                            }
                                        }
                                    }
                                });
                                
                                // Full equity chart
                                const equityFullCtx = document.getElementById('equity-chart-full').getContext('2d');
                                equityChartFull = new Chart(equityFullCtx, {
                                    type: 'line',
                                    data: {
                                        labels: [],
                                        datasets: [{
                                            label: 'Equity',
                                            data: [],
                                            borderColor: '#0066cc',
                                            backgroundColor: 'rgba(0, 102, 204, 0.1)',
                                            fill: true
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        scales: {
                                            x: {
                                                type: 'time',
                                                time: {
                                                    unit: 'day'
                                                }
                                            },
                                            y: {
                                                beginAtZero: false
                                            }
                                        }
                                    }
                                });
                                
                                // CPU chart
                                const cpuCtx = document.getElementById('cpu-chart').getContext('2d');
                                cpuChart = new Chart(cpuCtx, {
                                    type: 'line',
                                    data: {
                                        labels: [],
                                        datasets: [{
                                            label: 'CPU Usage',
                                            data: [],
                                            borderColor: '#dc3545',
                                            backgroundColor: 'rgba(220, 53, 69, 0.1)',
                                            fill: true
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        scales: {
                                            x: {
                                                type: 'time',
                                                time: {
                                                    unit: 'minute'
                                                }
                                            },
                                            y: {
                                                beginAtZero: true,
                                                max: 100
                                            }
                                        }
                                    }
                                });
                                
                                // Memory chart
                                const memoryCtx = document.getElementById('memory-chart').getContext('2d');
                                memoryChart = new Chart(memoryCtx, {
                                    type: 'line',
                                    data: {
                                        labels: [],
                                        datasets: [{
                                            label: 'Memory Usage',
                                            data: [],
                                            borderColor: '#fd7e14',
                                            backgroundColor: 'rgba(253, 126, 20, 0.1)',
                                            fill: true
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        scales: {
                                            x: {
                                                type: 'time',
                                                time: {
                                                    unit: 'minute'
                                                }
                                            },
                                            y: {
                                                beginAtZero: true,
                                                max: 100
                                            }
                                        }
                                    }
                                });
                            }
                            
                            // Update metrics
                            function updateMetrics() {
                                fetch('/api/metrics')
                                    .then(response => response.json())
                                    .then(data => {
                                        // Update performance metrics
                                        document.getElementById('metric-pnl').textContent = '$' + data.performance.pnl.toFixed(2);
                                        document.getElementById('metric-daily-pnl').textContent = '$' + data.performance.daily_pnl.toFixed(2);
                                        document.getElementById('metric-win-rate').textContent = (data.performance.win_rate * 100).toFixed(1) + '%';
                                        document.getElementById('metric-sharpe-ratio').textContent = data.performance.sharpe_ratio.toFixed(2);
                                        document.getElementById('metric-drawdown').textContent = (data.performance.drawdown * 100).toFixed(1) + '%';
                                        document.getElementById('metric-total-trades').textContent = data.performance.total_trades;
                                        
                                        // Update system metrics
                                        document.getElementById('metric-cpu-usage').textContent = data.system.cpu_usage.toFixed(1) + '%';
                                        document.getElementById('metric-memory-usage').textContent = data.system.memory_usage.toFixed(1) + '%';
                                        document.getElementById('metric-disk-usage').textContent = data.system.disk_usage.toFixed(1) + '%';
                                        document.getElementById('metric-error-rate').textContent = (data.system.error_rate * 100).toFixed(1) + '%';
                                        
                                        // Format uptime
                                        const uptime = data.system.uptime;
                                        let uptimeText = '';
                                        
                                        if (uptime >= 86400) {
                                            uptimeText = Math.floor(uptime / 86400) + 'd ';
                                            uptimeText += Math.floor((uptime % 86400) / 3600) + 'h';
                                        } else if (uptime >= 3600) {
                                            uptimeText = Math.floor(uptime / 3600) + 'h ';
                                            uptimeText += Math.floor((uptime % 3600) / 60) + 'm';
                                        } else if (uptime >= 60) {
                                            uptimeText = Math.floor(uptime / 60) + 'm ';
                                            uptimeText += Math.floor(uptime % 60) + 's';
                                        } else {
                                            uptimeText = Math.floor(uptime) + 's';
                                        }
                                        
                                        document.getElementById('metric-uptime').textContent = uptimeText;
                                        
                                        // Update full metrics
                                        const performanceMetricsFull = document.getElementById('performance-metrics-full');
                                        performanceMetricsFull.innerHTML = '';
                                        
                                        for (const [key, value] of Object.entries(data.performance)) {
                                            if (key === 'positions') continue;
                                            
                                            const metricDiv = document.createElement('div');
                                            metricDiv.className = 'metric';
                                            
                                            const metricTitle = document.createElement('h3');
                                            metricTitle.textContent = key;
                                            
                                            const metricValue = document.createElement('p');
                                            if (typeof value === 'number') {
                                                if (key === 'win_rate' || key === 'drawdown' || key === 'max_drawdown') {
                                                    metricValue.textContent = (value * 100).toFixed(1) + '%';
                                                } else if (key === 'pnl' || key === 'daily_pnl' || key === 'average_win' || key === 'average_loss') {
                                                    metricValue.textContent = '$' + value.toFixed(2);
                                                } else {
                                                    metricValue.textContent = value.toFixed(2);
                                                }
                                            } else {
                                                metricValue.textContent = value;
                                            }
                                            
                                            metricDiv.appendChild(metricTitle);
                                            metricDiv.appendChild(metricValue);
                                            performanceMetricsFull.appendChild(metricDiv);
                                        }
                                        
                                        const systemMetricsFull = document.getElementById('system-metrics-full');
                                        systemMetricsFull.innerHTML = '';
                                        
                                        for (const [key, value] of Object.entries(data.system)) {
                                            const metricDiv = document.createElement('div');
                                            metricDiv.className = 'metric';
                                            
                                            const metricTitle = document.createElement('h3');
                                            metricTitle.textContent = key;
                                            
                                            const metricValue = document.createElement('p');
                                            if (typeof value === 'number') {
                                                if (key === 'cpu_usage' || key === 'memory_usage' || key === 'disk_usage') {
                                                    metricValue.textContent = value.toFixed(1) + '%';
                                                } else if (key === 'error_rate') {
                                                    metricValue.textContent = (value * 100).toFixed(1) + '%';
                                                } else if (key === 'uptime') {
                                                    metricValue.textContent = uptimeText;
                                                } else {
                                                    metricValue.textContent = value.toFixed(0);
                                                }
                                            } else {
                                                metricValue.textContent = value;
                                            }
                                            
                                            metricDiv.appendChild(metricTitle);
                                            metricDiv.appendChild(metricValue);
                                            systemMetricsFull.appendChild(metricDiv);
                                        }
                                        
                                        // Update charts
                                        const timestamp = new Date(data.timestamp);
                                        
                                        // Update CPU chart
                                        cpuChart.data.labels.push(timestamp);
                                        cpuChart.data.datasets[0].data.push(data.system.cpu_usage);
                                        
                                        if (cpuChart.data.labels.length > 60) {
                                            cpuChart.data.labels.shift();
                                            cpuChart.data.datasets[0].data.shift();
                                        }
                                        
                                        cpuChart.update();
                                        
                                        // Update memory chart
                                        memoryChart.data.labels.push(timestamp);
                                        memoryChart.data.datasets[0].data.push(data.system.memory_usage);
                                        
                                        if (memoryChart.data.labels.length > 60) {
                                            memoryChart.data.labels.shift();
                                            memoryChart.data.datasets[0].data.shift();
                                        }
                                        
                                        memoryChart.update();
                                    })
                                    .catch(error => {
                                        console.error('Error fetching metrics:', error);
                                    });
                            }
                            
                            // Update equity curve
                            function updateEquityCurve() {
                                fetch('/api/equity_curve')
                                    .then(response => response.json())
                                    .then(data => {
                                        // Update equity chart
                                        equityChart.data.labels = data.map(point => new Date(point.timestamp));
                                        equityChart.data.datasets[0].data = data.map(point => point.equity);
                                        equityChart.update();
                                        
                                        // Update full equity chart
                                        equityChartFull.data.labels = data.map(point => new Date(point.timestamp));
                                        equityChartFull.data.datasets[0].data = data.map(point => point.equity);
                                        equityChartFull.update();
                                    })
                                    .catch(error => {
                                        console.error('Error fetching equity curve:', error);
                                    });
                            }
                            
                            // Update alerts
                            function updateAlerts() {
                                fetch('/api/alerts')
                                    .then(response => response.json())
                                    .then(data => {
                                        // Update recent alerts
                                        const recentAlerts = document.getElementById('recent-alerts');
                                        recentAlerts.innerHTML = '';
                                        
                                        if (data.length === 0) {
                                            recentAlerts.innerHTML = '<p>No alerts</p>';
                                        } else {
                                            // Show 5 most recent alerts
                                            const recentData = data.slice(-5).reverse();
                                            
                                            for (const alert of recentData) {
                                                const alertDiv = document.createElement('div');
                                                alertDiv.className = `alert alert-${alert.severity}`;
                                                
                                                const alertTitle = document.createElement('h3');
                                                alertTitle.textContent = alert.title;
                                                
                                                const alertMessage = document.createElement('p');
                                                alertMessage.textContent = alert.message;
                                                
                                                const alertTime = document.createElement('small');
                                                alertTime.textContent = new Date(alert.timestamp).toLocaleString();
                                                
                                                alertDiv.appendChild(alertTitle);
                                                alertDiv.appendChild(alertMessage);
                                                alertDiv.appendChild(alertTime);
                                                recentAlerts.appendChild(alertDiv);
                                            }
                                        }
                                        
                                        // Update all alerts
                                        const alertsList = document.getElementById('alerts-list');
                                        alertsList.innerHTML = '';
                                        
                                        if (data.length === 0) {
                                            alertsList.innerHTML = '<p>No alerts</p>';
                                        } else {
                                            // Show all alerts, most recent first
                                            const allData = [...data].reverse();
                                            
                                            for (const alert of allData) {
                                                const alertDiv = document.createElement('div');
                                                alertDiv.className = `alert alert-${alert.severity}`;
                                                
                                                const alertTitle = document.createElement('h3');
                                                alertTitle.textContent = alert.title;
                                                
                                                const alertMessage = document.createElement('p');
                                                alertMessage.textContent = alert.message;
                                                
                                                const alertTime = document.createElement('small');
                                                alertTime.textContent = new Date(alert.timestamp).toLocaleString();
                                                
                                                alertDiv.appendChild(alertTitle);
                                                alertDiv.appendChild(alertMessage);
                                                alertDiv.appendChild(alertTime);
                                                alertsList.appendChild(alertDiv);
                                            }
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching alerts:', error);
                                    });
                            }
                            
                            // Update reports
                            function updateReports() {
                                // Fetch daily reports
                                fetch('/reports/daily')
                                    .then(response => response.text())
                                    .then(html => {
                                        const parser = new DOMParser();
                                        const doc = parser.parseFromString(html, 'text/html');
                                        const links = doc.querySelectorAll('a');
                                        
                                        const dailyReports = document.getElementById('daily-reports');
                                        dailyReports.innerHTML = '';
                                        
                                        if (links.length === 0) {
                                            dailyReports.innerHTML = '<p>No daily reports</p>';
                                        } else {
                                            const ul = document.createElement('ul');
                                            
                                            for (const link of links) {
                                                if (link.href.includes('/reports/') || link.href.includes('/report/')) {
                                                    const li = document.createElement('li');
                                                    const a = document.createElement('a');
                                                    a.href = link.href;
                                                    a.textContent = link.textContent;
                                                    a.target = '_blank';
                                                    li.appendChild(a);
                                                    ul.appendChild(li);
                                                }
                                            }
                                            
                                            dailyReports.appendChild(ul);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching daily reports:', error);
                                        document.getElementById('daily-reports').innerHTML = '<p>Error loading daily reports</p>';
                                    });
                                
                                // Fetch weekly reports
                                fetch('/reports/weekly')
                                    .then(response => response.text())
                                    .then(html => {
                                        const parser = new DOMParser();
                                        const doc = parser.parseFromString(html, 'text/html');
                                        const links = doc.querySelectorAll('a');
                                        
                                        const weeklyReports = document.getElementById('weekly-reports');
                                        weeklyReports.innerHTML = '';
                                        
                                        if (links.length === 0) {
                                            weeklyReports.innerHTML = '<p>No weekly reports</p>';
                                        } else {
                                            const ul = document.createElement('ul');
                                            
                                            for (const link of links) {
                                                if (link.href.includes('/reports/') || link.href.includes('/report/')) {
                                                    const li = document.createElement('li');
                                                    const a = document.createElement('a');
                                                    a.href = link.href;
                                                    a.textContent = link.textContent;
                                                    a.target = '_blank';
                                                    li.appendChild(a);
                                                    ul.appendChild(li);
                                                }
                                            }
                                            
                                            weeklyReports.appendChild(ul);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching weekly reports:', error);
                                        document.getElementById('weekly-reports').innerHTML = '<p>Error loading weekly reports</p>';
                                    });
                                
                                // Fetch monthly reports
                                fetch('/reports/monthly')
                                    .then(response => response.text())
                                    .then(html => {
                                        const parser = new DOMParser();
                                        const doc = parser.parseFromString(html, 'text/html');
                                        const links = doc.querySelectorAll('a');
                                        
                                        const monthlyReports = document.getElementById('monthly-reports');
                                        monthlyReports.innerHTML = '';
                                        
                                        if (links.length === 0) {
                                            monthlyReports.innerHTML = '<p>No monthly reports</p>';
                                        } else {
                                            const ul = document.createElement('ul');
                                            
                                            for (const link of links) {
                                                if (link.href.includes('/reports/') || link.href.includes('/report/')) {
                                                    const li = document.createElement('li');
                                                    const a = document.createElement('a');
                                                    a.href = link.href;
                                                    a.textContent = link.textContent;
                                                    a.target = '_blank';
                                                    li.appendChild(a);
                                                    ul.appendChild(li);
                                                }
                                            }
                                            
                                            monthlyReports.appendChild(ul);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching monthly reports:', error);
                                        document.getElementById('monthly-reports').innerHTML = '<p>Error loading monthly reports</p>';
                                    });
                            }
                            
                            // Initialize
                            document.addEventListener('DOMContentLoaded', function() {
                                initCharts();
                                updateMetrics();
                                updateEquityCurve();
                                updateAlerts();
                                updateReports();
                                
                                // Update every 10 seconds
                                setInterval(updateMetrics, 10000);
                                setInterval(updateEquityCurve, 30000);
                                setInterval(updateAlerts, 30000);
                                setInterval(updateReports, 60000);
                            });
                        </script>
                    </body>
                    </html>
                    """
            
            # Create server
            monitoring_system = self.monitoring_system
            self.server = HTTPServer((self.host, self.port), DashboardHandler)
            
            # Run server
            while self.is_running:
                self.server.handle_request()
        
        except Exception as e:
            logger.error(f"Error running dashboard server: {e}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Example usage
    print("Monitoring and Reporting System module ready for use")
