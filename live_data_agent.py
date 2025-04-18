#!/usr/bin/env python3
"""
Live Data Agent for Nasdaq-100 E-mini Futures

This script collects real-time data for Nasdaq-100 E-mini futures (NQ) 
using web scraping techniques and public data sources.

The data is stored locally in a format compatible with Lightning AI for training.
"""

import os
import time
import json
import logging
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
import schedule
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_data_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NQ_SYMBOL = "NQ=F"  # Yahoo Finance symbol for Nasdaq-100 E-mini futures
CME_NQ_URL = "https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.quotes.html"
INVESTING_NQ_URL = "https://www.investing.com/indices/nq-100-futures"
DATA_DIR = "data"
LIVE_DATA_FILE = os.path.join(DATA_DIR, "nq_live_data.csv")
CONFIG_FILE = "config.json"

class NQLiveDataAgent:
    """Agent for collecting live data for Nasdaq-100 E-mini futures."""
    
    def __init__(self, config_file=CONFIG_FILE):
        """Initialize the agent."""
        self.config = self._load_config(config_file)
        self.data_dir = self.config.get("data_dir", DATA_DIR)
        self.live_data_file = os.path.join(self.data_dir, self.config.get("live_data_file", "nq_live_data.csv"))
        self.interval = self.config.get("interval", 60)  # Default to 60 seconds
        self.sources = self.config.get("sources", ["yahoo", "cme", "investing"])
        self.driver = None
        self.last_data = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data file if it doesn't exist
        if not os.path.exists(self.live_data_file):
            self._initialize_data_file()
    
    def _load_config(self, config_file):
        """Load configuration from file."""
        default_config = {
            "data_dir": DATA_DIR,
            "live_data_file": "nq_live_data.csv",
            "interval": 60,
            "sources": ["yahoo", "cme", "investing"],
            "use_selenium": True,
            "headless": True
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            # Save default config
            try:
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                
                logger.info(f"Created default configuration file: {config_file}")
            except Exception as e:
                logger.error(f"Error creating configuration file: {e}")
        
        return default_config
    
    def _initialize_data_file(self):
        """Initialize the data file with headers."""
        df = pd.DataFrame(columns=[
            "timestamp", "source", "open", "high", "low", "close", "volume", 
            "bid", "ask", "last", "change", "change_percent"
        ])
        df.to_csv(self.live_data_file, index=False)
        logger.info(f"Initialized data file: {self.live_data_file}")
    
    def _initialize_selenium(self):
        """Initialize Selenium WebDriver."""
        if self.config.get("use_selenium", True):
            try:
                chrome_options = Options()
                if self.config.get("headless", True):
                    chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                
                self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
                logger.info("Initialized Selenium WebDriver")
            except Exception as e:
                logger.error(f"Error initializing Selenium WebDriver: {e}")
                self.driver = None
    
    def _close_selenium(self):
        """Close Selenium WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Closed Selenium WebDriver")
            except Exception as e:
                logger.error(f"Error closing Selenium WebDriver: {e}")
            finally:
                self.driver = None
    
    def get_data_from_yahoo(self):
        """Get live data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(NQ_SYMBOL)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                logger.warning("No data received from Yahoo Finance")
                return None
            
            # Get the latest data point
            latest = data.iloc[-1]
            
            result = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "yahoo",
                "open": latest.get("Open", None),
                "high": latest.get("High", None),
                "low": latest.get("Low", None),
                "close": latest.get("Close", None),
                "volume": latest.get("Volume", None),
                "bid": None,
                "ask": None,
                "last": latest.get("Close", None),
                "change": None,
                "change_percent": None
            }
            
            # Try to get more detailed quote information
            try:
                quote = ticker.info
                result["bid"] = quote.get("bid", None)
                result["ask"] = quote.get("ask", None)
                result["change"] = quote.get("regularMarketChange", None)
                result["change_percent"] = quote.get("regularMarketChangePercent", None)
            except Exception as e:
                logger.warning(f"Error getting detailed quote from Yahoo Finance: {e}")
            
            logger.info(f"Got data from Yahoo Finance: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error getting data from Yahoo Finance: {e}")
            return None
    
    def get_data_from_cme(self):
        """Get live data from CME Group website."""
        if not self.driver:
            self._initialize_selenium()
            
        if not self.driver:
            logger.error("Selenium WebDriver not available")
            return None
        
        try:
            self.driver.get(CME_NQ_URL)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote-table"))
            )
            
            # Extract data
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            quote_table = soup.find("div", class_="quote-table")
            
            if not quote_table:
                logger.warning("Quote table not found on CME Group website")
                return None
            
            # Extract values
            last_price = None
            change = None
            change_percent = None
            open_price = None
            high_price = None
            low_price = None
            volume = None
            
            # Find the last price
            last_elem = quote_table.find("span", class_="last-price")
            if last_elem:
                last_price = float(last_elem.text.strip().replace(",", ""))
            
            # Find other values
            rows = quote_table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    label = cells[0].text.strip().lower()
                    value = cells[1].text.strip().replace(",", "")
                    
                    if "open" in label:
                        try:
                            open_price = float(value)
                        except:
                            pass
                    elif "high" in label:
                        try:
                            high_price = float(value)
                        except:
                            pass
                    elif "low" in label:
                        try:
                            low_price = float(value)
                        except:
                            pass
                    elif "volume" in label:
                        try:
                            volume = int(value)
                        except:
                            pass
                    elif "change" in label and "%" not in label:
                        try:
                            change = float(value)
                        except:
                            pass
                    elif "change" in label and "%" in label:
                        try:
                            change_percent = float(value.replace("%", ""))
                        except:
                            pass
            
            result = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "cme",
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": None,
                "volume": volume,
                "bid": None,
                "ask": None,
                "last": last_price,
                "change": change,
                "change_percent": change_percent
            }
            
            logger.info(f"Got data from CME Group: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error getting data from CME Group: {e}")
            return None
    
    def get_data_from_investing(self):
        """Get live data from Investing.com."""
        if not self.driver:
            self._initialize_selenium()
            
        if not self.driver:
            logger.error("Selenium WebDriver not available")
            return None
        
        try:
            self.driver.get(INVESTING_NQ_URL)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "last_last"))
            )
            
            # Extract data
            last_price = float(self.driver.find_element(By.ID, "last_last").text.replace(",", ""))
            
            # Get other elements
            bid = None
            ask = None
            change = None
            change_percent = None
            open_price = None
            high_price = None
            low_price = None
            volume = None
            
            try:
                change_elem = self.driver.find_element(By.ID, "last_change")
                change = float(change_elem.text.replace(",", ""))
            except:
                pass
            
            try:
                change_percent_elem = self.driver.find_element(By.ID, "last_pcp")
                change_percent = float(change_percent_elem.text.replace("%", "").replace(",", ""))
            except:
                pass
            
            # Find the summary table
            try:
                summary_table = self.driver.find_element(By.ID, "quotes_summary_secondary_data")
                rows = summary_table.find_elements(By.TAG_NAME, "tr")
                
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip().replace(",", "")
                        
                        if "open" in label:
                            try:
                                open_price = float(value)
                            except:
                                pass
                        elif "high" in label:
                            try:
                                high_price = float(value)
                            except:
                                pass
                        elif "low" in label:
                            try:
                                low_price = float(value)
                            except:
                                pass
                        elif "volume" in label:
                            try:
                                volume = int(value)
                            except:
                                pass
            except:
                pass
            
            result = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "investing",
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": None,
                "volume": volume,
                "bid": bid,
                "ask": ask,
                "last": last_price,
                "change": change,
                "change_percent": change_percent
            }
            
            logger.info(f"Got data from Investing.com: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error getting data from Investing.com: {e}")
            return None
    
    def collect_data(self):
        """Collect data from all configured sources."""
        all_data = []
        
        if "yahoo" in self.sources:
            data = self.get_data_from_yahoo()
            if data:
                all_data.append(data)
        
        if "cme" in self.sources:
            data = self.get_data_from_cme()
            if data:
                all_data.append(data)
        
        if "investing" in self.sources:
            data = self.get_data_from_investing()
            if data:
                all_data.append(data)
        
        if all_data:
            # Save data to file
            df = pd.DataFrame(all_data)
            
            # Append to existing file
            if os.path.exists(self.live_data_file):
                df.to_csv(self.live_data_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.live_data_file, index=False)
            
            logger.info(f"Saved {len(all_data)} data points to {self.live_data_file}")
            
            # Update last data
            self.last_data = all_data
            
            return all_data
        else:
            logger.warning("No data collected from any source")
            return None
    
    def run_once(self):
        """Run the data collection once."""
        logger.info("Running data collection...")
        return self.collect_data()
    
    def run_continuously(self):
        """Run the data collection continuously."""
        logger.info(f"Starting continuous data collection every {self.interval} seconds")
        
        # Run once immediately
        self.run_once()
        
        # Schedule regular runs
        schedule.every(self.interval).seconds.do(self.run_once)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
        finally:
            self._close_selenium()
    
    def get_latest_data(self):
        """Get the latest collected data."""
        return self.last_data
    
    def get_data_as_dataframe(self):
        """Get all collected data as a pandas DataFrame."""
        if os.path.exists(self.live_data_file):
            return pd.read_csv(self.live_data_file)
        else:
            return pd.DataFrame()
    
    def export_for_lightning_ai(self, output_file=None):
        """Export data in a format compatible with Lightning AI."""
        if not output_file:
            output_file = os.path.join(self.data_dir, "nq_live_data_lightning.csv")
        
        df = self.get_data_as_dataframe()
        
        if df.empty:
            logger.warning("No data to export")
            return None
        
        # Process data for Lightning AI
        # Group by timestamp and take the average of numeric columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by minute
        df['minute'] = df['timestamp'].dt.floor('min')
        
        # Aggregate data
        agg_df = df.groupby('minute').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'last': 'last'
        }).reset_index()
        
        # Rename columns
        agg_df = agg_df.rename(columns={
            'minute': 'timestamp',
            'last': 'last_price'
        })
        
        # Save to file
        agg_df.to_csv(output_file, index=False)
        logger.info(f"Exported data for Lightning AI to {output_file}")
        
        return output_file

def main():
    """Main function."""
    agent = NQLiveDataAgent()
    
    # Run continuously
    agent.run_continuously()

if __name__ == "__main__":
    main()
