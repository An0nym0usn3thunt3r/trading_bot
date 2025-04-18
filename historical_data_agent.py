#!/usr/bin/env python3
"""
Historical Data Agent for Nasdaq-100 E-mini Futures

This script collects 5 years of historical 1-minute data for Nasdaq-100 E-mini futures (NQ)
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
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("historical_data_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NQ_SYMBOL = "NQ=F"  # Yahoo Finance symbol for Nasdaq-100 E-mini futures
CME_NQ_URL = "https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.quotes.html"
INVESTING_NQ_URL = "https://www.investing.com/indices/nq-100-futures-historical-data"
DATA_DIR = "data"
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "nq_historical_data.csv")
CONFIG_FILE = "config.json"

class NQHistoricalDataAgent:
    """Agent for collecting historical data for Nasdaq-100 E-mini futures."""
    
    def __init__(self, config_file=CONFIG_FILE):
        """Initialize the agent."""
        self.config = self._load_config(config_file)
        self.data_dir = self.config.get("data_dir", DATA_DIR)
        self.historical_data_file = os.path.join(self.data_dir, self.config.get("historical_data_file", "nq_historical_data.csv"))
        self.sources = self.config.get("sources", ["yahoo", "investing"])
        self.years = self.config.get("years", 5)
        self.driver = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_config(self, config_file):
        """Load configuration from file."""
        default_config = {
            "data_dir": DATA_DIR,
            "historical_data_file": "nq_historical_data.csv",
            "sources": ["yahoo", "investing"],
            "years": 5,
            "use_selenium": True,
            "headless": True,
            "max_workers": 4,
            "chunk_size_days": 30
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
    
    def get_date_ranges(self):
        """Get date ranges for data collection."""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * self.years)
        
        # Create chunks of dates
        chunk_size = self.config.get("chunk_size_days", 30)
        date_ranges = []
        
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + datetime.timedelta(days=chunk_size), end_date)
            date_ranges.append((current_start, current_end))
            current_start = current_end
        
        return date_ranges
    
    def get_data_from_yahoo(self, start_date, end_date):
        """Get historical data from Yahoo Finance."""
        try:
            # Convert dates to string format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Getting data from Yahoo Finance for {start_str} to {end_str}")
            
            # Download data
            data = yf.download(
                NQ_SYMBOL,
                start=start_str,
                end=end_str,
                interval="1m",
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data received from Yahoo Finance for {start_str} to {end_str}")
                return None
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            
            # Rename columns to match our format
            data = data.rename(columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Add source column
            data["source"] = "yahoo"
            
            logger.info(f"Got {len(data)} rows from Yahoo Finance for {start_str} to {end_str}")
            return data
        
        except Exception as e:
            logger.error(f"Error getting data from Yahoo Finance: {e}")
            return None
    
    def get_data_from_investing(self, start_date, end_date):
        """Get historical data from Investing.com."""
        if not self.driver:
            self._initialize_selenium()
            
        if not self.driver:
            logger.error("Selenium WebDriver not available")
            return None
        
        try:
            # Convert dates to string format
            start_str = start_date.strftime("%m/%d/%Y")
            end_str = end_date.strftime("%m/%d/%Y")
            
            logger.info(f"Getting data from Investing.com for {start_str} to {end_str}")
            
            # Navigate to the historical data page
            self.driver.get(INVESTING_NQ_URL)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "widgetFieldDateRange"))
            )
            
            # Click on the date range picker
            date_range_picker = self.driver.find_element(By.ID, "widgetFieldDateRange")
            date_range_picker.click()
            
            # Wait for the date picker to appear
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "startDate"))
            )
            
            # Clear and set start date
            start_date_input = self.driver.find_element(By.ID, "startDate")
            start_date_input.clear()
            start_date_input.send_keys(start_str)
            
            # Clear and set end date
            end_date_input = self.driver.find_element(By.ID, "endDate")
            end_date_input.clear()
            end_date_input.send_keys(end_str)
            
            # Click Apply button
            apply_button = self.driver.find_element(By.ID, "applyBtn")
            apply_button.click()
            
            # Wait for the table to load
            time.sleep(3)
            
            # Extract data from the table
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            table = soup.find("table", {"id": "curr_table"})
            
            if not table:
                logger.warning(f"Data table not found on Investing.com for {start_str} to {end_str}")
                return None
            
            # Extract rows
            rows = table.find("tbody").find_all("tr")
            data_list = []
            
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 6:
                    date_str = cells[0].text.strip()
                    price = cells[1].text.strip().replace(",", "")
                    open_price = cells[2].text.strip().replace(",", "")
                    high_price = cells[3].text.strip().replace(",", "")
                    low_price = cells[4].text.strip().replace(",", "")
                    volume_str = cells[5].text.strip()
                    
                    # Parse date
                    try:
                        date = datetime.datetime.strptime(date_str, "%b %d, %Y")
                    except:
                        continue
                    
                    # Parse numeric values
                    try:
                        price = float(price)
                        open_price = float(open_price)
                        high_price = float(high_price)
                        low_price = float(low_price)
                        
                        # Parse volume (may have K, M, B suffixes)
                        volume = 0
                        if "K" in volume_str:
                            volume = float(volume_str.replace("K", "")) * 1000
                        elif "M" in volume_str:
                            volume = float(volume_str.replace("M", "")) * 1000000
                        elif "B" in volume_str:
                            volume = float(volume_str.replace("B", "")) * 1000000000
                        else:
                            volume = float(volume_str.replace(",", ""))
                        
                        data_list.append({
                            "timestamp": date,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": price,
                            "volume": volume,
                            "source": "investing"
                        })
                    except:
                        continue
            
            if not data_list:
                logger.warning(f"No data extracted from Investing.com for {start_str} to {end_str}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            logger.info(f"Got {len(df)} rows from Investing.com for {start_str} to {end_str}")
            return df
        
        except Exception as e:
            logger.error(f"Error getting data from Investing.com: {e}")
            return None
    
    def collect_data_for_range(self, date_range):
        """Collect data for a specific date range from all sources."""
        start_date, end_date = date_range
        all_data = []
        
        if "yahoo" in self.sources:
            data = self.get_data_from_yahoo(start_date, end_date)
            if data is not None and not data.empty:
                all_data.append(data)
        
        if "investing" in self.sources:
            data = self.get_data_from_investing(start_date, end_date)
            if data is not None and not data.empty:
                all_data.append(data)
        
        if not all_data:
            logger.warning(f"No data collected for range {start_date} to {end_date}")
            return None
        
        # Combine data from all sources
        combined_data = pd.concat(all_data, ignore_index=True)
        
        return combined_data
    
    def collect_all_historical_data(self):
        """Collect all historical data for the specified number of years."""
        logger.info(f"Collecting {self.years} years of historical data for Nasdaq-100 E-mini futures")
        
        # Get date ranges
        date_ranges = self.get_date_ranges()
        logger.info(f"Split data collection into {len(date_ranges)} chunks")
        
        all_data = []
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = self.config.get("max_workers", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_range = {executor.submit(self.collect_data_for_range, date_range): date_range for date_range in date_ranges}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_range), total=len(date_ranges), desc="Collecting data"):
                date_range = future_to_range[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Error collecting data for range {date_range}: {e}")
        
        if not all_data:
            logger.error("No data collected from any source")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_data = combined_data.sort_values("timestamp")
        
        # Remove duplicates
        combined_data = combined_data.drop_duplicates(subset=["timestamp", "source"])
        
        # Save to file
        combined_data.to_csv(self.historical_data_file, index=False)
        
        logger.info(f"Collected {len(combined_data)} rows of historical data")
        logger.info(f"Data saved to {self.historical_data_file}")
        
        return combined_data
    
    def process_data_for_lightning_ai(self, output_file=None):
        """Process the collected data for Lightning AI."""
        if not output_file:
            output_file = os.path.join(self.data_dir, "nq_historical_data_lightning.csv")
        
        # Load data
        if os.path.exists(self.historical_data_file):
            df = pd.read_csv(self.historical_data_file)
        else:
            logger.error(f"Historical data file not found: {self.historical_data_file}")
            return None
        
        if df.empty:
            logger.warning("No data to process")
            return None
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Group by minute and source
        df["minute"] = df["timestamp"].dt.floor("min")
        
        # Process data by source
        sources = df["source"].unique()
        processed_dfs = []
        
        for source in sources:
            source_df = df[df["source"] == source].copy()
            
            # Aggregate by minute
            agg_df = source_df.groupby("minute").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).reset_index()
            
            # Add source column
            agg_df["source"] = source
            
            processed_dfs.append(agg_df)
        
        # Combine processed data
        if not processed_dfs:
            logger.warning("No processed data available")
            return None
        
        processed_data = pd.concat(processed_dfs, ignore_index=True)
        
        # Create a complete minute-by-minute timeline
        min_date = processed_data["minute"].min()
        max_date = processed_data["minute"].max()
        
        # Create a complete timeline with 1-minute intervals
        timeline = pd.date_range(start=min_date, end=max_date, freq="1min")
        timeline_df = pd.DataFrame({"minute": timeline})
        
        # For each source, create a complete timeline and merge with actual data
        final_dfs = []
        
        for source in sources:
            source_data = processed_data[processed_data["source"] == source].copy()
            
            # Merge with timeline
            merged_df = pd.merge(timeline_df, source_data, on="minute", how="left")
            
            # Forward fill missing values
            merged_df = merged_df.fillna(method="ffill")
            
            # Add source column if it was dropped
            if "source" not in merged_df.columns:
                merged_df["source"] = source
            
            final_dfs.append(merged_df)
        
        # Combine final data
        final_data = pd.concat(final_dfs, ignore_index=True)
        
        # Rename minute column to timestamp
        final_data = final_data.rename(columns={"minute": "timestamp"})
        
        # Sort by timestamp and source
        final_data = final_data.sort_values(["timestamp", "source"])
        
        # Save to file
        final_data.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(final_data)} rows of data for Lightning AI")
        logger.info(f"Data saved to {output_file}")
        
        return final_data
    
    def run(self):
        """Run the historical data collection."""
        try:
            # Initialize Selenium if needed
            if "investing" in self.sources and self.config.get("use_selenium", True):
                self._initialize_selenium()
            
            # Collect historical data
            self.collect_all_historical_data()
            
            # Process data for Lightning AI
            self.process_data_for_lightning_ai()
            
            logger.info("Historical data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Error in historical data collection: {e}")
        
        finally:
            # Close Selenium
            self._close_selenium()

def main():
    """Main function."""
    agent = NQHistoricalDataAgent()
    agent.run()

if __name__ == "__main__":
    main()
