import yfinance as yf
import pandas as pd

def fetch_data_to_csv(tickers, start_date, end_date, filename="stock_data.csv"):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data.to_csv(filename)

tickers = ["XLK", "XLC", "XLY", "XLE", "XLI", "XLB", "XLF", "XLRE", "XLP", "XLU", "XLV", "FXI", "GDX"]
start_date = "2018-08-10"
end_date = "2023-08-10"

fetch_data_to_csv(tickers, start_date, end_date)
