import yfinance as yf
import pandas as pd
import numpy as np
import time

def get_user_input():
    tickers = input("Enter tickers separated by commas (e.g., AAPL,MSFT,BHP.AX): ").upper().split(',')
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")
    benchmark = input("Enter benchmark ticker (default: SPY): ").upper().strip() or "SPY"
    return [t.strip() for t in tickers], start, end, benchmark

def download_data(tickers, start, end, benchmark):
    try:
        all_tickers = tickers + [benchmark]
        data = yf.download(all_tickers, start=start, end=end)
        if 'Close' in data:
            return data['Close'], benchmark
        else:
            print("'Close' column not found in downloaded data.")
            return pd.DataFrame(), benchmark
    except Exception as e:
        print(f"Error downloading price data: {e}")
        return pd.DataFrame(), benchmark

def calculate_metrics(price_data, tickers, benchmark):
    returns = price_data.pct_change().dropna()
    avg_returns = returns[tickers].mean() * 252
    volatility = returns[tickers].std() * np.sqrt(252)
    sharpe = avg_returns / volatility

    # Beta calculation
    market_returns = returns[benchmark]
    beta_dict = {}
    for ticker in tickers:
        try:
            cov = np.cov(returns[ticker], market_returns)[0][1]
            var = np.var(market_returns)
            beta = cov / var
            beta_dict[ticker] = beta
        except Exception as e:
            print(f"Error calculating beta for {ticker}: {e}")
            beta_dict[ticker] = np.nan
    beta_series = pd.Series(beta_dict)

    # Analyst upside
    upside_dict = {}
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            time.sleep(1.5)  # avoid rate limits
            target_data = tkr.get_analyst_price_targets()
            target_price = target_data.get('mean')
            if target_price is not None:
                current_price = tkr.history(period="1d")['Close'].iloc[-1]
                upside = ((target_price - current_price) / current_price) * 100
                upside_dict[ticker] = upside
            else:
                print(f"No analyst target for {ticker}")
                upside_dict[ticker] = np.nan
        except Exception as e:
            print(f"Error fetching target for {ticker}: {e}")
            upside_dict[ticker] = np.nan
    upside_series = pd.Series(upside_dict)

    # Combine metrics
    metrics = pd.DataFrame({
        'Sharpe Ratio': sharpe,
        'Beta': beta_series,
        'Upside %': upside_series
    })

    return metrics, returns

def main():
    tickers, start, end, benchmark = get_user_input()
    prices, benchmark = download_data(tickers, start, end, benchmark)

    if prices.empty:
        print("No price data. Exiting.")
        return

    print("\n✅ Price data fetched.")
    metrics, returns = calculate_metrics(prices, tickers, benchmark)
    print("\n📊 Metrics:\n", metrics)

if __name__ == "__main__":
    main()