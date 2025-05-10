import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_user_input():
    tickers = input("Enter tickers separated by commas (e.g., AAPL,MSFT,BHP.AX): ").upper().split(',')
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")
    benchmark = input("Enter benchmark ticker (S&P500 - SPY, ASX200 - ^AXJO): ").upper().strip() or "SPY"
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

def plot_correlation_heatmap(returns):
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Asset Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_indexed_prices(price_data, tickers):
    plt.figure(figsize=(12, 6))

    colors = plt.colormaps["tab10"].resampled(len(tickers))
    indexed = price_data / price_data.iloc[0]

    for i, ticker in enumerate(tickers):
        # Plot indexed line
        plt.plot(indexed.index, indexed[ticker], label=ticker, color=colors(i))

        # Fetch analyst price target
        try:
            tkr = yf.Ticker(ticker)
            target_data = tkr.get_analyst_price_targets()
            target_price = target_data.get('mean')
            if target_price:
                first_price = price_data[ticker].iloc[0]
                target_indexed = target_price / first_price

                # Plot as a dot (same color)
                plt.scatter(indexed.index[-1], target_indexed,
                            color=colors(i), marker='o', s=80, edgecolors='black',
                            label=f"{ticker} target")
        except Exception as e:
            print(f"Could not fetch target for {ticker}: {e}")

    plt.title("Indexed Price Performance with Analyst Targets (Start = 1)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    tickers, start, end, benchmark = get_user_input()
    prices, benchmark = download_data(tickers, start, end, benchmark)

    if prices.empty:
        print("No price data. Exiting.")
        return

    print("\n Price data fetched.")
    metrics, returns = calculate_metrics(prices, tickers, benchmark)
    print("\n Metrics:\n", metrics)
    correlation_matrix = returns[tickers].corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)
    plot_correlation_heatmap(returns[tickers])
    plot_indexed_prices(prices[tickers], tickers)

if __name__ == "__main__":
    main()