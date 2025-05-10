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

def get_user_weights():
    print("\n Assign weights to each metric (total should add up to 100)")
    try:
        w_sharpe = float(input("Weight for Sharpe Ratio: "))
        w_beta = float(input("Weight for Beta: "))
        w_upside = float(input("Weight for Upside: "))
    except ValueError:
        print("Invalid input. Please enter numbers.")
        return get_user_weights()

    total = w_sharpe + w_beta + w_upside
    if total != 100:
        print(f"Total = {total}. Normalizing weights to sum to 100.")
        w_sharpe = w_sharpe / total * 100
        w_beta = w_beta / total * 100
        w_upside = w_upside / total * 100

    return {'Sharpe Ratio': w_sharpe, 'Beta': w_beta, 'Upside %': w_upside}

def get_backtest_dates():
    print("\n Backtest Portfolio Performance")
    start = input("Enter new start date (YYYY-MM-DD): ")
    end = input("Enter new end date (YYYY-MM-DD): ")
    return start, end


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

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def calculate_scores(metrics, weights):
    # Min-max normalize Sharpe and Upside
    normalized = pd.DataFrame()
    normalized['Sharpe Ratio'] = min_max_normalize(metrics['Sharpe Ratio'])
    normalized['Upside %'] = min_max_normalize(metrics['Upside %'])

    # Penalize Beta > 1 only
    beta_penalty = metrics['Beta'].copy()
    beta_penalty[beta_penalty <= 1] = 0
    beta_penalty[beta_penalty > 1] = beta_penalty[beta_penalty > 1] - 1
    normalized['Beta'] = -min_max_normalize(beta_penalty)

    # Weighted score
    score = (
        normalized['Sharpe Ratio'] * weights['Sharpe Ratio'] +
        normalized['Beta'] * weights['Beta'] +
        normalized['Upside %'] * weights['Upside %']
    )

    metrics['Score'] = score
    metrics['Weight'] = score / score.sum()

    return metrics

def backtest_portfolio(tickers, weights, benchmark, start, end):
    all_tickers = tickers + [benchmark]
    data = yf.download(all_tickers, start=start, end=end)['Close']

    # Normalize and calculate
    returns = data.pct_change().dropna()
    port_returns = (returns[tickers] * weights).sum(axis=1)
    port_value = (1 + port_returns).cumprod()
    bench_value = (1 + returns[benchmark]).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(port_value, label="Portfolio")
    plt.plot(bench_value, label=benchmark)
    plt.title("📊 Portfolio vs Benchmark (Indexed)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Backtest stats
    days = len(port_value)
    cagr_port = (port_value.iloc[-1] / port_value.iloc[0]) ** (252 / days) - 1
    cagr_bench = (bench_value.iloc[-1] / bench_value.iloc[0]) ** (252 / days) - 1

    total_return_port = port_value.iloc[-1] - 1
    total_return_bench = bench_value.iloc[-1] - 1
    active_return = total_return_port - total_return_bench

    sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
    beta = np.cov(port_returns, returns[benchmark])[0][1] / np.var(returns[benchmark])

    # Print stats
    print(f"\n Portfolio Stats ({start} to {end}):")
    print(f"  CAGR (Portfolio): {cagr_port:.2%}")
    print(f"  CAGR (Benchmark): {cagr_bench:.2%}")
    print(f"  Total Return: {total_return_port:.2%}")
    print(f"  Benchmark Return: {total_return_bench:.2%}")
    print(f"  Active Return: {active_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Beta to {benchmark}: {beta:.2f}")

def main():
    tickers, start, end, benchmark = get_user_input()
    prices, benchmark = download_data(tickers, start, end, benchmark)

    if prices.empty:
        print("No price data. Exiting.")
        return

    print("\n Price data fetched.")
    metrics, returns = calculate_metrics(prices, tickers, benchmark)
    print("\n Metrics:\n", metrics)
    weights = get_user_weights()
    metrics = calculate_scores(metrics, weights)
    print("\n Scored Metrics:\n", metrics[['Sharpe Ratio', 'Beta', 'Upside %', 'Score', 'Weight']])
    plot_correlation_heatmap(returns[tickers])
    plot_indexed_prices(prices[tickers], tickers)
    bt_start, bt_end = get_backtest_dates()
    portfolio_weights = metrics.set_index(metrics.index)['Weight']
    backtest_portfolio(
        tickers=tickers,
        weights=portfolio_weights,
        benchmark=benchmark,
        start=bt_start,
        end=bt_end
    )

if __name__ == "__main__":
    main()