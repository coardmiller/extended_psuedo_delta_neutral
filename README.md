# 🤖 Extended BTC-ETH Hedge Bot

This is a trading bot for the [Extended Exchange](https://extended.exchange/) that implements a pseudo delta-neutral (market-neutral) long/short strategy on BTC-PERP and ETH-PERP. It aims to generate returns from the basis between the two assets while remaining hedged against overall market movements.

The bot is data-driven, using historical price data to calculate a hedge ratio that determines the relative sizes of the long BTC and short ETH positions.

## ✨ Features

- **📈 Pseudo Delta-Neutral Strategy**: Goes long on BTC and short on ETH to hedge against directional market risk. While it aims to minimize directional risk, it's important to note that this is not a perfect delta-neutral hedge.
- **📊 Data-Driven Hedge Ratio**: Automatically calculates a minimum-variance hedge ratio based on the last 365 days of daily returns to determine position sizes.
- **💾 State Management**: Saves the bot's state (open positions, hedge ratio, etc.) to a `state.json` file, allowing it to resume operations after a restart.
- **🧠 Intelligent Reconciliation**: At startup, the bot checks if its saved state is consistent with the positions on the exchange and makes adjustments if necessary.
- **🛡️ Stop-Loss Protection**: Includes a configurable stop-loss to protect against excessive losses on either leg of the pair.
- **🔌 Safe Shutdown**: Designed to leave positions open when interrupted (e.g., with Ctrl+C) to prevent unintended closing of positions.

## Screenshot

![Bot in action](screen.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An account on the Extended Exchange: https://extended.exchange/
- API key credentials for your Extended account (account id, API key, API secret)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd hedge_bot
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the root of the project and add your Extended account id, API key, and API secret. You can copy the `.env.example` file to get started:
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file with your credentials:
    ```env
    EXTENDED_ACCOUNT_ID=YOUR_ACCOUNT_ID
    EXTENDED_API_KEY=YOUR_API_KEY
    EXTENDED_API_SECRET=YOUR_API_SECRET
    ```

- **Optional:** If your region requires a different API hostname you can set `EXTENDED_REST_BASE_URL` (or comma-separated `EXTENDED_REST_BASE_URLS`) in your `.env`. The bot will automatically try each base when making requests.
- **Optional:** Provide Pacifica market data credentials if Extended does not expose historical klines in your environment. Set `PACIFICA_REST_BASE_URL` (or `PACIFICA_REST_BASE_URLS`) to override the default `https://api.pacifica.fi/api/v1` endpoint and populate `PACIFICA_API_KEY` if your account requires an API token. The bot will fetch hedge-ratio history from Pacifica only when Extended responds with empty/404 kline payloads.

2.  **Edit the `config.json` file** to configure the bot's parameters:
    - `capital_pct`: The percentage of your account equity to use for the strategy.
    - `leverage`: The leverage to use for your positions.
    - `refresh_hours`: The number of hours between position refreshes.
    - `stoploss_pct`: The stop-loss percentage for each leg of the pair.
    - `slippage_bps`: The allowed slippage in basis points for market orders.
    - `loop_sleep_seconds`: The time in seconds the bot waits between checks in the main loop.
    - `status_interval_seconds`: The frequency in seconds of the status log output.

## ▶️ Usage

To run the bot, simply execute the `main.py` script:

```bash
python main.py
```

The bot will start, cancel any existing open orders, and then either open a new pair of positions or reconcile existing ones. It will then run continuously, checking for stop-loss conditions and refreshing the positions at the configured interval.

## 🐳 Running with Docker

Alternatively, you can run the bot in a Docker container using the provided `docker-compose.yml` file. This is a convenient way to run the bot in a controlled environment.

1.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```

2.  **Run the bot in detached mode:**
    ```bash
    docker-compose up -d
    ```

To view the bot's logs, you can use the following command:
```bash
docker-compose logs -f
```

## 🧪 Testing

The project includes a suite of tests to verify the core logic. To run the tests, use the `run_tests.py` script:

```bash
python run_tests.py
```

## 🔄 Avoiding Merge Conflicts

Keeping your branch up to date with `main` is the easiest way to prevent repeat conflict resolution. A lightweight workflow that
works well with this repository is:

1. Fetch the latest changes before you start working:
   ```bash
   git fetch origin
   ```
2. Rebase your feature branch on top of the most recent `main` history (this keeps the commit graph linear and avoids repeated
   merge commits):
   ```bash
   git rebase origin/main
   ```
3. If you have already pushed your branch, force-push the rebased history once the rebase completes:
   ```bash
   git push --force-with-lease
   ```

Doing this at the start of each work session—and again right before opening a pull request—ensures you only need to reconcile
each upstream change once. When collaborating with others, the `--force-with-lease` flag guarantees you do not overwrite their
work while still keeping your branch conflict-free.
