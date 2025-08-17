"""
Web server for the investment portfolio dashboard.

This FastAPI application exposes a simple web interface to manage investment
accounts, tickers and transactions, and to display portfolio summaries and
charts. It uses Jinja2 templates for rendering HTML and stores data in a
SQLite database via helper functions defined in `database.py`.

The server defines routes for:

* `/` – dashboard home displaying summary statistics and portfolio history.
* `/accounts` – list, create, edit and delete investment accounts.
* `/tickers` – list, create, edit and delete ticker definitions.
* `/transactions` – list, create, edit and delete transactions.
* `/update_prices` – manually trigger an update of ticker prices using
  yfinance (if available) or proxy logic for non‑public tickers.

The application attempts to import yfinance on demand; if not available,
price update functionality will gracefully skip remote fetching. Users should
install yfinance in their own environment to enable automatic price updates.

To run the application locally:

```bash
cd /path/to/portfolio_app
python -m uvicorn srcmain:app --reload --port 8000
```

Then navigate to http://localhost:8000 in a browser.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, date
from typing import Any, Dict, Optional, List
import base64
import io
import matplotlib
matplotlib.use("Agg")  # use non‑interactive backend for server
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import database as db
from pathlib import Path

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore


app = FastAPI(title="Investment Portfolio Dashboard")

# Mount static directory for CSS/JS
app.mount("/static", StaticFiles(directory=str((Path(__file__).parent / "static").resolve())), name="static")

# Set up Jinja2 environment
templates = Jinja2Templates(directory=str((Path(__file__).parent / "templates").resolve()))


@app.on_event("startup")
async def startup_event() -> None:
    """Ensure database schema exists on startup."""
    conn = db.get_db()
    db.init_db(conn)
    conn.close()


###############################
# Helpers for price updating   #
###############################

async def fetch_latest_price(symbol: str) -> Optional[float]:
    """Fetch the latest closing price for a symbol via yfinance.

    Args:
        symbol: ticker symbol, e.g. "AAPL"

    Returns:
        latest price as float or None if unavailable
    """
    if yf is None:
        return None
    try:
        ticker = yf.Ticker(symbol)
        # Use history to fetch last close; period=1d to get last trading day
        hist = ticker.history(period="1d")
        if not hist.empty:
            # Use 'Close' column
            price = hist['Close'].iloc[-1]
            return float(price)
    except Exception:
        return None
    return None


async def update_all_prices(conn: sqlite3.Connection) -> None:
    """Update price_history for all tickers.

    For each ticker, attempt to fetch its latest price from yfinance if the
    symbol is public. For tickers that specify a `proxy_symbol`, fetch the
    proxy price and multiply by the conversion ratio. The resulting price is
    inserted into `price_history` with today's date. If fetching fails,
    the ticker is skipped.

    Args:
        conn: database connection
    """
    tickers = db.list_tickers(conn)
    today_str = date.today().isoformat()
    for tk in tickers:
        symbol = tk["symbol"]
        # Check if there is a proxy ratio effective for today
        proxy_entry = db.get_effective_proxy(conn, tk["id"], today_str)
        price: Optional[float] = None
        if proxy_entry:
            proxy_symbol = proxy_entry["proxy_symbol"]
            ratio = proxy_entry["conversion_ratio"] or 1.0
            p = await fetch_latest_price(proxy_symbol)
            if p is not None:
                price = p * ratio
        else:
            proxy_symbol_fallback = tk["proxy_symbol"]
            ratio_fallback = tk["conversion_ratio"] or 1.0
            if proxy_symbol_fallback:
                p = await fetch_latest_price(proxy_symbol_fallback)
                if p is not None:
                    price = p * ratio_fallback
            else:
                p = await fetch_latest_price(symbol)
                if p is not None:
                    price = p
        if price is not None:
            # Insert price into history
            db.insert_price(conn, tk["id"], today_str, price)

# Helper to generate a PNG line chart encoded as a data URI
def generate_line_chart(labels: List[str], data: List[float]) -> str:
    """Generate a PNG line chart and return a base64 data URI.

    Args:
        labels: list of date strings on the x axis
        data: list of values for the y axis

    Returns:
        data URI string representing the PNG image
    """
    # Convert strings to dates for better spacing (matplotlib will format automatically)
    x_vals = [datetime.fromisoformat(d) for d in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, data, marker='o', linestyle='-', color='#007ACC')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Portfolio Value History (Last 60 Days)")
    # Rotate dates for readability
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


###############################
# Route definitions           #
###############################

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the dashboard home page.

    Displays summary statistics for each account and the total portfolio, along
    with a simple time series chart of total portfolio value over the last 60
    days. The chart uses Chart.js on the client side.
    """
    conn = db.get_db()
    accounts = db.list_accounts(conn)
    account_summaries: List[Dict[str, Any]] = []
    total_summary = {
        "total_value": 0.0,
        "total_cost": 0.0,
        "gain": 0.0,
        "stock_value": 0.0,
        "bond_value": 0.0,
        "other_value": 0.0,
    }
    for acc in accounts:
        summary = db.compute_account_summary(conn, acc["id"])
        summary["name"] = acc["name"]
        account_summaries.append(summary)
        # accumulate totals
        total_summary["total_value"] += summary["total_value"]
        total_summary["total_cost"] += summary["total_cost"]
        total_summary["gain"] += summary["gain"]
        total_summary["stock_value"] += summary["stock_value"]
        total_summary["bond_value"] += summary["bond_value"]
        total_summary["other_value"] += summary["other_value"]
    # Build date range for the last 60 days
    today = date.today()
    dates = [ (today - timedelta(days=i)).isoformat() for i in range(59, -1, -1) ]
    history = db.compute_portfolio_history(conn, dates)
    conn.close()
    # Extract for chart
    chart_labels = [ d for d, _ in history ]
    chart_data = [ v for _, v in history ]
    chart_uri = generate_line_chart(chart_labels, chart_data)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "account_summaries": account_summaries,
            "total_summary": total_summary,
            "chart_uri": chart_uri,
        },
    )


###############################
# Accounts management routes  #
###############################

@app.get("/accounts", response_class=HTMLResponse)
async def accounts_list(request: Request) -> HTMLResponse:
    """Render the list of accounts with an option to add a new one."""
    conn = db.get_db()
    accounts = db.list_accounts(conn)
    conn.close()
    return templates.TemplateResponse(
        "accounts.html",
        {"request": request, "accounts": accounts},
    )


@app.post("/accounts")
async def accounts_post(request: Request) -> RedirectResponse:
    """Handle account creation from form submission.

    Instead of relying on FastAPI's Form parser (which requires the optional
    `python-multipart` library), we manually parse the body. Forms use
    `application/x-www-form-urlencoded` encoding so we can decode the body
    using urllib.parse.parse_qs.
    """
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    name = data.get("name", "").strip()
    taxation = data.get("taxation") or None
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    short_rate = to_float(data.get("short_term_tax_rate"))
    long_rate = to_float(data.get("long_term_tax_rate"))
    if name:
        conn = db.get_db()
        db.create_account(conn, name, taxation, short_rate, long_rate)
        conn.close()
    return RedirectResponse("/accounts", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/accounts/edit/{account_id}", response_class=HTMLResponse)
async def accounts_edit(request: Request, account_id: int) -> HTMLResponse:
    """Render edit form for an account."""
    conn = db.get_db()
    account = db.get_account(conn, account_id)
    conn.close()
    if not account:
        return RedirectResponse("/accounts")
    return templates.TemplateResponse(
        "account_edit.html",
        {"request": request, "account": account},
    )


@app.post("/accounts/edit/{account_id}")
async def accounts_edit_post(request: Request, account_id: int) -> RedirectResponse:
    """Handle submission of edited account details."""
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    name = data.get("name", "").strip()
    taxation = data.get("taxation") or None
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    short_rate = to_float(data.get("short_term_tax_rate"))
    long_rate = to_float(data.get("long_term_tax_rate"))
    if name:
        conn = db.get_db()
        db.update_account(conn, account_id, name, taxation, short_rate, long_rate)
        conn.close()
    return RedirectResponse("/accounts", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/accounts/delete/{account_id}")
async def accounts_delete(account_id: int) -> RedirectResponse:
    """Delete an account and redirect back to the accounts list."""
    conn = db.get_db()
    db.delete_account(conn, account_id)
    conn.close()
    return RedirectResponse("/accounts", status_code=status.HTTP_303_SEE_OTHER)


###############################
# Tickers management routes   #
###############################

@app.get("/tickers", response_class=HTMLResponse)
async def tickers_list(request: Request) -> HTMLResponse:
    """Render the list of tickers with an option to add a new one."""
    conn = db.get_db()
    tickers = db.list_tickers(conn)
    conn.close()
    return templates.TemplateResponse(
        "tickers.html",
        {"request": request, "tickers": tickers},
    )


@app.post("/tickers")
async def tickers_post(request: Request) -> RedirectResponse:
    """Handle ticker creation from form submission."""
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    symbol = data.get("symbol", "").strip()
    classification = data.get("classification", "").strip()
    proxy_symbol = data.get("proxy_symbol") or None
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    conversion_ratio = to_float(data.get("conversion_ratio"))
    ratio_timestamp = data.get("ratio_timestamp") or None
    if symbol and classification:
        conn = db.get_db()
        db.create_ticker(conn, symbol, classification, proxy_symbol, conversion_ratio, ratio_timestamp)
        conn.close()
    return RedirectResponse("/tickers", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/tickers/edit/{ticker_id}", response_class=HTMLResponse)
async def tickers_edit(request: Request, ticker_id: int) -> HTMLResponse:
    """Render edit form for a ticker, including proxy history."""
    conn = db.get_db()
    ticker = db.get_ticker(conn, ticker_id)
    proxies = db.list_proxy_history(conn, ticker_id)
    conn.close()
    if not ticker:
        return RedirectResponse("/tickers")
    return templates.TemplateResponse(
        "ticker_edit.html",
        {
            "request": request,
            "ticker": ticker,
            "proxies": proxies,
        },
    )


@app.post("/tickers/edit/{ticker_id}")
async def tickers_edit_post(request: Request, ticker_id: int) -> RedirectResponse:
    """Handle submission of edited ticker details."""
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    symbol = data.get("symbol", "").strip()
    classification = data.get("classification", "").strip()
    proxy_symbol = data.get("proxy_symbol") or None
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    conversion_ratio = to_float(data.get("conversion_ratio"))
    ratio_timestamp = data.get("ratio_timestamp") or None
    if symbol and classification:
        conn = db.get_db()
        db.update_ticker(conn, ticker_id, symbol, classification, proxy_symbol, conversion_ratio, ratio_timestamp)
        conn.close()
    return RedirectResponse("/tickers", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/tickers/delete/{ticker_id}")
async def tickers_delete(ticker_id: int) -> RedirectResponse:
    """Delete a ticker and redirect back to the tickers list."""
    conn = db.get_db()
    db.delete_ticker(conn, ticker_id)
    conn.close()
    return RedirectResponse("/tickers", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/tickers/{ticker_id}/proxy")
async def tickers_add_proxy(request: Request, ticker_id: int) -> RedirectResponse:
    """Handle adding a new proxy ratio entry for a ticker."""
    body_bytes = await request.body()
    data: Dict[str, str] = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    proxy_symbol = data.get("proxy_symbol", "").strip()
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    ratio = to_float(data.get("conversion_ratio")) or 1.0
    ratio_timestamp = data.get("ratio_timestamp", "").strip()
    if proxy_symbol and ratio_timestamp:
        conn = db.get_db()
        db.add_proxy_history(conn, ticker_id, proxy_symbol, ratio, ratio_timestamp)
        conn.close()
    return RedirectResponse(f"/tickers/edit/{ticker_id}", status_code=status.HTTP_303_SEE_OTHER)


###############################
# Transactions management     #
###############################

@app.get("/transactions", response_class=HTMLResponse)
async def transactions_list(request: Request) -> HTMLResponse:
    """Render the list of transactions and a form to add new ones."""
    conn = db.get_db()
    transactions = db.list_transactions(conn)
    accounts = db.list_accounts(conn)
    tickers = db.list_tickers(conn)
    conn.close()
    return templates.TemplateResponse(
        "transactions.html",
        {
            "request": request,
            "transactions": transactions,
            "accounts": accounts,
            "tickers": tickers,
        },
    )


@app.post("/transactions")
async def transactions_post(request: Request) -> RedirectResponse:
    """Handle transaction creation from form submission."""
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    date_str = data.get("date", "")
    type_str = data.get("type", "")
    account_id = int(data.get("account_id", 0)) if data.get("account_id") else 0
    # ticker_id may be empty for cash transactions
    ticker_raw = data.get("ticker_id")
    ticker_id = int(ticker_raw) if ticker_raw not in (None, "") else 0
    units = to_float(data.get("units")) or 0.0
    value = to_float(data.get("value")) or 0.0
    price_per_unit = to_float(data.get("price_per_unit")) or 0.0
    fees = to_float(data.get("fees"))
    realized_tax = to_float(data.get("realized_tax"))
    cost_calc = to_float(data.get("cost_calculation"))
    if date_str and type_str and account_id:
        # Determine sign of units based on transaction type
        t_upper = type_str.upper()
        # For cash transactions (deposit/withdraw) we ignore ticker and units
        if t_upper == "SELL":
            units = -abs(units)
        elif t_upper == "BUY":
            units = abs(units)
        elif t_upper in ("DEPOSIT", "WITHDRAW"):
            ticker_id = 0
            units = 0.0
        conn = db.get_db()
        # Perform validations
        error_msg: Optional[str] = None
        if t_upper in ("BUY", "WITHDRAW"):
            cash = db.compute_account_cash(conn, account_id)
            # For withdraw, we use value; for buy we also use value as cost
            if cash < value:
                error_msg = f"Insufficient cash: available {cash:.2f}, required {value:.2f}"
        elif t_upper == "SELL":
            # Check units available
            if ticker_id:
                units_available = db.compute_units(conn, account_id, ticker_id)
                if units_available < abs(units):
                    error_msg = f"Insufficient shares: available {units_available:.4f}, trying to sell {abs(units):.4f}"
            else:
                error_msg = "Sell transaction requires a ticker"
        # If error, close conn and render error page
        if error_msg:
            # Re-render transactions page with error
            transactions = db.list_transactions(conn)
            accounts = db.list_accounts(conn)
            tickers = db.list_tickers(conn)
            conn.close()
            return templates.TemplateResponse(
                "transactions.html",
                {
                    "request": request,
                    "transactions": transactions,
                    "accounts": accounts,
                    "tickers": tickers,
                    "error": error_msg,
                },
            )
        # Otherwise insert transaction
        db.create_transaction(
            conn,
            date_str,
            t_upper,
            account_id,
            ticker_id if ticker_id != 0 else None,
            units,
            value,
            price_per_unit,
            fees,
            realized_tax,
            cost_calc,
        )
        conn.close()
    return RedirectResponse("/transactions", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/transactions/edit/{transaction_id}", response_class=HTMLResponse)
async def transactions_edit(request: Request, transaction_id: int) -> HTMLResponse:
    """Render form for editing a transaction."""
    conn = db.get_db()
    transaction = db.get_transaction(conn, transaction_id)
    accounts = db.list_accounts(conn)
    tickers = db.list_tickers(conn)
    conn.close()
    if not transaction:
        return RedirectResponse("/transactions")
    return templates.TemplateResponse(
        "transaction_edit.html",
        {
            "request": request,
            "transaction": transaction,
            "accounts": accounts,
            "tickers": tickers,
        },
    )


@app.post("/transactions/edit/{transaction_id}")
async def transactions_edit_post(request: Request, transaction_id: int) -> RedirectResponse:
    """Handle submission of edited transaction details."""
    body_bytes = await request.body()
    data = {}
    if body_bytes:
        import urllib.parse as _urlparse
        parsed = _urlparse.parse_qs(body_bytes.decode(), keep_blank_values=True)
        data = {k: v[0] for k, v in parsed.items()}
    def to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None
    date_str = data.get("date", "")
    type_str = data.get("type", "")
    account_id = int(data.get("account_id", 0)) if data.get("account_id") else 0
    ticker_raw = data.get("ticker_id")
    ticker_id = int(ticker_raw) if ticker_raw not in (None, "") else 0
    units = to_float(data.get("units")) or 0.0
    value_f = to_float(data.get("value")) or 0.0
    price_per_unit_f = to_float(data.get("price_per_unit")) or 0.0
    fees = to_float(data.get("fees"))
    realized_tax = to_float(data.get("realized_tax"))
    cost_calc = to_float(data.get("cost_calculation"))
    if date_str and type_str and account_id:
        t_upper = type_str.upper()
        if t_upper == "SELL":
            units = -abs(units)
        elif t_upper == "BUY":
            units = abs(units)
        elif t_upper in ("DEPOSIT", "WITHDRAW"):
            ticker_id = 0
            units = 0.0
        conn = db.get_db()
        error_msg: Optional[str] = None
        if t_upper in ("BUY", "WITHDRAW"):
            cash = db.compute_account_cash(conn, account_id, exclude_transaction_id=transaction_id)
            if cash < value_f:
                error_msg = f"Insufficient cash: available {cash:.2f}, required {value_f:.2f}"
        elif t_upper == "SELL":
            if ticker_id:
                units_available = db.compute_units(conn, account_id, ticker_id, exclude_transaction_id=transaction_id)
                if units_available < abs(units):
                    error_msg = f"Insufficient shares: available {units_available:.4f}, trying to sell {abs(units):.4f}"
            else:
                error_msg = "Sell transaction requires a ticker"
        if error_msg:
            accounts = db.list_accounts(conn)
            tickers = db.list_tickers(conn)
            transaction = db.get_transaction(conn, transaction_id)
            conn.close()
            return templates.TemplateResponse(
                "transaction_edit.html",
                {
                    "request": request,
                    "transaction": transaction,
                    "accounts": accounts,
                    "tickers": tickers,
                    "error": error_msg,
                },
            )
        # Otherwise perform update
        db.update_transaction(
            conn,
            transaction_id,
            date_str,
            t_upper,
            account_id,
            ticker_id if ticker_id != 0 else None,
            units,
            value_f,
            price_per_unit_f,
            fees,
            realized_tax,
            cost_calc,
        )
        conn.close()
    return RedirectResponse("/transactions", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/transactions/delete/{transaction_id}")
async def transactions_delete(transaction_id: int) -> RedirectResponse:
    """Delete a transaction."""
    conn = db.get_db()
    db.delete_transaction(conn, transaction_id)
    conn.close()
    return RedirectResponse("/transactions", status_code=status.HTTP_303_SEE_OTHER)


###############################
# Price update route          #
###############################

@app.get("/update_prices")
async def update_prices() -> RedirectResponse:
    """Trigger an update of all ticker prices.

    This route asynchronously iterates over all tickers and fetches the latest
    price using yfinance. It then inserts a price point into `price_history`
    with today's date. If yfinance is not installed or a fetch fails, the
    corresponding ticker is skipped silently. After completion the user is
    redirected back to the home page.
    """
    conn = db.get_db()
    await update_all_prices(conn)
    conn.close()
    # Redirect back to homepage after update
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)