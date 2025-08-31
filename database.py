"""
Database layer for the investment portfolio application.

This module encapsulates low‑level interactions with a SQLite database. It defines
functions for creating the schema, inserting and updating records, and
performing common queries required by the web application. We avoid using
external ORM libraries in order to keep the installation footprint minimal and
compatible with the execution environment, which lacks `sqlalchemy`. Instead
we use Python's built‑in `sqlite3` module with a row factory to return
dictionary‑like rows.

Tables created by this module:

* **accounts** – stores investment account definitions along with tax
  information (short and long term rates). Each row has an auto‑incrementing
  primary key.
* **tickers** – stores ticker metadata including whether the instrument is
  classified as a stock or bond. For non‑public tickers (such as proprietary
  funds inside a retirement account) a proxy ticker and conversion ratio may
  be specified along with a timestamp indicating when the ratio became
  effective.
* **transactions** – stores individual investment transactions. Each
  transaction is associated with an account and a ticker and records the
  quantity of units transacted, the total value, the price per unit, fees,
  realized taxes and cost basis. The `type` column is free‑form but
  conventionally stores values like "BUY", "SELL", "DEPOSIT", "WITHDRAW" etc.
* **price_history** – stores historical prices for tickers. When updating
  prices, a row is inserted with the date and closing price. The web
  application uses the most recent price when calculating portfolio values.

All date fields are stored as ISO 8601 strings (YYYY‑MM‑DD). Real numbers are
stored for monetary amounts, unit quantities and tax rates.

Usage:

>>> from portfolio_app.database import get_db, init_db
>>> conn = get_db()
>>> init_db(conn)
>>> # call other helper functions as needed
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DB_FILENAME = Path(__file__).resolve().parent / "portfolio.db"


def get_db() -> sqlite3.Connection:
    """Return a SQLite connection to the database.

    The connection uses a row factory so that query results behave like
    dictionaries. The database file lives alongside this module. A single
    connection should be reused per request in the web application to avoid
    creating too many connections.

    Returns:
        sqlite3.Connection: a connection object
    """
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not exist.

    This function is idempotent; it can be called multiple times without
    destroying existing data. It ensures all required tables for accounts,
    tickers, transactions and price history exist.

    Args:
        conn (sqlite3.Connection): an open connection
    """
    cur = conn.cursor()
    # Create accounts table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            taxation TEXT,
            short_term_tax_rate REAL,
            long_term_tax_rate REAL
        )
        """
    )
    # Create tickers table. We drop and recreate it to remove legacy proxy fields.
    cur.execute("DROP TABLE IF EXISTS tickers")
    cur.execute(
        """
        CREATE TABLE tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            classification TEXT
        )
        """
    )
    # Create transactions table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT,
            account_id INTEGER,
            ticker_id INTEGER,
            units REAL,
            value REAL,
            price_per_unit REAL,
            fees REAL,
            realized_tax REAL,
            cost_calculation REAL,
            FOREIGN KEY(account_id) REFERENCES accounts(id),
            FOREIGN KEY(ticker_id) REFERENCES tickers(id)
        )
        """
    )
    # Create price history table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_id INTEGER,
            date TEXT,
            price REAL,
            FOREIGN KEY(ticker_id) REFERENCES tickers(id)
        )
        """
    )

    # Create proxy history table for time‑series conversion ratios
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS proxy_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_id INTEGER,
            proxy_symbol TEXT,
            conversion_ratio REAL,
            ratio_timestamp TEXT,
            FOREIGN KEY(ticker_id) REFERENCES tickers(id)
        )
        """
    )
    conn.commit()


###########################
# Account operations      #
###########################

def list_accounts(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return all accounts.

    Args:
        conn: database connection

    Returns:
        list of rows representing accounts
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM accounts ORDER BY name")
    return cur.fetchall()


def get_account(conn: sqlite3.Connection, account_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single account by id.

    Args:
        conn: database connection
        account_id: id of the account

    Returns:
        row or None if not found
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
    return cur.fetchone()


def create_account(
    conn: sqlite3.Connection,
    name: str,
    taxation: Optional[str],
    short_term_tax_rate: Optional[float],
    long_term_tax_rate: Optional[float],
) -> int:
    """Insert a new account.

    Args:
        conn: database connection
        name: account name
        taxation: description of account taxation (e.g. "taxable", "401k")
        short_term_tax_rate: short term tax rate (decimal fraction, e.g. 0.24)
        long_term_tax_rate: long term tax rate

    Returns:
        newly inserted account id
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO accounts (name, taxation, short_term_tax_rate, long_term_tax_rate)
        VALUES (?, ?, ?, ?)
        """,
        (name, taxation, short_term_tax_rate, long_term_tax_rate),
    )
    conn.commit()
    return cur.lastrowid


def update_account(
    conn: sqlite3.Connection,
    account_id: int,
    name: str,
    taxation: Optional[str],
    short_term_tax_rate: Optional[float],
    long_term_tax_rate: Optional[float],
) -> None:
    """Update an existing account.

    Args:
        conn: database connection
        account_id: id of the account to update
        name: new name
        taxation: new taxation description
        short_term_tax_rate: new short term tax rate
        long_term_tax_rate: new long term tax rate
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE accounts
        SET name = ?, taxation = ?, short_term_tax_rate = ?, long_term_tax_rate = ?
        WHERE id = ?
        """,
        (name, taxation, short_term_tax_rate, long_term_tax_rate, account_id),
    )
    conn.commit()


def delete_account(conn: sqlite3.Connection, account_id: int) -> None:
    """Delete an account and its transactions.

    Args:
        conn: database connection
        account_id: id of the account to delete
    """
    cur = conn.cursor()
    # First delete related transactions
    cur.execute("DELETE FROM transactions WHERE account_id = ?", (account_id,))
    # Then delete the account
    cur.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    conn.commit()


###########################
# Ticker operations        #
###########################

def list_tickers(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return all tickers.

    Args:
        conn: database connection

    Returns:
        list of rows representing tickers
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickers ORDER BY symbol")
    return cur.fetchall()


def get_ticker(conn: sqlite3.Connection, ticker_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single ticker by id.

    Args:
        conn: database connection
        ticker_id: id of the ticker

    Returns:
        row or None
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickers WHERE id = ?", (ticker_id,))
    return cur.fetchone()


def get_ticker_by_symbol(conn: sqlite3.Connection, symbol: str) -> Optional[sqlite3.Row]:
    """Fetch a ticker by its symbol.

    Args:
        conn: database connection
        symbol: ticker symbol

    Returns:
        row or None
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickers WHERE symbol = ?", (symbol,))
    return cur.fetchone()


def create_ticker(
    conn: sqlite3.Connection,
    symbol: str,
    classification: str,
) -> int:
    """Insert a new ticker.

    Args:
        conn: database connection
        symbol: primary ticker symbol
        classification: 'stock' or 'bond'

    Returns:
        newly inserted ticker id
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tickers (symbol, classification)
        VALUES (?, ?)
        """,
        (symbol, classification),
    )
    conn.commit()
    return cur.lastrowid


def update_ticker(
    conn: sqlite3.Connection,
    ticker_id: int,
    symbol: str,
    classification: str,
) -> None:
    """Update an existing ticker.

    Args:
        conn: database connection
        ticker_id: id of the ticker to update
        symbol: new symbol
        classification: new classification
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tickers
        SET symbol = ?, classification = ?
        WHERE id = ?
        """,
        (symbol, classification, ticker_id),
    )
    conn.commit()


def delete_ticker(conn: sqlite3.Connection, ticker_id: int) -> None:
    """Delete a ticker and its price history and transactions.

    Args:
        conn: database connection
        ticker_id: id of the ticker to delete
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE ticker_id = ?", (ticker_id,))
    cur.execute("DELETE FROM price_history WHERE ticker_id = ?", (ticker_id,))
    cur.execute("DELETE FROM tickers WHERE id = ?", (ticker_id,))
    conn.commit()


###########################
# Transaction operations   #
###########################

def list_transactions(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return all transactions ordered by date descending.

    Args:
        conn: database connection

    Returns:
        list of rows representing transactions
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.*, a.name AS account_name, k.symbol AS ticker_symbol
        FROM transactions t
        LEFT JOIN accounts a ON t.account_id = a.id
        LEFT JOIN tickers k ON t.ticker_id = k.id
        ORDER BY date DESC, t.id DESC
        """
    )
    return cur.fetchall()


def list_transactions_for_account(conn: sqlite3.Connection, account_id: int) -> List[sqlite3.Row]:
    """Return all transactions for a specific account.

    Args:
        conn: database connection
        account_id: account id

    Returns:
        list of rows
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.*, k.symbol AS ticker_symbol
        FROM transactions t
        LEFT JOIN tickers k ON t.ticker_id = k.id
        WHERE t.account_id = ?
        ORDER BY date DESC, t.id DESC
        """,
        (account_id,),
    )
    return cur.fetchall()


def get_transaction(conn: sqlite3.Connection, transaction_id: int) -> Optional[sqlite3.Row]:
    """Fetch a transaction by id.

    Args:
        conn: database connection
        transaction_id: id of the transaction

    Returns:
        row or None
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
    return cur.fetchone()


def create_transaction(
    conn: sqlite3.Connection,
    date: str,
    type_: str,
    account_id: int,
    ticker_id: int,
    units: float,
    value: float,
    price_per_unit: float,
    fees: Optional[float],
    realized_tax: Optional[float],
    cost_calculation: Optional[float],
) -> int:
    """Insert a new transaction record.

    Args:
        conn: database connection
        date: transaction date (YYYY‑MM‑DD)
        type_: transaction type
        account_id: account reference
        ticker_id: ticker reference
        units: quantity transacted (positive for buy, negative for sell)
        value: total transacted value
        price_per_unit: price per unit
        fees: fees associated with the transaction
        realized_tax: realized tax component (if any)
        cost_calculation: optional cost basis for this transaction

    Returns:
        newly inserted transaction id
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions (
            date, type, account_id, ticker_id, units, value, price_per_unit, fees, realized_tax, cost_calculation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date,
            type_,
            account_id,
            ticker_id,
            units,
            value,
            price_per_unit,
            fees,
            realized_tax,
            cost_calculation,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_transaction(
    conn: sqlite3.Connection,
    transaction_id: int,
    date: str,
    type_: str,
    account_id: int,
    ticker_id: int,
    units: float,
    value: float,
    price_per_unit: float,
    fees: Optional[float],
    realized_tax: Optional[float],
    cost_calculation: Optional[float],
) -> None:
    """Update an existing transaction.

    Args:
        conn: database connection
        transaction_id: id of the transaction to update
        date: new date
        type_: new type
        account_id: account id
        ticker_id: ticker id
        units: quantity
        value: total value
        price_per_unit: price per unit
        fees: fees
        realized_tax: realized tax
        cost_calculation: cost basis
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transactions
        SET date = ?, type = ?, account_id = ?, ticker_id = ?, units = ?, value = ?, price_per_unit = ?, fees = ?, realized_tax = ?, cost_calculation = ?
        WHERE id = ?
        """,
        (
            date,
            type_,
            account_id,
            ticker_id,
            units,
            value,
            price_per_unit,
            fees,
            realized_tax,
            cost_calculation,
            transaction_id,
        ),
    )
    conn.commit()


def delete_transaction(conn: sqlite3.Connection, transaction_id: int) -> None:
    """Delete a transaction.

    Args:
        conn: database connection
        transaction_id: id of the transaction
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()


###########################
# Price history operations #
###########################

def insert_price(
    conn: sqlite3.Connection,
    ticker_id: int,
    date: str,
    price: float,
) -> int:
    """Insert a price point for a ticker.

    Args:
        conn: database connection
        ticker_id: id of the ticker
        date: date of the price (YYYY‑MM‑DD)
        price: closing price

    Returns:
        id of inserted row
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO price_history (ticker_id, date, price)
        VALUES (?, ?, ?)
        """,
        (ticker_id, date, price),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_price(conn: sqlite3.Connection, ticker_id: int) -> Optional[float]:
    """Return the most recent price for a ticker.

    Args:
        conn: database connection
        ticker_id: ticker id

    Returns:
        latest price or None if no price stored
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT price FROM price_history WHERE ticker_id = ? ORDER BY date DESC LIMIT 1",
        (ticker_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_price_history(conn: sqlite3.Connection, ticker_id: int) -> List[Tuple[str, float]]:
    """Return full price history for a ticker ordered by date.

    Args:
        conn: database connection
        ticker_id: id of the ticker

    Returns:
        list of (date, price)
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT date, price FROM price_history WHERE ticker_id = ? ORDER BY date ASC",
        (ticker_id,),
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def get_price_on_or_before(conn: sqlite3.Connection, ticker_id: int, date: str) -> Optional[float]:
    """Return the price for the latest date on or before the specified date.

    Useful for reconstructing portfolio values on arbitrary dates. If no price
    exists before the given date, returns None.

    Args:
        conn: database connection
        ticker_id: ticker id
        date: target date (YYYY‑MM‑DD)

    Returns:
        price or None
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT price FROM price_history WHERE ticker_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
        (ticker_id, date),
    )
    row = cur.fetchone()
    return row[0] if row else None

###############################
# Proxy history operations    #
###############################

def list_proxy_history(conn: sqlite3.Connection, ticker_id: int) -> List[sqlite3.Row]:
    """Return all proxy history entries for a ticker ordered by timestamp descending."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM proxy_history
        WHERE ticker_id = ?
        ORDER BY ratio_timestamp DESC
        """,
        (ticker_id,),
    )
    return cur.fetchall()


def add_proxy_history(
    conn: sqlite3.Connection,
    ticker_id: int,
    proxy_symbol: str,
    conversion_ratio: float,
    ratio_timestamp: str,
) -> int:
    """Insert a new proxy ratio entry for a ticker.

    Args:
        conn: database connection
        ticker_id: id of the ticker
        proxy_symbol: symbol used as proxy
        conversion_ratio: conversion ratio
        ratio_timestamp: date when this ratio becomes effective

    Returns:
        id of inserted row
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO proxy_history (ticker_id, proxy_symbol, conversion_ratio, ratio_timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (ticker_id, proxy_symbol, conversion_ratio, ratio_timestamp),
    )
    conn.commit()
    return cur.lastrowid


def get_effective_proxy(
    conn: sqlite3.Connection,
    ticker_id: int,
    as_of_date: str,
) -> Optional[sqlite3.Row]:
    """Get the most recent proxy ratio entry effective on or before a given date.

    Args:
        conn: database connection
        ticker_id: id of the ticker
        as_of_date: ISO date string

    Returns:
        the proxy_history row or None
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM proxy_history
        WHERE ticker_id = ? AND ratio_timestamp <= ?
        ORDER BY ratio_timestamp DESC
        LIMIT 1
        """,
        (ticker_id, as_of_date),
    )
    return cur.fetchone()


###############################
# Cash and unit computations  #
###############################

def compute_account_cash(
    conn: sqlite3.Connection,
    account_id: int,
    exclude_transaction_id: Optional[int] = None,
) -> float:
    """Compute available cash for an account.

    The cash balance is calculated as the sum of deposits and sells minus
    withdrawals and buys. Transactions associated with tickers indicate
    investment activity; deposits and withdrawals have no ticker. When editing
    an existing transaction the record can be excluded from the calculation by
    passing its id to avoid double counting.

    Args:
        conn: database connection
        account_id: account identifier
        exclude_transaction_id: optional id of a transaction to exclude from calculation

    Returns:
        available cash balance
    """
    cur = conn.cursor()
    query = "SELECT type, value FROM transactions WHERE account_id = ?"
    params: List[Any] = [account_id]
    if exclude_transaction_id is not None:
        query += " AND id != ?"
        params.append(exclude_transaction_id)
    cur.execute(query, params)
    rows = cur.fetchall()
    cash = 0.0
    for row in rows:
        ttype = (row[0] or "").upper()
        val = row[1] or 0.0
        if ttype == "DEPOSIT":
            cash += val
        elif ttype == "WITHDRAW":
            cash -= val
        elif ttype == "BUY":
            cash -= val
        elif ttype == "SELL":
            cash += val
        # other types could be ignored or extend here
    return cash


def compute_units(
    conn: sqlite3.Connection,
    account_id: int,
    ticker_id: int,
    exclude_transaction_id: Optional[int] = None,
) -> float:
    """Compute the number of units held for a given account and ticker.

    Args:
        conn: database connection
        account_id: account id
        ticker_id: ticker id
        exclude_transaction_id: optional transaction id to exclude

    Returns:
        float representing units held (positive for net long positions)
    """
    cur = conn.cursor()
    query = "SELECT SUM(units) FROM transactions WHERE account_id = ? AND ticker_id = ?"
    params: List[Any] = [account_id, ticker_id]
    if exclude_transaction_id is not None:
        query += " AND id != ?"
        params.append(exclude_transaction_id)
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else 0.0


###########################
# Portfolio computations   #
###########################

def compute_account_positions(
    conn: sqlite3.Connection,
    account_id: int,
    as_of_date: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """Compute positions for each ticker within an account.

    A position consists of the number of units held, cost basis (sum of
    purchased value), and current market value using the latest price in
    `price_history` or the price on `as_of_date` if provided.

    Args:
        conn: database connection
        account_id: account id
        as_of_date: optional ISO date to value the portfolio; uses most recent
            price on or before this date. If None, uses the latest price.

    Returns:
        dict mapping ticker_id -> {"units": float, "cost": float, "value": float, "ticker": row}
    """
    cur = conn.cursor()
    # Query all transactions for the account grouped by ticker
    cur.execute(
        """
        SELECT ticker_id, SUM(units) AS total_units, SUM(value) AS total_value, SUM(cost_calculation) AS total_cost
        FROM transactions
        WHERE account_id = ?
        GROUP BY ticker_id
        """,
        (account_id,),
    )
    rows = cur.fetchall()
    positions: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        ticker_id = row[0]
        units = row[1] if row[1] is not None else 0.0
        cost = row[3] if row[3] is not None else row[2] if row[2] is not None else 0.0
        # Determine price
        price: Optional[float]
        if as_of_date:
            price = get_price_on_or_before(conn, ticker_id, as_of_date)
        else:
            price = get_latest_price(conn, ticker_id)
        value = units * price if price is not None else 0.0
        ticker_row = get_ticker(conn, ticker_id)
        positions[ticker_id] = {
            "units": units,
            "cost": cost,
            "value": value,
            "ticker": ticker_row,
        }
    return positions


def compute_account_summary(
    conn: sqlite3.Connection,
    account_id: int,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a summary for an account.

    The summary includes total value, total cost, gain (value minus cost), and
    allocation breakdown by classification (stock vs bond). Any position with
    zero units is ignored. Cash contributions (deposits/withdrawals) are
    implicitly handled by considering transactions whose type is not BUY/SELL
    and do not relate to a ticker; however, this simple model treats
    uninvested cash as zero unless explicit cash transactions are modelled as
    tickers of type 'cash'.

    Args:
        conn: database connection
        account_id: account id
        as_of_date: optional date for valuation

    Returns:
        dictionary with keys: total_value, total_cost, gain, stock_value,
        bond_value, other_value
    """
    positions = compute_account_positions(conn, account_id, as_of_date)
    total_value = 0.0
    total_cost = 0.0
    stock_value = 0.0
    bond_value = 0.0
    other_value = 0.0
    for pos in positions.values():
        units = pos["units"]
        if units == 0:
            continue
        cost = pos["cost"] or 0.0
        value = pos["value"] or 0.0
        total_cost += cost
        total_value += value
        classification = pos["ticker"]["classification"] if pos["ticker"] else None
        if classification == "stock":
            stock_value += value
        elif classification == "bond":
            bond_value += value
        else:
            other_value += value
    gain = total_value - total_cost
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "gain": gain,
        "stock_value": stock_value,
        "bond_value": bond_value,
        "other_value": other_value,
    }


def compute_portfolio_history(
    conn: sqlite3.Connection,
    as_of_dates: List[str],
) -> List[Tuple[str, float]]:
    """Compute total portfolio value for each provided date.

    This function sums positions across all accounts for each date. For each
    ticker and date we look up the price on or before that date (using
    `get_price_on_or_before`). Units held are derived from all transactions up
    to the date. The resulting list can be used to render a time series chart.

    Args:
        conn: database connection
        as_of_dates: list of ISO dates sorted ascending

    Returns:
        list of (date, total_value)
    """
    # Precompute cumulative units per ticker for each date
    cur = conn.cursor()
    # Get list of all unique tickers present in transactions
    cur.execute("SELECT DISTINCT ticker_id FROM transactions")
    tickers = [row[0] for row in cur.fetchall()]
    history: List[Tuple[str, float]] = []
    # For each date compute value
    for date in as_of_dates:
        total_value = 0.0
        for ticker_id in tickers:
            # Sum units up to date
            cur.execute(
                """
                SELECT SUM(units) FROM transactions
                WHERE ticker_id = ? AND date <= ?
                """,
                (ticker_id, date),
            )
            units_row = cur.fetchone()
            units = units_row[0] if units_row and units_row[0] is not None else 0.0
            if units == 0:
                continue
            price = get_price_on_or_before(conn, ticker_id, date)
            if price is None:
                continue
            total_value += units * price
        history.append((date, total_value))
    return history