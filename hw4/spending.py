"""Persistent daily totals for the MCP model stub's simulated charges."""

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / "tmp/mcp_daily_spend.sqlite3"
DEFAULT_BUDGET_USD = 0.01
MICROS_PER_USD = 1_000_000
STUB_CALL_MICROS = 1_000
MOSCOW = "Europe/Moscow"
MAX_BUDGET_USD = Decimal(2**63 - 1).scaleb(-6)


def limit_micros(value):
    """Convert a USD limit to integer microdollars without rounding."""
    message = "Лимит должен быть конечной неотрицательной суммой USD с точностью до 6 знаков."
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or not 0 <= amount <= MAX_BUDGET_USD:
            raise ValueError(message)
        with localcontext() as context:
            context.prec = max(context.prec, len(amount.as_tuple().digits))
            micros = amount * MICROS_PER_USD
            if micros.is_zero() != amount.is_zero() or micros != micros.to_integral_value():
                raise ValueError(message)
        return int(micros)
    except DecimalException:
        raise ValueError(message) from None


def as_usd(micros):
    return micros / MICROS_PER_USD


class DailySpend:
    """Store only aggregate simulated costs, keyed by Moscow calendar date."""

    def __init__(self, path=DEFAULT_LEDGER, *, clock=None):
        self.path = Path(path)
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)

    def _date(self):
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("Для дневного счётчика нужно время с часовым поясом.")
        return now.astimezone(ZoneInfo(MOSCOW)).date().isoformat()

    @contextmanager
    def _database(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS daily_stub_spend (
                    day TEXT PRIMARY KEY,
                    total_micros INTEGER NOT NULL,
                    calls INTEGER NOT NULL
                )
            """)
            yield connection

    def record_stub_call(self, amount_micros):
        if type(amount_micros) is not int or amount_micros <= 0:
            raise ValueError("Условное списание должно быть целым положительным числом микродолларов.")
        day = self._date()
        with self._database() as connection:
            connection.execute("""
                INSERT INTO daily_stub_spend(day, total_micros, calls) VALUES (?, ?, 1)
                ON CONFLICT(day) DO UPDATE SET
                    total_micros = total_micros + excluded.total_micros,
                    calls = calls + 1
            """, (day, amount_micros))

    def snapshot(self):
        day = self._date()
        with self._database() as connection:
            row = connection.execute(
                "SELECT total_micros, calls FROM daily_stub_spend WHERE day = ?", (day,),
            ).fetchone()
        micros, calls = row or (0, 0)
        return {"date": day, "timezone": MOSCOW, "simulated_usd": as_usd(micros), "simulated_calls": calls}
