"""
Risk Manager for Kalshi Trading Bot

Enforces position sizing, daily loss limits, max concurrent positions,
and per trade risk controls. No trade gets executed without passing
through the risk manager first.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


def kalshi_taker_fee(contracts: int, price_cents: int) -> float:
    """
    Kalshi taker fee per trade: 0.07 * contracts * P * (1 - P)
    where P = price in dollars, price_cents / 100.
    Total rounded up to the nearest cent.
    """
    p = price_cents / 100.0
    fee = 0.07 * contracts * p * (1 - p)
    return math.ceil(fee * 100) / 100


def kalshi_maker_fee(contracts: int, price_cents: int) -> float:
    """
    Kalshi maker fee: $0.00 for resting limit orders.
    """
    return 0.0


@dataclass
class RiskConfig:
    """High volume demo configuration with dynamic risk sizing."""

    stake_usd: float = 1.00
    kelly_fraction: float = 0.00

    max_daily_loss_usd: float = 100.00
    max_weekly_loss_usd: float = 500.00

    max_concurrent_positions: int = 25
    max_position_pct: float = 0.01

    min_confidence: float = 0.00
    cooldown_after_loss_secs: int = 0
    max_trades_per_hour: int = 250

    risk_sizing_enabled: bool = True
    min_edge: float = -1.00
    max_risk_score: float = 1.00
    min_sizing_multiplier: float = 0.25
    max_sizing_multiplier: float = 1.50
    min_contracts: int = 1
    max_contracts: int = 100


@dataclass
class TradeRecord:
    """Record of a placed trade."""
    timestamp: float
    ticker: str
    strategy: str
    side: str
    price_cents: int
    contracts: int
    stake_usd: float
    order_id: str = ""
    client_order_id: str = ""
    outcome: str = ""
    payout_usd: float = 0.0
    profit_usd: float = 0.0
    entry_fee_usd: float = 0.0
    settle_fee_usd: float = 0.0
    profit_after_fees: float = 0.0
    is_maker: bool = False

    risk_score: float = 0.0
    risk_level: str = ""
    sizing_multiplier: float = 1.0
    market_probability: float = 0.0
    estimated_probability: float = 0.0
    edge: float = 0.0
    expected_value_per_contract: float = 0.0
    risk_notes: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Risk score used to size a proposed trade."""
    risk_score: float
    risk_level: str
    sizing_multiplier: float
    market_probability: float
    estimated_probability: float
    edge: float
    expected_value_per_contract: float
    estimated_fee_per_contract: float
    notes: list[str] = field(default_factory=list)


class RiskManager:
    """
    Enforces risk limits and position sizing.

    All trade requests must pass through approve_trade before execution.
    The risk manager tracks PnL, open positions, and daily or weekly limits.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.trades: list[TradeRecord] = []
        self.open_positions: dict[str, TradeRecord] = {}
        self._last_loss_ts: float = 0
        self._daily_reset_ts: float = time.time()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    @property
    def daily_pnl(self) -> float:
        """Sum of realized PnL since last daily reset."""
        return sum(
            t.profit_usd for t in self.trades
            if t.timestamp >= self._daily_reset_ts and t.outcome != ""
        )

    @property
    def weekly_pnl(self) -> float:
        """Sum of realized PnL over the last 7 days."""
        cutoff = time.time() - 7 * 86400
        return sum(
            t.profit_usd for t in self.trades
            if t.timestamp >= cutoff and t.outcome != ""
        )

    @property
    def trades_this_hour(self) -> int:
        """Number of trades placed in the last hour."""
        cutoff = time.time() - 3600
        return sum(1 for t in self.trades if t.timestamp >= cutoff)

    @property
    def win_rate(self) -> Optional[float]:
        """Win rate of all completed trades."""
        completed = [t for t in self.trades if t.outcome != ""]
        if not completed:
            return None
        wins = sum(1 for t in completed if t.outcome == "win")
        return wins / len(completed)

    @property
    def total_pnl(self) -> float:
        """Total realized PnL."""
        return sum(t.profit_usd for t in self.trades if t.outcome != "")

    @property
    def total_pnl_after_fees(self) -> float:
        """Total realized PnL after entry fees."""
        return sum(t.profit_after_fees for t in self.trades if t.outcome != "")

    def reset_daily(self):
        """Reset the daily PnL counter."""
        self._daily_reset_ts = time.time()

    def approve_trade(
        self,
        ticker: str,
        strategy_name: str,
        side: str,
        confidence: float,
        price_cents: int,
        balance_usd: float = None,
        calibrated_probability: float = None,
        is_maker: bool = False,
    ) -> tuple[bool, str]:
        """
        Check if a proposed trade passes all risk filters.

        Returns:
            approved, reason
        """
        if confidence < self.config.min_confidence:
            return False, f"Confidence {confidence:.2f} below threshold {self.config.min_confidence}"

        risk = self.assess_trade_risk(
            ticker=ticker,
            confidence=confidence,
            price_cents=price_cents,
            balance_usd=balance_usd,
            calibrated_probability=calibrated_probability,
            is_maker=is_maker,
        )

        if risk.edge < self.config.min_edge:
            return False, f"Edge {risk.edge:.2%} below threshold {self.config.min_edge:.2%}"

        if risk.risk_score > self.config.max_risk_score:
            return False, f"Trade risk too high ({risk.risk_score:.2f}, {risk.risk_level})"

        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"

        if self.weekly_pnl <= -self.config.max_weekly_loss_usd:
            return False, f"Weekly loss limit reached (${self.weekly_pnl:.2f})"

        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return False, f"Max positions reached ({len(self.open_positions)})"

        if ticker in self.open_positions:
            return False, f"Already have position on {ticker}"

        if self.trades_this_hour >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.trades_this_hour})"

        if time.time() - self._last_loss_ts < self.config.cooldown_after_loss_secs:
            remaining = self.config.cooldown_after_loss_secs - (time.time() - self._last_loss_ts)
            return False, f"Loss cooldown ({remaining:.0f}s remaining)"

        if balance_usd is not None:
            max_stake = balance_usd * self.config.max_position_pct
            actual_contracts = self.calculate_contracts(
                price_cents=price_cents,
                confidence=confidence,
                balance_usd=balance_usd,
                calibrated_probability=calibrated_probability,
                is_maker=is_maker,
                ticker=ticker,
            )
            actual_stake = actual_contracts * (price_cents / 100.0)

            if actual_stake > max_stake:
                return False, f"Stake ${actual_stake:.2f} exceeds {self.config.max_position_pct * 100:.0f}% of balance (${max_stake:.2f})"

        return True, (
            f"Approved | risk={risk.risk_score:.2f} {risk.risk_level} | "
            f"size_mult={risk.sizing_multiplier:.2f}x | edge={risk.edge:.2%}"
        )

    def correlated_position_count(self) -> int:
        """Count open positions that share the same market window."""
        windows = set()

        for ticker in self.open_positions:
            parts = ticker.split("-", 1)
            if len(parts) > 1:
                windows.add(parts[1])

        if not windows:
            return 0

        return len(self.open_positions)

    def assess_trade_risk(
        self,
        ticker: str,
        confidence: float,
        price_cents: int,
        balance_usd: float = None,
        calibrated_probability: float = None,
        is_maker: bool = False,
    ) -> RiskAssessment:
        """
        Score a proposed trade from 0.00 to 1.00.

        0.00 means very low risk.
        1.00 means very high risk.
        """
        if price_cents <= 0 or price_cents >= 100:
            return RiskAssessment(
                risk_score=1.0,
                risk_level="invalid",
                sizing_multiplier=0.0,
                market_probability=0.0,
                estimated_probability=0.0,
                edge=-1.0,
                expected_value_per_contract=-1.0,
                estimated_fee_per_contract=0.0,
                notes=["Invalid price"],
            )

        price_usd = price_cents / 100.0
        market_probability = price_usd

        estimated_probability = (
            calibrated_probability if calibrated_probability is not None else confidence
        )
        estimated_probability = self._clamp(estimated_probability, 0.0, 1.0)

        edge = estimated_probability - market_probability

        estimated_fee_per_contract = 0.0
        if not is_maker:
            estimated_fee_per_contract = 0.07 * price_usd * (1 - price_usd)

        expected_value_per_contract = edge - estimated_fee_per_contract

        notes: list[str] = []

        edge_risk = self._clamp((0.08 - edge) / 0.16, 0.0, 1.0)

        if edge < 0:
            notes.append("negative edge")
        elif edge < 0.03:
            notes.append("thin edge")
        elif edge >= 0.08:
            notes.append("strong edge")

        confidence_risk = 1.0 - estimated_probability

        if estimated_probability < 0.50:
            notes.append("low win probability")

        price_risk = abs(price_usd - 0.50) * 2.0

        if price_usd <= 0.15:
            notes.append("cheap longshot contract")
        elif price_usd >= 0.85:
            notes.append("expensive contract")

        daily_drawdown = 0.0
        if self.daily_pnl < 0 and self.config.max_daily_loss_usd > 0:
            daily_drawdown = self._clamp(
                abs(self.daily_pnl) / self.config.max_daily_loss_usd,
                0.0,
                1.0,
            )

        weekly_drawdown = 0.0
        if self.weekly_pnl < 0 and self.config.max_weekly_loss_usd > 0:
            weekly_drawdown = self._clamp(
                abs(self.weekly_pnl) / self.config.max_weekly_loss_usd,
                0.0,
                1.0,
            )

        drawdown_risk = max(daily_drawdown, weekly_drawdown)

        if drawdown_risk >= 0.50:
            notes.append("large drawdown")

        position_load = 0.0
        if self.config.max_concurrent_positions > 0:
            position_load = self._clamp(
                len(self.open_positions) / self.config.max_concurrent_positions,
                0.0,
                1.0,
            )

        correlation_load = self._clamp(self.correlated_position_count() / 5.0, 0.0, 1.0)

        if correlation_load >= 0.40:
            notes.append("correlated exposure")

        velocity_load = 0.0
        if self.config.max_trades_per_hour > 0:
            velocity_load = self._clamp(
                self.trades_this_hour / self.config.max_trades_per_hour,
                0.0,
                1.0,
            )

        if expected_value_per_contract < 0:
            fee_risk = 1.0
            notes.append("fee adjusted EV negative")
        elif estimated_fee_per_contract > 0 and edge < estimated_fee_per_contract * 2:
            fee_risk = 0.75
            notes.append("fee drag high")
        else:
            fee_risk = 0.0

        risk_score = (
            0.28 * edge_risk
            + 0.18 * confidence_risk
            + 0.14 * price_risk
            + 0.14 * drawdown_risk
            + 0.10 * position_load
            + 0.08 * correlation_load
            + 0.04 * velocity_load
            + 0.04 * fee_risk
        )

        risk_score = self._clamp(risk_score, 0.0, 1.0)

        if risk_score < 0.25:
            risk_level = "low"
        elif risk_score < 0.50:
            risk_level = "medium"
        elif risk_score < 0.75:
            risk_level = "high"
        else:
            risk_level = "extreme"

        sizing_multiplier = self.config.max_sizing_multiplier - (
            risk_score * (self.config.max_sizing_multiplier - self.config.min_sizing_multiplier)
        )

        if edge < 0:
            sizing_multiplier *= 0.50
        elif edge < 0.03:
            sizing_multiplier *= 0.75
        elif edge >= 0.10 and estimated_probability >= 0.60:
            sizing_multiplier *= 1.10

        sizing_multiplier = self._clamp(
            sizing_multiplier,
            self.config.min_sizing_multiplier,
            self.config.max_sizing_multiplier,
        )

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            sizing_multiplier=sizing_multiplier,
            market_probability=market_probability,
            estimated_probability=estimated_probability,
            edge=edge,
            expected_value_per_contract=expected_value_per_contract,
            estimated_fee_per_contract=estimated_fee_per_contract,
            notes=notes,
        )

    def calculate_contracts(
        self,
        price_cents: int,
        confidence: float = 0.0,
        balance_usd: float = None,
        calibrated_probability: float = None,
        is_maker: bool = False,
        ticker: str = "",
    ) -> int:
        """
        Risk adjusted position size.
        """
        if price_cents <= 0 or price_cents >= 100:
            return 0

        price_usd = price_cents / 100.0

        if balance_usd is not None and balance_usd > 0:
            balance_cap = balance_usd * self.config.max_position_pct
            stake = min(self.config.stake_usd, balance_cap)
        else:
            stake = self.config.stake_usd

        if (
            self.config.kelly_fraction > 0
            and confidence > 0
            and balance_usd is not None
            and balance_usd > 0
        ):
            p = calibrated_probability if calibrated_probability is not None else confidence
            p = self._clamp(p, 0.0, 1.0)
            q = 1 - p
            b = (100 - price_cents) / price_cents
            kelly_f = (p * b - q) / b if b > 0 else 0
            kelly_f = max(0, kelly_f)
            kelly_f *= self.config.kelly_fraction

            kelly_stake = kelly_f * balance_usd
            stake = min(stake, kelly_stake)

        open_count = self.correlated_position_count()

        if open_count >= 2:
            stake *= 0.6
        elif open_count == 1:
            stake *= 0.8

        if self.config.risk_sizing_enabled:
            risk = self.assess_trade_risk(
                ticker=ticker,
                confidence=confidence,
                price_cents=price_cents,
                balance_usd=balance_usd,
                calibrated_probability=calibrated_probability,
                is_maker=is_maker,
            )
            stake *= risk.sizing_multiplier

        if balance_usd is not None and balance_usd > 0:
            stake = min(stake, balance_usd * self.config.max_position_pct)

        stake = min(stake, self.config.stake_usd)

        contracts = int(stake / price_usd)

        return max(
            self.config.min_contracts,
            min(contracts, self.config.max_contracts),
        )

    def sizing_preview(
        self,
        ticker: str,
        confidence: float,
        price_cents: int,
        balance_usd: float = None,
        calibrated_probability: float = None,
        is_maker: bool = False,
    ) -> dict:
        """Preview the risk score and final size before placing an order."""
        risk = self.assess_trade_risk(
            ticker=ticker,
            confidence=confidence,
            price_cents=price_cents,
            balance_usd=balance_usd,
            calibrated_probability=calibrated_probability,
            is_maker=is_maker,
        )

        contracts = self.calculate_contracts(
            price_cents=price_cents,
            confidence=confidence,
            balance_usd=balance_usd,
            calibrated_probability=calibrated_probability,
            is_maker=is_maker,
            ticker=ticker,
        )

        stake_usd = contracts * (price_cents / 100.0)

        return {
            "approved_size_contracts": contracts,
            "stake_usd": round(stake_usd, 2),
            "risk_score": round(risk.risk_score, 3),
            "risk_level": risk.risk_level,
            "sizing_multiplier": round(risk.sizing_multiplier, 3),
            "market_probability": round(risk.market_probability, 3),
            "estimated_probability": round(risk.estimated_probability, 3),
            "edge": round(risk.edge, 3),
            "expected_value_per_contract": round(risk.expected_value_per_contract, 3),
            "notes": risk.notes,
        }

    def record_trade(self, record: TradeRecord):
        """Record a new trade and add to open positions."""
        fee_fn = kalshi_maker_fee if record.is_maker else kalshi_taker_fee
        record.entry_fee_usd = fee_fn(record.contracts, record.price_cents)

        risk = self.assess_trade_risk(
            ticker=record.ticker,
            confidence=record.estimated_probability or record.market_probability or 0.0,
            price_cents=record.price_cents,
            is_maker=record.is_maker,
        )

        record.risk_score = risk.risk_score
        record.risk_level = risk.risk_level
        record.sizing_multiplier = risk.sizing_multiplier
        record.market_probability = risk.market_probability
        record.estimated_probability = risk.estimated_probability
        record.edge = risk.edge
        record.expected_value_per_contract = risk.expected_value_per_contract
        record.risk_notes = risk.notes

        self.trades.append(record)
        self.open_positions[record.ticker] = record

    def settle_trade(self, ticker: str, result: str):
        """
        Settle an open position.

        Args:
            ticker: Market ticker
            result: The market result, yes or no
        """
        if ticker not in self.open_positions:
            return

        record = self.open_positions.pop(ticker)

        if record.side == result:
            record.outcome = "win"
            record.payout_usd = record.contracts * 1.00
            record.profit_usd = record.payout_usd - record.stake_usd
            record.settle_fee_usd = 0.0
        else:
            record.outcome = "loss"
            record.payout_usd = 0.0
            record.profit_usd = -record.stake_usd
            record.settle_fee_usd = 0.0
            self._last_loss_ts = time.time()

        record.profit_after_fees = record.profit_usd - record.entry_fee_usd

    def stats_summary(self) -> str:
        """Human readable summary of trading stats."""
        completed = [t for t in self.trades if t.outcome != ""]

        if not completed:
            return "No completed trades yet"

        wins = sum(1 for t in completed if t.outcome == "win")
        losses = len(completed) - wins
        wr = self.win_rate or 0

        risk_records = [t for t in completed if t.risk_level]
        avg_risk = (
            sum(t.risk_score for t in risk_records) / len(risk_records)
            if risk_records
            else 0.0
        )

        lines = [
            f"Trades: {len(completed)} ({wins}W/{losses}L, {wr * 100:.1f}% WR)",
            f"Total PnL: ${self.total_pnl:+.2f}",
            f"After fee PnL: ${self.total_pnl_after_fees:+.2f}",
            f"Daily PnL: ${self.daily_pnl:+.2f}",
            f"Open positions: {len(self.open_positions)}",
            f"Avg risk: {avg_risk:.2f}",
        ]

        return " | ".join(lines)
