"""
IDX COST & TAX — Biaya + pajak transaksi saham Indonesia
"""
from __future__ import annotations

PPN_RATE = 0.12
LEVY_RATE = 0.00043
PPH_FINAL_SELL = 0.001


def buy_cost_rate(broker_buy_pct: float) -> float:
    broker = broker_buy_pct / 100.0
    return broker * (1 + PPN_RATE) + LEVY_RATE


def sell_cost_rate(broker_sell_pct: float) -> float:
    broker = broker_sell_pct / 100.0
    return broker * (1 + PPN_RATE) + LEVY_RATE + PPH_FINAL_SELL


def net_profit_loss(entry, exit_price, shares, broker_buy_pct=0.15, broker_sell_pct=0.25) -> dict:
    if shares <= 0 or entry <= 0 or exit_price <= 0:
        return {
            "GrossPL": 0.0, "NetPL": 0.0, "BuyCost": 0.0, "SellCost": 0.0,
            "TotalFeeTax": 0.0, "CostBuyPct": round(buy_cost_rate(broker_buy_pct) * 100, 4),
            "CostSellPct": round(sell_cost_rate(broker_sell_pct) * 100, 4),
        }
    buy_value = float(entry) * int(shares)
    sell_value = float(exit_price) * int(shares)
    c_buy, c_sell = buy_cost_rate(broker_buy_pct), sell_cost_rate(broker_sell_pct)
    buy_cost, sell_cost = buy_value * c_buy, sell_value * c_sell
    gross_pl = sell_value - buy_value
    net_pl = (sell_value - sell_cost) - (buy_value + buy_cost)
    return {
        "GrossPL": round(gross_pl, 0),
        "NetPL": round(net_pl, 0),
        "BuyCost": round(buy_cost, 0),
        "SellCost": round(sell_cost, 0),
        "TotalFeeTax": round(buy_cost + sell_cost, 0),
        "CostBuyPct": round(c_buy * 100, 4),
        "CostSellPct": round(c_sell * 100, 4),
    }


def apply_costs_to_row(row: dict, broker_buy_pct=0.15, broker_sell_pct=0.25) -> dict:
    out = dict(row)
    entry = float(row.get("Entry") or row.get("EntryBreakout") or row.get("Close") or 0)
    stop = float(row.get("StopLoss") or 0)
    shares = int(row.get("SuggestedShares") or 0)
    if shares <= 0:
        shares = int(row.get("Lots") or row.get("SuggestedLots") or 0) * 100

    target = 0.0
    for k in ("Target1", "Target1(Peak)", "Target(Peak)", "Target(Liquidity)", "Target2"):
        if row.get(k) is not None:
            try:
                target = float(row[k])
                break
            except (TypeError, ValueError):
                pass

    if stop > 0 and shares > 0 and entry > 0:
        loss = net_profit_loss(entry, stop, shares, broker_buy_pct, broker_sell_pct)
        out["EstLossGross(Rp)"] = loss["GrossPL"]
        out["EstLossNet(Rp)"] = loss["NetPL"]
    if target > 0 and shares > 0 and entry > 0:
        profit = net_profit_loss(entry, target, shares, broker_buy_pct, broker_sell_pct)
        out["EstProfitGross(Rp)"] = profit["GrossPL"]
        out["EstProfitNet(Rp)"] = profit["NetPL"]
        out["TotalFeeTax(Rp)"] = profit["TotalFeeTax"]
        out["CostBuyPct"] = profit["CostBuyPct"]
        out["CostSellPct"] = profit["CostSellPct"]
    return out


def enrich_dataframe_with_costs(df, broker_buy_pct=0.15, broker_sell_pct=0.25):
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    rows = [apply_costs_to_row(r, broker_buy_pct, broker_sell_pct) for r in df.to_dict("records")]
    return pd.DataFrame(rows)