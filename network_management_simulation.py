#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Economic simulation of network management (based on the supplied assignment).

Run:
    python network_management_simulation.py
    python network_management_simulation.py --seed 7 --output-dir output

The default seed makes the generated simulation reproducible. Costs and timing values
that were specified or exemplified in the assignment are declared in constants below;
traffic, users, event days, and the incident-to-branch mapping are simulated.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Save charts in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Assignment parameters and explicit modelling assumptions
# ---------------------------------------------------------------------------
SEED_DEFAULT = 42
SIMULATION_DAYS = 30
BRANCH_COUNT = 10
REVENUE_LOSS_PER_DOWNTIME_MINUTE_USD = 100.0
BOTTLENECK_WARNING_PCT = 90.0
UPGRADE_COST_PER_BRANCH_USD = 600.0
FORECAST_OBSERVATION_DAYS = 22

# Timing explicitly provided in the assignment (Section 6).
TRADITIONAL_TIMES_MIN = {"detection": 30, "diagnosis": 60, "repair": 90}
SMART_TIMES_MIN = {"detection": 2, "diagnosis": 5, "repair": 30}

# Fixed, itemized monthly cost assumptions. The totals reproduce the assignment's
# illustrative annual OPEX values ($38,000 and $25,000); they are inputs, not outcomes
# hard-coded into the final report. The smart scenario includes a higher monitoring
# software expense, offset by lower staffing, maintenance, and training costs.
OPEX_COMPONENTS = {
    "traditional": {
        "Engineer salary": 1400.00,
        "Internet": 900.00,
        "Electricity": 200.00,
        "Maintenance": 500.00,
        "Training": 100.00,
        "Software license": 66.6666666667,
    },
    "smart": {
        "Engineer salary": 750.00,
        "Internet": 900.00,
        "Electricity": 150.00,
        "Maintenance": 100.00,
        "Training": 60.00,
        "Software / monitoring license": 123.3333333333,
    },
}

# These per-branch equipment costs expand the examples in the assignment to ten branches.
# Totals are Router=$5,800, Switch=$3,400, Firewall=$4,200. Cable + installation=$2,000.
ROUTER_COSTS = [500, 600, 500, 550, 600, 550, 650, 600, 650, 600]
SWITCH_COSTS = [300, 350, 300, 320, 350, 320, 360, 350, 380, 370]
FIREWALL_COSTS = [400, 450, 400, 420, 450, 420, 440, 420, 400, 400]
CABLE_COST_TOTAL_USD = 600.0
INSTALLATION_COST_TOTAL_USD = 1400.0

EVENT_TYPES = [
    "link_failure",
    "router_failure",
    "firewall_failure",
    "configuration_error",
    "high_traffic",
    "bandwidth_overuse",
    "user_growth",
    "bottleneck",
]
OUTAGE_EVENT_TYPES = {
    "link_failure", "router_failure", "firewall_failure", "configuration_error"
}


@dataclass
class SimulationResult:
    seed: int
    branches: pd.DataFrame
    events: pd.DataFrame
    opex: pd.DataFrame
    forecast: pd.DataFrame
    forecast_summary: dict[str, float]
    metrics: pd.DataFrame
    summary: dict[str, Any]


def money(value: float) -> str:
    """Format a dollar value without hiding meaningful cents for small values."""
    return f"${value:,.2f}" if not math.isclose(value, round(value), abs_tol=0.004) else f"${value:,.0f}"


def create_network(rng: np.random.Generator) -> pd.DataFrame:
    """Create a ten-branch network; user counts and traffic are simulated."""
    capacities = np.array([100, 200, 100, 150, 100, 200, 100, 150, 100, 200], dtype=float)
    users = rng.integers(60, 221, size=BRANCH_COUNT)

    # Keep two intentional high-usage cases so the bottleneck and optimisation sections
    # of the assignment are demonstrated; other branch usage is randomized below 90%.
    utilization_pct = rng.uniform(40, 88, size=BRANCH_COUNT)
    utilization_pct[2] = rng.uniform(92, 97)  # Branch-3 is near saturation.
    utilization_pct[4] = rng.uniform(95, 99)  # Branch-5 is the backup scheduling case.

    # The assignment's Branch-5 example uses 60% backup, 25% normal business, 15% other.
    backup_pct = rng.uniform(0.10, 0.40, size=BRANCH_COUNT)
    business_pct = rng.uniform(0.30, 0.60, size=BRANCH_COUNT)
    other_pct = 1.0 - backup_pct - business_pct
    # Ensure non-negative shares and normalize all rows.
    other_pct = np.clip(other_pct, 0.10, 0.45)
    shares = np.vstack([backup_pct, business_pct, other_pct]).T
    shares = shares / shares.sum(axis=1, keepdims=True)
    backup_pct, business_pct, other_pct = shares.T
    backup_pct[4], business_pct[4], other_pct[4] = 0.60, 0.25, 0.15

    names = [f"Branch-{i}" for i in range(1, BRANCH_COUNT + 1)]
    df = pd.DataFrame({
        "branch": names,
        "routers": np.ones(BRANCH_COUNT, dtype=int),
        "switches": np.ones(BRANCH_COUNT, dtype=int),
        "firewalls": np.ones(BRANCH_COUNT, dtype=int),
        "internet_links": np.ones(BRANCH_COUNT, dtype=int),
        "users_initial": users,
        "users": users.copy(),
        "bandwidth_mbps": capacities.astype(int),
        "router_cost_usd": ROUTER_COSTS,
        "switch_cost_usd": SWITCH_COSTS,
        "firewall_cost_usd": FIREWALL_COSTS,
        "backup_share_pct": backup_pct * 100,
        "business_share_pct": business_pct * 100,
        "other_share_pct": other_pct * 100,
        "peak_traffic_before_mbps": capacities * utilization_pct / 100,
        "utilization_before_pct": utilization_pct,
    })

    # The assignment's example recommends moving Branch-5 backup from 10:00 to 02:00.
    # After moving its 60% backup workload out of business hours, peak business-hour load
    # is modelled as the normal-business + other portions of its original peak.
    df["peak_traffic_after_mbps"] = df["peak_traffic_before_mbps"].astype(float)
    branch5_idx = 4
    df.loc[branch5_idx, "peak_traffic_after_mbps"] = (
        df.loc[branch5_idx, "peak_traffic_before_mbps"]
        * (df.loc[branch5_idx, "business_share_pct"] + df.loc[branch5_idx, "other_share_pct"])
        / 100.0
    )
    df["utilization_after_pct"] = (
        df["peak_traffic_after_mbps"] / df["bandwidth_mbps"] * 100.0
    )
    df["bottleneck_before"] = df["utilization_before_pct"] >= BOTTLENECK_WARNING_PCT
    df["bottleneck_after"] = df["utilization_after_pct"] >= BOTTLENECK_WARNING_PCT
    df["backup_schedule_before"] = "10:00"
    df["backup_schedule_after"] = "10:00"
    df.loc[branch5_idx, "backup_schedule_after"] = "02:00"
    return df


def create_events(rng: np.random.Generator, branches: pd.DataFrame) -> pd.DataFrame:
    """Generate eight events across the 30-day month, including four outage incidents."""
    # One of each event class ensures every scenario requirement is demonstrated.
    days = rng.choice(np.arange(1, SIMULATION_DAYS + 1), size=len(EVENT_TYPES), replace=False)
    all_branch_names = branches["branch"].tolist()
    records: list[dict[str, Any]] = []

    for event_type, day in zip(EVENT_TYPES, days):
        users_added = 0
        if event_type == "bandwidth_overuse":
            branch = "Branch-5"
        elif event_type == "bottleneck":
            branch = "Branch-3"
        elif event_type == "user_growth":
            # Place the growth event on Branch-4, a moderate-usage branch, so this effect
            # can be observed without making the example's two intended bottlenecks vanish.
            branch = "Branch-4"
            users_added = int(rng.integers(10, 31))
        else:
            branch = str(rng.choice(all_branch_names))

        is_outage = event_type in OUTAGE_EVENT_TYPES
        traditional_downtime = sum(TRADITIONAL_TIMES_MIN.values()) if is_outage else 0
        smart_downtime = sum(SMART_TIMES_MIN.values()) if is_outage else 0
        records.append({
            "day": int(day),
            "branch": branch,
            "event_type": event_type,
            "users_added": users_added,
            "is_outage_incident": is_outage,
            "traditional_detection_min": TRADITIONAL_TIMES_MIN["detection"] if is_outage else 0,
            "traditional_diagnosis_min": TRADITIONAL_TIMES_MIN["diagnosis"] if is_outage else 0,
            "traditional_repair_min": TRADITIONAL_TIMES_MIN["repair"] if is_outage else 0,
            "traditional_downtime_min": traditional_downtime,
            "smart_detection_min": SMART_TIMES_MIN["detection"] if is_outage else 0,
            "smart_diagnosis_min": SMART_TIMES_MIN["diagnosis"] if is_outage else 0,
            "smart_repair_min": SMART_TIMES_MIN["repair"] if is_outage else 0,
            "smart_downtime_min": smart_downtime,
            "notes": event_note(event_type),
        })
    return pd.DataFrame(records).sort_values(["day", "branch"]).reset_index(drop=True)


def event_note(event_type: str) -> str:
    notes = {
        "link_failure": "Internet link failure; counted as an outage incident.",
        "router_failure": "Router failure; counted as an outage incident.",
        "firewall_failure": "Firewall failure; counted as an outage incident.",
        "configuration_error": "Configuration error; counted as an outage incident.",
        "high_traffic": "Traffic spike; operational event, not a full outage in this model.",
        "bandwidth_overuse": "Bandwidth usage alert; assigned to Branch-5.",
        "user_growth": "Simulated user growth; may increase future load.",
        "bottleneck": "Bottleneck alert; assigned to Branch-3.",
    }
    return notes[event_type]


def apply_user_growth(branches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Apply the generated user-growth event to the branch's month-end load."""
    result = branches.copy()
    for _, event in events.loc[events["event_type"] == "user_growth"].iterrows():
        idx = result.index[result["branch"] == event["branch"]]
        if len(idx) == 0:
            continue
        i = idx[0]
        old_users = float(result.at[i, "users"])
        added = int(event["users_added"])
        if old_users <= 0 or added <= 0:
            continue
        result.at[i, "users"] = int(old_users + added)
        # Assumption: average traffic per user stays constant, so peak traffic scales
        # with the users ratio after the growth event.
        scale = (old_users + added) / old_users
        result.at[i, "peak_traffic_before_mbps"] *= scale
        result.at[i, "peak_traffic_after_mbps"] *= scale
    result["utilization_before_pct"] = result["peak_traffic_before_mbps"] / result["bandwidth_mbps"] * 100.0
    result["utilization_after_pct"] = result["peak_traffic_after_mbps"] / result["bandwidth_mbps"] * 100.0
    result["bottleneck_before"] = result["utilization_before_pct"] >= BOTTLENECK_WARNING_PCT
    result["bottleneck_after"] = result["utilization_after_pct"] >= BOTTLENECK_WARNING_PCT
    return result


def compute_opex() -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    monthly_totals: dict[str, float] = {}
    for scenario, components in OPEX_COMPONENTS.items():
        for component, cost in components.items():
            rows.append({"scenario": scenario, "component": component, "monthly_cost_usd": cost})
        monthly_totals[scenario] = float(sum(components.values()))
    return pd.DataFrame(rows), monthly_totals


def create_forecast(rng: np.random.Generator, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate 22 days of traffic history and estimate when Branch-3 reaches capacity."""
    days_observed = np.arange(1, FORECAST_OBSERVATION_DAYS + 1, dtype=float)
    true_slope = float(rng.uniform(2.1, 2.5))
    baseline = float(rng.uniform(39.0, 41.0))
    noise = rng.normal(0.0, 1.0, size=len(days_observed))
    traffic = baseline + true_slope * (days_observed - 1) + noise
    traffic = np.maximum(traffic, 0.0)

    # Estimate a trend using linear least squares from simulated observations.
    slope, intercept = np.polyfit(days_observed, traffic, 1)
    capacity = 100.0  # Branch-3 capacity follows the 100 Mbps example in the assignment.
    saturation_day = float((capacity - intercept) / slope) if slope > 0 else float("inf")
    days_remaining = max(0.0, saturation_day - FORECAST_OBSERVATION_DAYS)

    forecast_days = np.arange(1, max(FORECAST_OBSERVATION_DAYS + 1, int(math.ceil(min(saturation_day, 60))) + 3))
    fitted_traffic = intercept + slope * forecast_days
    forecast_df = pd.DataFrame({
        "day": forecast_days,
        "traffic_mbps": np.nan,
        "fitted_or_forecast_mbps": fitted_traffic,
        "data_type": "forecast",
    })
    forecast_df.loc[forecast_df["day"] <= FORECAST_OBSERVATION_DAYS, "traffic_mbps"] = traffic
    forecast_df.loc[forecast_df["day"] <= FORECAST_OBSERVATION_DAYS, "data_type"] = "observed"
    forecast_df["capacity_mbps"] = capacity

    summary = {
        "estimated_daily_growth_mbps": float(slope),
        "estimated_intercept_mbps": float(intercept),
        "estimated_saturation_day": saturation_day,
        "days_remaining_from_last_observation": float(days_remaining),
        "last_observed_day": float(FORECAST_OBSERVATION_DAYS),
        "last_observed_traffic_mbps": float(traffic[-1]),
        "capacity_mbps": capacity,
    }

    # Advanced-level extra chart for forecast interpretation.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(days_observed, traffic, marker="o", linestyle="", label="Simulated observed traffic")
    ax.plot(forecast_days, fitted_traffic, label="Linear trend / forecast")
    ax.axhline(capacity, linestyle="--", label="Link capacity (100 Mbps)")
    ax.axvline(saturation_day, linestyle=":", label=f"Estimated saturation: day {saturation_day:.1f}")
    ax.set_title("Branch-3 Traffic Trend and Estimated Saturation")
    ax.set_xlabel("Day")
    ax.set_ylabel("Traffic (Mbps)")
    ax.set_xlim(1, max(30, min(40, int(math.ceil(saturation_day)) + 3)))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "forecast_saturation.png", dpi=160)
    plt.close(fig)
    return forecast_df, summary


def compute_metrics(
    branches: pd.DataFrame,
    events: pd.DataFrame,
    monthly_opex: dict[str, float],
    forecast_summary: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate CAPEX, OPEX, TCO, outages, loss, availability and comparisons."""
    base_equipment = {
        "Router": float(branches["router_cost_usd"].sum()),
        "Switch": float(branches["switch_cost_usd"].sum()),
        "Firewall": float(branches["firewall_cost_usd"].sum()),
        "Cable": CABLE_COST_TOTAL_USD,
        "Installation": INSTALLATION_COST_TOTAL_USD,
    }
    base_capex = sum(base_equipment.values())
    if not math.isclose(base_capex, 15400.0, abs_tol=0.01):
        raise AssertionError(f"Base CAPEX expected to total $15,400, got {base_capex}")

    # Upgrade counts are the assignment's simple decision proxy: every branch above 90%
    # triggers a purchase recommendation if no traffic analysis is available. In the smart
    # case the Branch-5 backup is moved off-peak, so its alert disappears if load falls below
    # the threshold; remaining alerts still trigger upgrade decisions.
    upgrades_traditional = int(branches["bottleneck_before"].sum())
    upgrades_smart = int(branches["bottleneck_after"].sum())
    capex = {
        "traditional": base_capex + upgrades_traditional * UPGRADE_COST_PER_BRANCH_USD,
        "smart": base_capex + upgrades_smart * UPGRADE_COST_PER_BRANCH_USD,
    }

    outage_events = events.loc[events["is_outage_incident"]]
    traditional_downtime = int(outage_events["traditional_downtime_min"].sum())
    smart_downtime = int(outage_events["smart_downtime_min"].sum())
    period_minutes = SIMULATION_DAYS * 24 * 60
    availability = {
        "traditional": max(0.0, (period_minutes - traditional_downtime) / period_minutes * 100),
        "smart": max(0.0, (period_minutes - smart_downtime) / period_minutes * 100),
    }
    lost_revenue = {
        "traditional": traditional_downtime * REVENUE_LOSS_PER_DOWNTIME_MINUTE_USD,
        "smart": smart_downtime * REVENUE_LOSS_PER_DOWNTIME_MINUTE_USD,
    }
    annual_opex = {s: monthly_opex[s] * 12 for s in monthly_opex}
    tco_5y = {s: capex[s] + monthly_opex[s] * 60 for s in monthly_opex}
    mean_repair = {
        "traditional": float(outage_events["traditional_repair_min"].mean()),
        "smart": float(outage_events["smart_repair_min"].mean()),
    }

    def reduction_pct(before: float, after: float) -> float:
        return (before - after) / before * 100.0 if before else 0.0

    summary: dict[str, Any] = {
        "seed": SEED_DEFAULT,
        "simulation_days": SIMULATION_DAYS,
        "branch_count": BRANCH_COUNT,
        "outage_incident_count": int(outage_events.shape[0]),
        "operational_event_count": int(events.shape[0] - outage_events.shape[0]),
        "base_capex_usd": base_capex,
        "base_capex_breakdown_usd": base_equipment,
        "traditional_upgrade_cost_usd": upgrades_traditional * UPGRADE_COST_PER_BRANCH_USD,
        "smart_upgrade_cost_usd": upgrades_smart * UPGRADE_COST_PER_BRANCH_USD,
        "traditional": {
            "capex_usd": capex["traditional"],
            "monthly_opex_usd": monthly_opex["traditional"],
            "annual_opex_usd": annual_opex["traditional"],
            "tco_5_years_usd": tco_5y["traditional"],
            "downtime_min": traditional_downtime,
            "lost_revenue_usd": lost_revenue["traditional"],
            "availability_pct": availability["traditional"],
            "mean_repair_time_min": mean_repair["traditional"],
            "upgrade_decisions_proxy": upgrades_traditional,
        },
        "smart": {
            "capex_usd": capex["smart"],
            "monthly_opex_usd": monthly_opex["smart"],
            "annual_opex_usd": annual_opex["smart"],
            "tco_5_years_usd": tco_5y["smart"],
            "downtime_min": smart_downtime,
            "lost_revenue_usd": lost_revenue["smart"],
            "availability_pct": availability["smart"],
            "mean_repair_time_min": mean_repair["smart"],
            "upgrade_decisions_proxy": upgrades_smart,
        },
        "tco_reduction_pct": reduction_pct(tco_5y["traditional"], tco_5y["smart"]),
        "capex_reduction_pct": reduction_pct(capex["traditional"], capex["smart"]),
        "opex_reduction_pct": reduction_pct(annual_opex["traditional"], annual_opex["smart"]),
        "downtime_reduction_pct": reduction_pct(traditional_downtime, smart_downtime),
        "lost_revenue_saving_usd": lost_revenue["traditional"] - lost_revenue["smart"],
        "unnecessary_upgrade_avoided_proxy": max(0, upgrades_traditional - upgrades_smart),
        "forecast": forecast_summary,
    }
    rows = []
    metric_labels = [
        ("CAPEX", "capex_usd", "USD"),
        ("Monthly OPEX", "monthly_opex_usd", "USD/month"),
        ("Annual OPEX", "annual_opex_usd", "USD/year"),
        ("TCO (5 years)", "tco_5_years_usd", "USD"),
        ("Downtime", "downtime_min", "minutes in 30-day simulation"),
        ("Lost Revenue", "lost_revenue_usd", "USD"),
        ("Network Availability", "availability_pct", "%"),
        ("Mean Repair Time", "mean_repair_time_min", "minutes"),
        ("Upgrade decisions (proxy)", "upgrade_decisions_proxy", "count"),
    ]
    for label, key, unit in metric_labels:
        rows.append({
            "metric": label,
            "unit": unit,
            "without_smart_management": summary["traditional"][key],
            "with_smart_management": summary["smart"][key],
        })
    rows.extend([
        {"metric": "CAPEX Reduction", "unit": "%", "without_smart_management": 0.0, "with_smart_management": summary["capex_reduction_pct"]},
        {"metric": "Annual OPEX Reduction", "unit": "%", "without_smart_management": 0.0, "with_smart_management": summary["opex_reduction_pct"]},
        {"metric": "Downtime Reduction", "unit": "%", "without_smart_management": 0.0, "with_smart_management": summary["downtime_reduction_pct"]},
        {"metric": "TCO Reduction (5 years)", "unit": "%", "without_smart_management": 0.0, "with_smart_management": summary["tco_reduction_pct"]},
        {"metric": "Lost Revenue Saving", "unit": "USD", "without_smart_management": 0.0, "with_smart_management": summary["lost_revenue_saving_usd"]},
    ])
    return pd.DataFrame(rows), summary


def create_charts(
    output_dir: Path,
    branches: pd.DataFrame,
    opex_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Create the four charts requested in the assignment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: cumulative TCO from month 0 through month 60.
    months = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario, label in [("traditional", "Without smart management"), ("smart", "With smart management")]:
        cost = summary[scenario]["capex_usd"] + months * summary[scenario]["monthly_opex_usd"]
        ax.plot(months, cost, label=label)
    ax.set_title("Total Cost of Ownership (TCO) over Five Years")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tco_5_years.png", dpi=160)
    plt.close(fig)

    # Chart 2: monthly OPEX composition.
    pivot = opex_df.pivot(index="component", columns="scenario", values="monthly_cost_usd").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Monthly OPEX by Cost Component")
    ax.set_xlabel("OPEX component")
    ax.set_ylabel("Monthly cost (USD)")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(["With smart management", "Without smart management"] if list(pivot.columns) == ["smart", "traditional"] else None)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_opex.png", dpi=160)
    plt.close(fig)

    # Chart 3: total downtime in the two management modes.
    fig, ax = plt.subplots(figsize=(8, 5.2))
    labels = ["Without smart management", "With smart management"]
    values = [summary["traditional"]["downtime_min"], summary["smart"]["downtime_min"]]
    bars = ax.bar(labels, values)
    ax.set_title("Downtime Comparison in the 30-Day Simulation")
    ax.set_ylabel("Total downtime (minutes)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f")
    fig.tight_layout()
    fig.savefig(output_dir / "downtime_comparison.png", dpi=160)
    plt.close(fig)

    # Chart 4: utilization before and after moving Branch-5 backup to off-peak hours.
    x = np.arange(len(branches))
    width = 0.37
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars_before = ax.bar(x - width / 2, branches["utilization_before_pct"], width, label="Before optimisation")
    bars_after = ax.bar(x + width / 2, branches["utilization_after_pct"], width, label="After optimisation")
    ax.axhline(BOTTLENECK_WARNING_PCT, linestyle="--", label="Warning threshold (90%)")
    ax.set_title("Bandwidth Utilisation by Branch")
    ax.set_xlabel("Branch")
    ax.set_ylabel("Utilisation (%)")
    ax.set_xticks(x, branches["branch"], rotation=30, ha="right")
    ax.set_ylim(0, max(105, float(branches["utilization_before_pct"].max()) + 8))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bandwidth_by_branch.png", dpi=160)
    plt.close(fig)

    # Combined dashboard: all four required views on one page, in addition to standalone PNGs.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    for scenario, label in [("traditional", "Without smart management"), ("smart", "With smart management")]:
        cost = summary[scenario]["capex_usd"] + months * summary[scenario]["monthly_opex_usd"]
        ax.plot(months, cost, label=label)
    ax.set_title("TCO over Five Years")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    component_order = list(dict.fromkeys(opex_df["component"].tolist()))
    component_pivot = opex_df.pivot(index="component", columns="scenario", values="monthly_cost_usd").reindex(component_order)
    xx = np.arange(len(component_pivot.index))
    w = 0.38
    traditional_vals = component_pivot["traditional"].fillna(0).values
    smart_vals = component_pivot["smart"].fillna(0).values
    ax.bar(xx - w / 2, traditional_vals, w, label="Without smart management")
    ax.bar(xx + w / 2, smart_vals, w, label="With smart management")
    ax.set_title("Monthly OPEX by Component")
    ax.set_ylabel("Cost (USD)")
    ax.set_xticks(xx, component_pivot.index, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    dt_labels = ["Without management", "With management"]
    dt_vals = [summary["traditional"]["downtime_min"], summary["smart"]["downtime_min"]]
    bars = ax.bar(dt_labels, dt_vals)
    ax.bar_label(bars, fmt="%.0f")
    ax.set_title("Downtime Comparison")
    ax.set_ylabel("Minutes in 30-day simulation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=10)

    ax = axes[1, 1]
    xx = np.arange(len(branches))
    w = 0.37
    ax.bar(xx - w / 2, branches["utilization_before_pct"], w, label="Before optimisation")
    ax.bar(xx + w / 2, branches["utilization_after_pct"], w, label="After optimisation")
    ax.axhline(BOTTLENECK_WARNING_PCT, linestyle="--", label="90% warning threshold")
    ax.set_title("Bandwidth Utilisation by Branch")
    ax.set_ylabel("Utilisation (%)")
    ax.set_xticks(xx, branches["branch"], rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Network Management Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "dashboard.png", dpi=160)
    plt.close(fig)


def save_outputs(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.branches.to_csv(output_dir / "branches.csv", index=False, float_format="%.4f")
    capex_rows = [{"item": item, "base_cost_usd": amount} for item, amount in result.summary["base_capex_breakdown_usd"].items()]
    capex_rows.extend([
        {"item": "Base CAPEX", "base_cost_usd": result.summary["base_capex_usd"]},
        {"item": "Traditional scenario upgrades", "base_cost_usd": result.summary["traditional_upgrade_cost_usd"]},
        {"item": "Smart scenario upgrades", "base_cost_usd": result.summary["smart_upgrade_cost_usd"]},
        {"item": "Total CAPEX without management", "base_cost_usd": result.summary["traditional"]["capex_usd"]},
        {"item": "Total CAPEX with management", "base_cost_usd": result.summary["smart"]["capex_usd"]},
    ])
    pd.DataFrame(capex_rows).to_csv(output_dir / "capex_breakdown.csv", index=False, float_format="%.2f")
    result.events.to_csv(output_dir / "events.csv", index=False)
    result.opex.to_csv(output_dir / "opex_breakdown.csv", index=False, float_format="%.4f")
    result.forecast.to_csv(output_dir / "traffic_forecast.csv", index=False, float_format="%.4f")
    result.metrics.to_csv(output_dir / "comparison_metrics.csv", index=False, float_format="%.4f")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result.summary, f, ensure_ascii=False, indent=2)

    report = create_text_report(result)
    (output_dir / "network_management_report.txt").write_text(report, encoding="utf-8")

    # Include a simple warning log for the 90% threshold.
    lines = ["NETWORK MONITORING WARNINGS", "=" * 30]
    before = result.branches.loc[result.branches["bottleneck_before"]]
    after = result.branches.loc[result.branches["bottleneck_after"]]
    if before.empty:
        lines.append("No branch exceeded the 90% threshold before optimisation.")
    else:
        lines.append("WARNING before optimisation:")
        for _, r in before.iterrows():
            lines.append(f"{r['branch']} -> {r['utilization_before_pct']:.1f}%")
    if after.empty:
        lines.append("No branch exceeded the 90% threshold after optimisation.")
    else:
        lines.append("WARNING after optimisation:")
        for _, r in after.iterrows():
            lines.append(f"{r['branch']} -> {r['utilization_after_pct']:.1f}%")
    (output_dir / "warnings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_text_report(result: SimulationResult) -> str:
    s = result.summary
    t = s["traditional"]
    m = s["smart"]
    f = s["forecast"]
    lines = [
        "NETWORK MANAGEMENT REPORT",
        "=" * 60,
        f"Random seed                : {result.seed}",
        f"Simulation period          : {s['simulation_days']} days",
        f"Number of branches         : {s['branch_count']}",
        f"Outage incidents           : {s['outage_incident_count']}",
        f"Other operational events   : {s['operational_event_count']}",
        "",
        "CAPEX BREAKDOWN (BASE NETWORK)",
        "-" * 60,
        f"Total Router Cost          : {money(s['base_capex_breakdown_usd']['Router'])}",
        f"Total Switch Cost          : {money(s['base_capex_breakdown_usd']['Switch'])}",
        f"Total Firewall Cost        : {money(s['base_capex_breakdown_usd']['Firewall'])}",
        f"Cable                      : {money(s['base_capex_breakdown_usd']['Cable'])}",
        f"Installation               : {money(s['base_capex_breakdown_usd']['Installation'])}",
        f"Base CAPEX                 : {money(s['base_capex_usd'])}",
        f"Traditional upgrades       : {money(s['traditional_upgrade_cost_usd'])}",
        f"Smart-scenario upgrades    : {money(s['smart_upgrade_cost_usd'])}",
        "",
        "COMPARISON",
        "-" * 60,
        f"{'Metric':30s} {'Without management':>18s} {'With management':>18s}",
        f"{'CAPEX':30s} {money(t['capex_usd']):>18s} {money(m['capex_usd']):>18s}",
        f"{'Monthly OPEX':30s} {money(t['monthly_opex_usd']):>18s} {money(m['monthly_opex_usd']):>18s}",
        f"{'Annual OPEX':30s} {money(t['annual_opex_usd']):>18s} {money(m['annual_opex_usd']):>18s}",
        f"{'TCO (5 years)':30s} {money(t['tco_5_years_usd']):>18s} {money(m['tco_5_years_usd']):>18s}",
        f"{'Downtime (minutes)':30s} {t['downtime_min']:>18,.0f} {m['downtime_min']:>18,.0f}",
        f"{'Lost revenue':30s} {money(t['lost_revenue_usd']):>18s} {money(m['lost_revenue_usd']):>18s}",
        f"{'Network availability':30s} {t['availability_pct']:>17.3f}% {m['availability_pct']:>17.3f}%",
        f"{'Mean repair time':30s} {t['mean_repair_time_min']:>18.1f} {m['mean_repair_time_min']:>18.1f}",
        f"{'Upgrade decisions (proxy)':30s} {t['upgrade_decisions_proxy']:>18,.0f} {m['upgrade_decisions_proxy']:>18,.0f}",
        "",
        "IMPACT OF SMART MANAGEMENT",
        "-" * 60,
        f"TCO reduction              : {s['tco_reduction_pct']:.2f}%",
        f"CAPEX reduction            : {s['capex_reduction_pct']:.2f}%",
        f"Annual OPEX reduction      : {s['opex_reduction_pct']:.2f}%",
        f"Downtime reduction         : {s['downtime_reduction_pct']:.2f}%",
        f"Lost revenue saving        : {money(s['lost_revenue_saving_usd'])}",
        f"Upgrade decisions avoided  : {s['unnecessary_upgrade_avoided_proxy']}",
        "",
        "BOTTLENECK / RESOURCE OPTIMISATION",
        "-" * 60,
    ]
    for _, row in result.branches.iterrows():
        if row["bottleneck_before"] or row["bottleneck_after"]:
            lines.append(
                f"{row['branch']}: {row['utilization_before_pct']:.1f}% before -> "
                f"{row['utilization_after_pct']:.1f}% after; backup {row['backup_schedule_before']} "
                f"-> {row['backup_schedule_after']}"
            )
    lines.extend([
        "",
        "TRAFFIC FORECAST (Branch-3, 100 Mbps capacity)",
        "-" * 60,
        f"Estimated growth           : {f['estimated_daily_growth_mbps']:.2f} Mbps/day",
        f"Last observed traffic      : {f['last_observed_traffic_mbps']:.2f} Mbps on day {int(f['last_observed_day'])}",
        f"Estimated saturation day   : {f['estimated_saturation_day']:.2f}",
        f"Estimated days remaining   : {f['days_remaining_from_last_observation']:.2f}",
        "",
        "RECOMMENDATION",
        "-" * 60,
        "Smart monitoring, alarm correlation and traffic analysis reduce the modelled OPEX and downtime.",
        "Move backup traffic from 10:00 to 02:00 for Branch-5; review any branch still above 90% utilisation.",
        "Use the traffic forecast to plan capacity before the link saturates.",
        "",
        "MODELLING NOTES",
        "- Downtime totals sum the outage incidents in the 30-day simulation; operational alerts are not treated as full outages.",
        "- Availability follows the assignment example: (30-day minutes - summed outage minutes) / 30-day minutes.",
        "- Lost revenue is modelled at $100 per downtime minute, as specified by the assignment.",
        "- Upgrade decisions are a simple proxy: a branch at or above 90% utilisation triggers an upgrade decision.",
        "- The seed makes random user counts, traffic, event days, and generated forecast data reproducible.",
        "- Cost values not fully specified by the assignment are assumptions documented in the README.",
    ])
    return "\n".join(lines) + "\n"


def run_simulation(seed: int = SEED_DEFAULT, output_dir: str | Path = "output") -> SimulationResult:
    """Run all project sections, write deliverables and return computed results."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    branches = create_network(rng)
    events = create_events(rng, branches)
    branches = apply_user_growth(branches, events)
    opex_df, monthly_opex = compute_opex()
    forecast_df, forecast_summary = create_forecast(rng, output_path)
    metrics_df, summary = compute_metrics(branches, events, monthly_opex, forecast_summary)
    summary["seed"] = seed

    result = SimulationResult(
        seed=seed,
        branches=branches,
        events=events,
        opex=opex_df,
        forecast=forecast_df,
        forecast_summary=forecast_summary,
        metrics=metrics_df,
        summary=summary,
    )
    create_charts(output_path, branches, opex_df, summary)
    save_outputs(result, output_path)
    return result


def print_console_summary(result: SimulationResult, output_dir: str | Path) -> None:
    s = result.summary
    print("\n" + "=" * 64)
    print("NETWORK MANAGEMENT SIMULATION COMPLETED")
    print("=" * 64)
    print(f"Branches: {s['branch_count']} | outage incidents: {s['outage_incident_count']} | seed: {result.seed}")
    print(f"CAPEX:           {money(s['traditional']['capex_usd'])} -> {money(s['smart']['capex_usd'])}")
    print(f"Annual OPEX:     {money(s['traditional']['annual_opex_usd'])} -> {money(s['smart']['annual_opex_usd'])}")
    print(f"5-year TCO:      {money(s['traditional']['tco_5_years_usd'])} -> {money(s['smart']['tco_5_years_usd'])}")
    print(f"Downtime:        {s['traditional']['downtime_min']} -> {s['smart']['downtime_min']} minutes")
    print(f"Lost revenue:    {money(s['traditional']['lost_revenue_usd'])} -> {money(s['smart']['lost_revenue_usd'])}")
    print(f"Availability:    {s['traditional']['availability_pct']:.3f}% -> {s['smart']['availability_pct']:.3f}%")
    print(f"TCO reduction:   {s['tco_reduction_pct']:.2f}%")
    print(f"OPEX reduction:  {s['opex_reduction_pct']:.2f}%")
    print(f"Downtime saved:  {s['downtime_reduction_pct']:.2f}%")
    print(f"Estimated Branch-3 saturation: day {s['forecast']['estimated_saturation_day']:.2f}")
    print(f"\nFiles written to: {Path(output_dir).expanduser().resolve()}")
    print("See network_management_report.txt for the full report.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Network management economic simulation")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="output", help="Directory for CSV, JSON, TXT, and charts")
    args = parser.parse_args()
    result = run_simulation(seed=args.seed, output_dir=args.output_dir)
    print_console_summary(result, args.output_dir)


if __name__ == "__main__":
    main()
