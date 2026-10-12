import tempfile
import unittest
from pathlib import Path

import pandas as pd

from network_management_simulation import (
    run_simulation,
    ROUTER_COSTS,
    SWITCH_COSTS,
    FIREWALL_COSTS,
)


class NetworkManagementSimulationTests(unittest.TestCase):
    def test_network_has_ten_branches_and_expected_base_capex(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_simulation(seed=42, output_dir=tmp)
            self.assertEqual(len(result.branches), 10)
            self.assertEqual(int(result.branches["router_cost_usd"].sum()), sum(ROUTER_COSTS))
            self.assertEqual(int(result.branches["switch_cost_usd"].sum()), sum(SWITCH_COSTS))
            self.assertEqual(int(result.branches["firewall_cost_usd"].sum()), sum(FIREWALL_COSTS))
            self.assertAlmostEqual(result.summary["base_capex_usd"], 15400.0)

    def test_outage_times_and_lost_revenue(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_simulation(seed=42, output_dir=tmp)
            self.assertEqual(int(result.events["is_outage_incident"].sum()), 4)
            self.assertEqual(result.summary["traditional"]["downtime_min"], 720)
            self.assertEqual(result.summary["smart"]["downtime_min"], 148)
            self.assertEqual(result.summary["traditional"]["lost_revenue_usd"], 72000)
            self.assertEqual(result.summary["smart"]["lost_revenue_usd"], 14800)

    def test_backup_shift_reduces_branch_5_peak_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_simulation(seed=42, output_dir=tmp)
            branch5 = result.branches.loc[result.branches["branch"] == "Branch-5"].iloc[0]
            self.assertEqual(branch5["backup_schedule_after"], "02:00")
            self.assertGreaterEqual(branch5["utilization_before_pct"], 90)
            self.assertLess(branch5["utilization_after_pct"], 90)

    def test_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_simulation(seed=42, output_dir=tmp)
            expected = [
                "branches.csv", "events.csv", "capex_breakdown.csv", "opex_breakdown.csv",
                "traffic_forecast.csv", "comparison_metrics.csv", "summary.json",
                "network_management_report.txt", "warnings.txt", "tco_5_years.png",
                "monthly_opex.png", "downtime_comparison.png", "bandwidth_by_branch.png",
                "forecast_saturation.png", "dashboard.png",
            ]
            for name in expected:
                self.assertTrue((Path(tmp) / name).exists(), name)

    def test_same_seed_reproduces_simulated_events(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            r1 = run_simulation(seed=123, output_dir=t1)
            r2 = run_simulation(seed=123, output_dir=t2)
            pd.testing.assert_frame_equal(r1.events, r2.events)
            pd.testing.assert_frame_equal(r1.branches, r2.branches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
