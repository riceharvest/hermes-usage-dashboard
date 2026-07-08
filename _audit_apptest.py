import os
import subprocess

os.chdir("/home/dario/Documents/dev workspace/hermes-usage-dashboard")

script = '''
from streamlit.testing.v1 import AppTest
import pandas as pd
import json
from datetime import date, timedelta

at = AppTest.from_file("app.py")
at.run()

issues = []

def check(cond, msg):
    if not cond:
        issues.append(msg)

# 1. App renders without exception
check(bool(at.exception) is False, f"AppTest exception: {at.exception}")

# 2. Title/caption present
main = at.main
titles = [el for el in main if type(el).__name__ in ("Title", "Caption")]
check(len(titles) >= 2, "Title/caption missing")

# 3. Expected top-level tabs exist
labels = [getattr(el, "label", None) for el in main if type(el).__name__ == "Tab"]
for expected in ["By Provider", "By Model", "By Period", "Top-N Trend", "Session Logs", "Model Catalog", "Insights"]:
    check(expected in labels, f"Missing tab: {expected}")

# 4. Expected KPI metrics exist
metrics = [getattr(el, "label", None) for el in main if type(el).__name__ == "Metric"]
for expected in ["Non-cached input", "Cached input", "Output", "Cache Hit Rate", "Est. cost (OR)", "Realized Savings"]:
    check(expected in metrics, f"Missing KPI metric: {expected}")

# 5. No duplicated tab labels (redundancy)
from collections import Counter
for label, count in Counter(labels).items():
    check(count == 1, f"Duplicate tab label: {label} ({count}x)")

# 6. Collect dataframes per tab by walking top-level Tabs
prov_df = None
model_df = None
period_df = None
for el in main:
    if type(el).__name__ == "Tab" and getattr(el, "label", None) == "By Provider":
        for child in el:
            if type(child).__name__ == "Dataframe":
                prov_df = child.value
    if type(el).__name__ == "Tab" and getattr(el, "label", None) == "By Model":
        for child in el:
            if type(child).__name__ == "Dataframe":
                model_df = child.value
    if type(el).__name__ == "Tab" and getattr(el, "label", None) == "By Period":
        for child in el:
            if type(child).__name__ == "Dataframe":
                period_df = child.value

if prov_df is not None:
    check("% cost" in prov_df.columns, "By Provider missing % cost column")
    check(prov_df["% cost"].dtype.kind in "fi", "By Provider % cost not numeric")
    check("Cost (OR)" in prov_df.columns, "By Provider missing Cost (OR)")
if model_df is not None:
    check("% cost" in model_df.columns, "By Model missing % cost column")

# 7. Period default check via safe session_state access
ss = at.session_state
default_period = "daily"
if hasattr(ss, "get"):
    default_period = ss.get("filter_time_bucket", "daily")
else:
    try:
        default_period = ss["filter_time_bucket"]
    except KeyError:
        default_period = "daily"
check(default_period in ["daily", "weekly", "monthly", "all"], f"Invalid default period {default_period}")
if period_df is not None and default_period in ["daily", "weekly", "monthly"]:
    check(default_period.capitalize() in period_df.columns, f"By Period missing {default_period.capitalize()} column")

# 8. Currency symbols consistent with session state
try:
    sym = ss["currency_symbol"]
except KeyError:
    sym = "$"
try:
    rate = ss["currency_rate"]
except KeyError:
    rate = 1.0
check(sym in ("$", "€", "£", "¥", "C$", "A$"), f"Unexpected currency symbol: {sym}")
check(rate > 0, f"Invalid currency rate: {rate}")

# 9. Date range filter actually filters data down to selected timeframe
#    We run a second AppTest instance with a narrow user-selected range.
from streamlit.testing.v1 import AppTest as AppTest2
at2 = AppTest.from_file("app.py")
# Use the last 2 days of available data so the range is within the dataset.
# The default range will be (dmax - 6 days, dmax), so we override with a narrower slice.
# Pick a fixed known range from the dataset (2026-07-01 to 2026-07-02).
at2.session_state["filter_date_range"] = (date(2026, 7, 1), date(2026, 7, 2))
at2.session_state["filter_date_range_user_set"] = True
at2.run()
check(bool(at2.exception) is False, f"AppTest (date filter) exception: {at2.exception}")

period_df2 = None
for el in at2.main:
    if type(el).__name__ == "Tab" and getattr(el, "label", None) == "By Period":
        for child in el:
            if type(child).__name__ == "Dataframe":
                period_df2 = child.value
                break

if period_df2 is not None and not period_df2.empty:
    date_col = None
    for c in ["Daily", "Weekly", "Monthly"]:
        if c in period_df2.columns:
            date_col = c
            break
    check(date_col is not None, "By Period (date filter) missing date column")
    if date_col:
        for v in period_df2[date_col].astype(str).tolist():
            if not ("2026-07-01" <= v <= "2026-07-02"):
                issues.append(f"Date {v} outside selected 2026-07-01 to 2026-07-02 range")
else:
    issues.append("By Period (date filter) dataframe empty or missing")

# 10. No all-negative numeric columns in any dataframe
for el in main:
    if type(el).__name__ == "Dataframe":
        df = el.value
        if df is not None and not df.empty:
            numeric = df.select_dtypes(include="number")
            for col in numeric.columns:
                s = numeric[col].dropna()
                if not s.empty and (s.max() < 0):
                    issues.append(f"Dataframe column {col} all negative")

print(json.dumps({"issues": issues, "tabs": labels, "metrics": metrics}))
'''
res = subprocess.run([".venv/bin/python", "-c", script], capture_output=True, text=True, timeout=120)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("EXIT:", res.returncode)
