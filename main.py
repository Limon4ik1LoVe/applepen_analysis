import pandas as pd
from pathlib import Path

ARCHIVE_DIR = Path("input")
OUTPUT_DIR = Path("output")

STORES = ["b1", "m1", "s1"]
STORE_STATE = {"b1": "MS", "m1": "MS", "s1": "MS"}

PRODUCT_CODE = {"ap": "apple", "pe": "pen"}


# -------------------- Load & preprocess --------------------

def load_store(store: str):
    """Read sell / supply / inventory CSVs."""
    return (
        pd.read_csv(ARCHIVE_DIR / f"MS-{store}-sell.csv", parse_dates=["date"]),
        pd.read_csv(ARCHIVE_DIR / f"MS-{store}-supply.csv", parse_dates=["date"]),
        pd.read_csv(ARCHIVE_DIR / f"MS-{store}-inventory.csv", parse_dates=["date"]),
    )


def get_daily_sales(sell: pd.DataFrame) -> pd.DataFrame:
    """Convert sell log → daily counts [apple, pen]."""
    sell["product"] = sell["sku_num"].str.split("-").str[2].map(PRODUCT_CODE)

    df = sell.groupby(["date", "product"]).size().unstack(fill_value=0)

    for col in ["apple", "pen"]:
        if col not in df:
            df[col] = 0

    return df[["apple", "pen"]]


# -------------------- Core logic --------------------

def process_store(store: str):
    sell, supply_raw, inventory_raw = load_store(store)

    sales = get_daily_sales(sell)
    supply = supply_raw.set_index("date").sort_index()
    inventory = inventory_raw.set_index("date").sort_index()

    inv_dates = inventory.index

    # --- старт: восстанавливаем остаток до первого месяца (без краж)
    first_end = inv_dates[0]
    first_start = first_end.replace(day=1)

    m_sup = supply.loc[first_start:first_end].sum()
    m_sal = sales.loc[first_start:first_end].sum()

    prev = {
        "apple": inventory.loc[first_end, "apple"] + m_sal["apple"] - m_sup["apple"],
        "pen":   inventory.loc[first_end, "pen"]   + m_sal["pen"]   - m_sup["pen"],
    }

    daily, stolen = [], []

    # --- идём по месяцам
    for end in inv_dates:
        start = end.replace(day=1)
        days = pd.date_range(start, end)

        mo_sup = supply.loc[start:end].reindex(days, fill_value=0)
        mo_sal = sales.loc[start:end].reindex(days, fill_value=0)

        cum_sup = mo_sup.cumsum()
        cum_sal = mo_sal.cumsum()

        # (1) ежедневный остаток
        for d in days:
            daily.append({
                "date": d.date(),
                "apple": prev["apple"] + cum_sup.loc[d, "apple"] - cum_sal.loc[d, "apple"],
                "pen":   prev["pen"]   + cum_sup.loc[d, "pen"]   - cum_sal.loc[d, "pen"],
            })

        # (2) кражи за месяц
        total_sup = mo_sup.sum()
        total_sal = mo_sal.sum()
        actual = inventory.loc[end]

        stolen.append({
            "date": end.date(),
            "apple": max(0, prev["apple"] + total_sup["apple"] - total_sal["apple"] - actual["apple"]),
            "pen":   max(0, prev["pen"]   + total_sup["pen"]   - total_sal["pen"]   - actual["pen"]),
        })

        # обновляем базу (берём реальный инвентарь)
        prev = {"apple": int(actual["apple"]), "pen": int(actual["pen"])}

    return pd.DataFrame(daily), pd.DataFrame(stolen), sales


# -------------------- Main --------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_stolen, all_sales = [], []

    for store in STORES:
        state = STORE_STATE[store]
        print(f"Processing {store}...")

        daily, stolen, sales = process_store(store)

        # (1) daily inventory
        daily.to_csv(OUTPUT_DIR / f"MS-{store}-daily_inventory.csv", index=False)

        # (2) monthly stolen
        stolen.to_csv(OUTPUT_DIR / f"MS-{store}-monthly_stolen.csv", index=False)

        # подготовка для (3)
        stolen["state"] = state
        stolen["year"] = pd.to_datetime(stolen["date"]).dt.year
        all_stolen.append(stolen)

        sales = sales.copy()
        sales["state"] = state
        sales["year"] = sales.index.year
        all_sales.append(sales.reset_index(drop=True))

    # (3) агрегация
    stolen_all = pd.concat(all_stolen)
    sales_all = pd.concat(all_sales)

    stolen_agg = stolen_all.groupby(["year", "state"])[["apple", "pen"]].sum()
    sales_agg = sales_all.groupby(["year", "state"])[["apple", "pen"]].sum()

    result = (
        sales_agg.rename(columns={"apple": "apple_sold", "pen": "pen_sold"})
        .merge(
            stolen_agg.rename(columns={"apple": "apple_stolen", "pen": "pen_stolen"}),
            on=["year", "state"],
        )
        .reset_index()
    )

    result.to_csv(OUTPUT_DIR / "sales_stolen_by_state_year.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()