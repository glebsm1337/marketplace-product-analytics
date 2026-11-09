"""Расчёт продуктовых метрик и создание графиков для учебной симуляции."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


FUNNEL_STAGES = [
    ("Просмотр товара", "product_view"),
    ("Добавление в корзину", "add_to_cart"),
    ("Начало оформления", "checkout_started"),
    ("Покупка", "purchase"),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(config.DATA_DIR / "users.csv", parse_dates=["signup_date"])
    events = pd.read_csv(config.DATA_DIR / "events.csv", parse_dates=["event_time"])
    return users, events


def calculate_funnel(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Считает уникальных пользователей, достигших каждого шага."""
    reached = [users["user_id"].nunique()]
    rows = [{
        "этап": "Регистрация",
        "пользователи": reached[0],
        "конверсия_от_предыдущего_шага": 1.0,
        "конверсия_от_начала": 1.0,
    }]

    for label, event_name in FUNNEL_STAGES:
        count = events.loc[events["event_name"] == event_name, "user_id"].nunique()
        previous = reached[-1]
        reached.append(count)
        rows.append({
            "этап": label,
            "пользователи": count,
            "конверсия_от_предыдущего_шага": count / previous if previous else 0.0,
            "конверсия_от_начала": count / reached[0] if reached[0] else 0.0,
        })
    return pd.DataFrame(rows)


def calculate_retention(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Считает точное дневное удержание D1, D7 и D30."""
    active_dates = (
        events.assign(activity_date=events["event_time"].dt.normalize())
        .groupby("user_id")["activity_date"]
        .agg(lambda values: set(values))
        .to_dict()
    )
    data_end = pd.Timestamp(config.DATA_END_DATE)
    rows = []

    for day in (1, 7, 30):
        eligible = users[users["signup_date"] + pd.to_timedelta(day, unit="D") <= data_end]
        returned = 0
        for user in eligible.itertuples():
            target_date = user.signup_date.normalize() + pd.Timedelta(days=day)
            if target_date in active_dates.get(user.user_id, set()):
                returned += 1
        rows.append({
            "день": f"D{day}",
            "доступная_когорта": len(eligible),
            "вернувшиеся": returned,
            "удержание": returned / len(eligible) if len(eligible) else 0.0,
        })
    return pd.DataFrame(rows)


def calculate_cohorts(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу долей активных пользователей по месяцу когорты."""
    user_cohorts = users[["user_id", "signup_date"]].copy()
    user_cohorts["когорта"] = user_cohorts["signup_date"].dt.to_period("M").astype(str)

    activity = events[["user_id", "event_time"]].copy()
    activity["месяц_активности"] = activity["event_time"].dt.to_period("M")
    activity = activity.merge(user_cohorts[["user_id", "когорта"]], on="user_id", how="left")
    activity["месяц_когорты"] = pd.PeriodIndex(activity["когорта"], freq="M")
    activity["период"] = (
        (activity["месяц_активности"].dt.year - activity["месяц_когорты"].dt.year) * 12
        + activity["месяц_активности"].dt.month
        - activity["месяц_когорты"].dt.month
    )

    cohort_sizes = user_cohorts.groupby("когорта")["user_id"].nunique()
    active = activity.groupby(["когорта", "период"])["user_id"].nunique()
    rates = active.div(cohort_sizes, level="когорта").rename("удержание").reset_index()
    return rates.pivot(index="когорта", columns="период", values="удержание").sort_index()


def calculate_ab_test(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает долю покупателей в контрольной и тестовой группах."""
    buyers = set(events.loc[events["event_name"] == "purchase", "user_id"])
    frame = users[["user_id", "experiment_group"]].copy()
    frame["converted"] = frame["user_id"].isin(buyers)

    grouped = frame.groupby("experiment_group")["converted"].agg(["sum", "count"])
    control_n = int(grouped.loc["control", "count"])
    treatment_n = int(grouped.loc["treatment", "count"])
    control_rate = float(grouped.loc["control", "sum"] / control_n)
    treatment_rate = float(grouped.loc["treatment", "sum"] / treatment_n)
    difference = treatment_rate - control_rate

    pooled_rate = (
        int(grouped.loc["control", "sum"]) + int(grouped.loc["treatment", "sum"])
    ) / (control_n + treatment_n)
    pooled_se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / control_n + 1 / treatment_n))
    z_score = difference / pooled_se if pooled_se else 0.0
    p_value = math.erfc(abs(z_score) / math.sqrt(2))

    unpooled_se = math.sqrt(
        control_rate * (1 - control_rate) / control_n
        + treatment_rate * (1 - treatment_rate) / treatment_n
    )
    ci_low = difference - 1.96 * unpooled_se
    ci_high = difference + 1.96 * unpooled_se

    return pd.DataFrame([
        {
            "группа": "control",
            "пользователи": control_n,
            "покупатели": int(grouped.loc["control", "sum"]),
            "конверсия": control_rate,
            "разница_с_control": 0.0,
            "95%_ДИ_нижняя": np.nan,
            "95%_ДИ_верхняя": np.nan,
            "p_value": np.nan,
            "z_score": np.nan,
        },
        {
            "группа": "treatment",
            "пользователи": treatment_n,
            "покупатели": int(grouped.loc["treatment", "sum"]),
            "конверсия": treatment_rate,
            "разница_с_control": difference,
            "95%_ДИ_нижняя": ci_low,
            "95%_ДИ_верхняя": ci_high,
            "p_value": p_value,
            "z_score": z_score,
        },
    ])


def calculate_segments(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    buyers = set(events.loc[events["event_name"] == "purchase", "user_id"])
    frame = users.copy()
    frame["покупатель"] = frame["user_id"].isin(buyers)
    rows = []
    for dimension in ("channel", "platform"):
        grouped = frame.groupby(dimension, dropna=False)["покупатель"].agg(["sum", "count"])
        for segment, metrics in grouped.iterrows():
            rows.append({
                "срез": dimension,
                "значение": segment,
                "пользователи": int(metrics["count"]),
                "покупатели": int(metrics["sum"]),
                "конверсия": float(metrics["sum"] / metrics["count"]),
            })
    return pd.DataFrame(rows)


def save_charts(funnel: pd.DataFrame, retention: pd.DataFrame, ab_test: pd.DataFrame) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(funnel["этап"], funnel["пользователи"], color="#1769aa")
    plt.title("Воронка покупки")
    plt.ylabel("Уникальные пользователи")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "funnel.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(retention["день"], retention["удержание"] * 100, color="#00a676")
    plt.title("Точное дневное удержание")
    plt.ylabel("Пользователи, %")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "retention.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(ab_test["группа"], ab_test["конверсия"] * 100, color=["#8da0cb", "#fc8d62"])
    plt.title("Конверсия в покупку по A/B-группам")
    plt.ylabel("Пользователи с покупкой, %")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "ab_test.png", dpi=160)
    plt.close()


def main() -> None:
    users, events = load_data()
    funnel = calculate_funnel(users, events)
    retention = calculate_retention(users, events)
    cohorts = calculate_cohorts(users, events)
    ab_test = calculate_ab_test(users, events)
    segments = calculate_segments(users, events)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(config.REPORTS_DIR / "funnel.csv", index=False)
    retention.to_csv(config.REPORTS_DIR / "retention.csv", index=False)
    cohorts.to_csv(config.REPORTS_DIR / "cohort_retention.csv")
    ab_test.to_csv(config.REPORTS_DIR / "ab_test.csv", index=False)
    segments.to_csv(config.REPORTS_DIR / "segments.csv", index=False)
    save_charts(funnel, retention, ab_test)

    summary = {
        "пользователи": int(users["user_id"].nunique()),
        "события": int(len(events)),
        "покупатели": int(events.loc[events["event_name"] == "purchase", "user_id"].nunique()),
        "выручка_руб": float(events["amount_rub"].sum()),
        "эксперимент_p_value": float(ab_test.loc[1, "p_value"]),
        "эксперимент_uplift_пп": float(ab_test.loc[1, "разница_с_control"] * 100),
        "примечание": "Все данные синтетические; это учебная симуляция, не показатели реального бизнеса.",
    }
    (config.REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Пользователи:", summary["пользователи"])
    print("События:", summary["события"])
    print("Покупатели:", summary["покупатели"])
    print(f'Синтетическая выручка: {summary["выручка_руб"]:,.0f} ₽')
    print(f'Разница конверсии теста: {summary["эксперимент_uplift_пп"]:.2f} п.п.')
    print(f'p-value: {summary["эксперимент_p_value"]:.4f}')
    print("Отчёты сохранены в:", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
