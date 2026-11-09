"""Создание учебных данных о пути покупателя в интернет-магазине.

Все строки создаются искусственно. В файле нет данных Ozon или реальных клиентов.
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from src import config


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Вернуть таблицы пользователей и событий с воспроизводимой случайностью."""
    rng = np.random.default_rng(config.SEED)
    signup_start = date.fromisoformat(config.START_DATE)
    signup_end = date.fromisoformat(config.SIGNUP_END_DATE)
    data_end = date.fromisoformat(config.DATA_END_DATE)
    signup_span = (signup_end - signup_start).days

    users = []
    for number in range(1, config.N_USERS + 1):
        signup_date = signup_start + timedelta(days=int(rng.integers(0, signup_span + 1)))
        channel = str(rng.choice(config.CHANNELS, p=config.CHANNEL_WEIGHTS))
        platform = str(rng.choice(config.PLATFORMS, p=config.PLATFORM_WEIGHTS))
        group = str(rng.choice(["control", "treatment"]))
        engagement = float(rng.beta(2.4, 3.2))

        users.append(
            {
                "user_id": f"U{number:05d}",
                "signup_date": signup_date.isoformat(),
                "channel": channel,
                "platform": platform,
                "experiment_group": group,
                "engagement_score": engagement,
            }
        )

    users_df = pd.DataFrame(users)
    events = []
    event_number = 0

    for user in users:
        signup_date = date.fromisoformat(user["signup_date"])
        max_day = min(
            config.MAX_FOLLOWUP_DAYS,
            (data_end - signup_date).days,
        )
        engagement = user["engagement_score"]
        channel_effect = {
            "Поиск": 0.01,
            "Реклама": -0.015,
            "Рекомендация": 0.035,
            "Прямой переход": 0.02,
        }[user["channel"]]

        active_days = [0]
        for day_offset in range(1, max_day + 1):
            probability = 0.025 + 0.12 * engagement + channel_effect
            if day_offset == 1:
                probability += 0.12
            elif day_offset == 7:
                probability += 0.10
            elif day_offset == 30:
                probability += 0.07
            probability = float(np.clip(probability, 0.01, 0.42))
            if rng.random() < probability:
                active_days.append(day_offset)

        for session_number, day_offset in enumerate(active_days, start=1):
            session_id = f'{user["user_id"]}_S{session_number:03d}'
            session_date = signup_date + timedelta(days=day_offset)
            start_time = datetime.combine(
                session_date,
                time(hour=int(rng.integers(8, 23)), minute=int(rng.integers(0, 60))),
            )
            cursor = start_time

            def add_event(event_name: str, amount: float | None = None) -> None:
                nonlocal event_number, cursor
                event_number += 1
                cursor += timedelta(minutes=int(rng.integers(1, 5)))
                events.append(
                    {
                        "event_id": f"E{event_number:08d}",
                        "user_id": user["user_id"],
                        "session_id": session_id,
                        "event_time": cursor.isoformat(sep=" "),
                        "event_name": event_name,
                        "platform": user["platform"],
                        "channel": user["channel"],
                        "experiment_group": user["experiment_group"],
                        "amount_rub": amount,
                    }
                )

            add_event("app_open")
            add_event("product_view")

            add_to_cart_probability = 0.30 + 0.15 * engagement
            if rng.random() < add_to_cart_probability:
                add_event("add_to_cart")
                if rng.random() < 0.48 + 0.12 * engagement:
                    add_event("checkout_started")

                    # Учебный эффект эксперимента: упрощённый чекаут
                    # повышает вероятность завершить покупку.
                    purchase_probability = 0.53 + 0.10 * (user["experiment_group"] == "treatment")
                    purchase_probability += 0.06 * (engagement - 0.5)
                    purchase_probability = float(np.clip(purchase_probability, 0.25, 0.85))
                    if rng.random() < purchase_probability:
                        amount = float(np.round(np.clip(rng.lognormal(8.15, 0.55), 300, 40000), -1))
                        add_event("purchase", amount)

    events_df = pd.DataFrame(events).sort_values(["event_time", "event_id"]).reset_index(drop=True)
    return users_df, events_df


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    users, events = generate()
    users.to_csv(config.DATA_DIR / "users.csv", index=False)
    events.to_csv(config.DATA_DIR / "events.csv", index=False)
    print(f"Создано пользователей: {len(users):,}")
    print(f"Создано событий: {len(events):,}")
    print(f"Файлы сохранены в: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
