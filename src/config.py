"""Параметры учебной симуляции и пути к файлам."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

SEED = 20261109
N_USERS = 3000
START_DATE = "2025-01-01"
SIGNUP_END_DATE = "2025-06-30"
DATA_END_DATE = "2025-09-30"
MAX_FOLLOWUP_DAYS = 90

CHANNELS = ["Поиск", "Реклама", "Рекомендация", "Прямой переход"]
CHANNEL_WEIGHTS = [0.35, 0.25, 0.18, 0.22]
PLATFORMS = ["Android", "iOS", "Web"]
PLATFORM_WEIGHTS = [0.48, 0.32, 0.20]
