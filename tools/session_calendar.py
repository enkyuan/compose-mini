"""Read the frozen U.S. equities core-session calendar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALENDAR = (
    ROOT / "universes/us-equities-core-2024-07-22_2026-07-21.json"
)
FIELDS = {
    "schema", "purpose", "venues", "timezone", "start", "end",
    "open_minute", "close_minute", "closed_dates", "early_closes", "sources",
}


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("session calendar contains a duplicate field")
        value[name] = item
    return value


def _date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("session calendar date must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("session calendar date must be an ISO date") from error
    if str(parsed) != value:
        raise ValueError("session calendar date must be an ISO date")
    return parsed


@dataclass(frozen=True)
class SessionCalendar:
    start: date
    end: date
    open_minute: int
    close_minute: int
    venues: tuple[str, ...]
    closed_dates: tuple[date, ...]
    early_closes: tuple[tuple[date, int], ...]

    @classmethod
    def read(cls, path: Path) -> SessionCalendar:
        """Parse one strict, nonsymlink calendar input."""
        if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
            raise ValueError("session calendar must be a regular file")
        try:
            value = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("session calendar is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != FIELDS or \
           type(value["schema"]) is not int or value["schema"] != 1 or \
           value["timezone"] != "America/New_York" or \
           not isinstance(value["purpose"], str) or \
           not value["purpose"].strip() or value["venues"] != ["XNAS", "XNYS"]:
            raise ValueError("session calendar fields are invalid")
        start, end = _date(value["start"]), _date(value["end"])
        open_minute, close_minute = value["open_minute"], value["close_minute"]
        if type(open_minute) is not int or type(close_minute) is not int or \
           not 0 <= open_minute < close_minute <= 24 * 60 or start > end:
            raise ValueError("session calendar bounds are invalid")
        closed_value, early_value, sources = (
            value["closed_dates"], value["early_closes"], value["sources"],
        )
        if not isinstance(closed_value, list) or \
           not isinstance(early_value, Mapping) or \
           not isinstance(sources, list):
            raise ValueError("session calendar collections are invalid")
        closed = tuple(_date(item) for item in closed_value)
        early = tuple((_date(day), minute) for day, minute in early_value.items())
        if closed != tuple(sorted(set(closed))) or \
           tuple(day for day, _ in early) != tuple(sorted({
               day for day, _ in early
           })) or any(
               day.weekday() >= 5 or not start <= day <= end for day in closed
           ) or any(
               day.weekday() >= 5 or not start <= day <= end or
               type(minute) is not int or not open_minute < minute < close_minute
               for day, minute in early
           ) or set(closed) & {day for day, _ in early}:
            raise ValueError("session calendar dates are invalid")
        if not sources or any(
            not isinstance(source, str) or not source.startswith("https://")
            for source in sources
        ) or len(sources) != len(set(sources)):
            raise ValueError("session calendar sources are invalid")
        return cls(
            start, end, open_minute, close_minute,
            tuple(value["venues"]), closed, early,
        )

    def session(self, day: date) -> tuple[int, int] | None:
        """Return local core-session bounds, or None for a closed date."""
        if not self.start <= day <= self.end:
            raise ValueError("date is outside the session calendar")
        if day.weekday() >= 5 or day in self.closed_dates:
            return None
        return self.open_minute, next(
            (close for session, close in self.early_closes if session == day),
            self.close_minute,
        )
