from datetime import datetime, timedelta


def parse_date(value: str) -> datetime:
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid date '{value}'. Use YYYYMMDD or YYYY-MM-DD.")


def daterange(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
