from decimal import ROUND_HALF_UP, Decimal


def fmt(x, show_sign=False):
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = str(d).rstrip("0").rstrip(".")
    if show_sign and x >= 0:
        s = "+" + s
    return s


def fmt_mm(x):
    if x == 0:
        return "0"
    return f"{fmt(x)}mm"


def fmt_deg(x):
    return f"{fmt(x)}°"


def fmt_reldeg(x):
    if x == 0:
        return "0"
    return f"{fmt(x, show_sign=True)}°"
