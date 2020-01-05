"""
Functions to build new features from source dataframe.
"""

import pandas as pd


def split_ticket(df):
    return pd.DataFrame(df["Ticket"].apply(_split_ticket_apply_fn).tolist())


def cabin_character(df):
    return pd.DataFrame(df["Cabin"].apply(_clean_cabin).tolist())


def _make_ticket(prefix, number):
    return {"ticket_prefix": prefix, "ticket_number": number}


def _transform_ticket_number(ticket_no, use_nan_map=False):
    ticket_set = {"LINE"}
    if ticket_no in ticket_set:
        return pd.np.nan
    return int(ticket_no)


def _transform_ticket_prefix(ticket_prefix):
    if ticket_prefix == "STON/O2.":
        return "STON/O 2."
    return ticket_prefix


def _split_ticket_apply_fn(ticket):
    parts = ticket.split()
    if len(parts) == 0:
        return _make_ticket(pd.np.nan, np.pd.nan)
    elif len(parts) == 1:
        return _make_ticket(pd.np.nan, _transform_ticket_number(parts[0]))
    return _make_ticket(
        _transform_ticket_prefix(" ".join(parts[0:-1])),
        _transform_ticket_number(parts[-1]),
    )


def _clean_cabin(cabin):
    col_name = "cabin_char"
    if any(
        [cabin is None, cabin == pd.np.nan, isinstance(cabin, str) and len(cabin) == 0]
    ):
        return {col_name: pd.np.nan}
    if isinstance(cabin, str):
        return {col_name: cabin[0]}
    return {col_name: pd.np.nan}
