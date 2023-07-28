import pandas as pd
import numpy as np
import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from functools import reduce

def combine_product(vars: list, data: pd.DataFrame):
    d = data[vars].astype(object)
    for v in vars:
        prod = v.split("_prod_")[1]
        d[v] = np.where(d[v] == 1, prod, None)
    d = d.replace("", None)
    return d.fillna(method='bfill', axis=1).iloc[:, 0]
        #reduce(lambda x, y: pd.concat([x, y], axis=1).combine_first(x), d)

def combine_product_all(vars_list: dict, data: pd.DataFrame):
    for v in vars_list.keys():
        data[v] = combine_product(vars_list[v], data)
    return data

def map_perm_product(df: pd.DataFrame, column_name: str):
    p = r"^(Whole Life Legacy|MassMutual Whole Life) (.+)"
    mapped_column = df[column_name].apply(lambda x: re.sub(p, r"\2", x) if re.search(p, x) else None)
    mapped_column[mapped_column.isin(['10', '12', '15', '20'])] = mapped_column[mapped_column.isin(
        ['10', '12', '15', '20'])] + ' Pay'
    return mapped_column


def map_term_product(df: pd.DataFrame, column_name: str):
    p = r"(HAVENTSIT|HAVENTSE|HAVENTSI|HAVENT|HAVENT2|MMCT|CP|TERM|VT|CVT|ET|^)(03|5|05|10|15|20|25|30|ART)(EH|H|G|E|REND|$)"
    mapped_column = df[column_name].apply(lambda x: re.sub(p, r"\2", x) if re.search(p, x) else None)
    mapped_column[mapped_column.isin(['5', '03', '05'])] = '3or5'
    return mapped_column


def map_di_product(df: pd.DataFrame, column_name: str):
    x = df[column_name].str.upper()
    y = np.repeat(None, len(x))
    y[np.where(x.str.contains("^MAX"))] = 'max'
    y[np.where(x.str.contains("RADIUS"))] = "radius"
    return y


def map_fa_product(df: pd.DataFrame, column_name: str):
    x = df[column_name].str.upper()
    y = np.repeat(None, len(x))
    y[np.where(x.str.contains("RETIREEASE"))] = 'retireease'
    y[np.where(x.str.contains("SINGLE PREM IMMEDIATE"))] = "retireease"
    y[np.where(x.str.contains("VOYAGE"))] = 'voyage'
    y[np.where(x.str.contains("INDEX"))] = 'index'
    y[np.where(x.str.contains("ODYSSEY"))] = 'odyssey'
    return y


def map_perm_risk(df: pd.DataFrame, column_name: str):
    mp = {"UP": "UP",
          "NT": "NT",
          "SPNT": "SPNT",
          "T": "T",
          "SPT": "SPT",
          "UPNT": "UP",
          "PNS": "SPNT",
          "NTP": "SPNT",
          "SNS": "NT",
          "NA": "NT",
          "SS": "T",
          "SINT": "NT",
          "SIT": "T",
          'SNT': 'NT',
          'ST': 'T'}

    if not all(df[column_name].dropna().isin(mp.keys())):
        raise ValueError("New Whole Life risk class detected")
    risk_column = df[column_name].map(mp)
    return risk_column


def map_term_risk(df, column_name):
    mp = {"UP": "UP",
          "NT": "NT",
          "SPNT": "SPNT",
          "T": "T",
          "SPT": "SPT",
          "UPNT": "UP",
          "PNS": "SPNT",
          "NTP": "SPNT",
          "SNS": "NT",
          "NA": "NT",
          "SS": "T",
          'PT': "SPT",
          'SNT': 'NT',
          'ST': 'T'}

    if not all(df[column_name].dropna().isin(mp.keys())):
        raise ValueError("New Term Life risk class detected")
    risk_column = df[column_name].map(mp)
    return risk_column


def map_ntl_risk(df, column_name):
    mp = {"UP": "UP",
          "NT": "NT",
          "SPNT": "SPNT",
          "T": "T",
          "SPT": "SPT",
          "UPNT": "UP",
          "PNS": "SPNT",
          "NTP": "SPNT",
          "SNS": "NT",
          "NA": "NT",
          "SS": "T",
          "SINT": "NT",
          "SIT": "T"}

    if not all(df[column_name].dropna().isin(mp.keys())):
        raise ValueError("New Non-Trad Life risk class detected")
    risk_column = df[column_name].map(mp)
    return risk_column


def map_di_risk(df: pd.DataFrame, column_name: str):
    x = df[column_name].replace(["", "NA"], None).str.upper()
    check_p = "^(1|2|3|4|5|6)[A-Za-z]$"
    if not all(x.str.contains(check_p, na=True)):
        raise ValueError("Unexpected DI occupation class")
    p = "(3|4|5)(A|P)"
    y = x.copy()
    id = y[(~y.str.contains(p, na=False)) & (~y.isna())].index
    y.loc[id] = y.loc[id].str[:1] + "A"
    return y


def map_term_ecp(column):
    y = [1 if re.search("ECP$", val, flags=re.IGNORECASE) else 0 for val in column]
    return y


def map_single_premium(df: pd.DataFrame, column_name: str):
    x = df[column_name]
    p = "NBL|No Bill"
    y = np.zeros(len(x))
    y[np.where(x.str.contains(p, case=False))] = 1
    y[np.where(x == "")] = 1
    y[np.where(x.isna())] = 1
    y[np.where(x == "Unknown")] = 1
    return y


def find_snapshot(date, target):
    date1 = date + relativedelta(years=target.year - date.year)
    date1 = date + relativedelta(years=target.year - date.year) - relativedelta(years=1)
    date1 = min(date, target) if date > target else date1
    date1 = date2 if date1 > target else date1
    return date1


def life_features(data: pd.DataFrame, product: str):
    '''
    Aggregate the Life policy features at the owner_ssn level

        Args:
            dad: pd.DataFrame of the Whole Life, Term Life, or Non-Trad Life data to be aggregated
            product: string of the product name, either "perm", "term", or "ntl"

        Returns:
            pd.DataFrame of the Life policy features after processing and aggregation, one row per owner
        '''
    # Take the max of each rider indicator
    max_data = data.groupby('owner_ssn').agg({
        col: lambda column: column.max(skipna=True) for col in data.columns if col.startswith(f"{product}_rider") or
                                                                               col.startswith(f"{product}_prod")
    }).reset_index()

    # Calculate sum of face amount and premium
    sum_data = data.groupby('owner_ssn').agg({
        f"{product}_face_amount": 'sum',
        f"{product}_premium": 'sum'
    }).reset_index()

    # Get the most recent risk for each owner_ssn
    recent_data = data.sort_values(by=['owner_ssn', f"{product}_risk", 'issue_date'], ascending=[True, True, False])
    recent_data = recent_data.drop_duplicates(subset='owner_ssn', keep='first')[['owner_ssn', f"{product}_risk"]]

    # Join the resulting DataFrames
    result = max_data.merge(sum_data, on='owner_ssn').merge(recent_data, on='owner_ssn')
    return pd.DataFrame(result)