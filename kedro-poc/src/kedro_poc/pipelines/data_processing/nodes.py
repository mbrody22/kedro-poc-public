import pandas as pd
import numpy as np
import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from typing import Dict
from ...extras.utils.model_data_utils import *
from datetime import datetime


def process_demo(client: pd.DataFrame, zipincome: pd.DataFrame) -> pd.DataFrame:
    '''
    Args:
        client: pd.DataFrame of the raw client demographics data
        zipincome: pd.DataFrame of median income by zip code

    Returns:
        pd.DataFrame of the client demographics feature set after processing and feature engineering
    '''
    # Create date columns
    client[['issue_date', 'term_date', 'owner_dob', 'begin_dt', 'insured_dob']] = client[['issue_date', 'term_date', 'owner_dob', 'begin_dt', 'insured_dob']].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Set term date for in-force policies as 200 years from today
    client.loc[client['policy_status'] == 'IF', 'term_date'] = dt.datetime.now() + relativedelta(years=200)

    # Remove time aspect from dates
    #client[['issue_date', 'term_date', 'owner_dob', 'begin_dt', 'insured_dob']].apply(dt.date)

    client['self_insured'] = (client['owner_id'] == client['insured_id']).astype(int)
    client['self_insured'].fillna(1)
    client['age_diff'] = round((client['insured_dob'] - client['owner_dob']).dt.days / 365.25, 1)
    client['age_diff'] = client['age_diff'].fillna(0)

    client = client[['agreement_id', 'major_product', 'policy_status', 'status_rsn', 'term_date', 'issue_date',
                     'owner_id', 'owner_dob', "owner_gender", "owner_ssn", "owner_zip", 'self_insured', 'age_diff']]
    client = client.drop_duplicates()

    # Deduplicate by owner_ssn
    client = client.sort_values(by=['agreement_id', 'owner_ssn', 'self_insured'], ascending=False)
    client.drop_duplicates(subset=["agreement_id", "owner_ssn"], keep="first", inplace=True)

    zipincome['owner_zip'] = zipincome['owner_zip'].astype(int).astype(str).str.rjust(5, '0')
    client = client.merge(zipincome, on='owner_zip', how='left')

    return client


def process_riders(riders: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            riders: pd.DataFrame of the raw policy rider data

        Returns:
            pd.DataFrame of the unique riders for each policy
    '''
    # Convert hldg_key column to numeric
    riders['hldg_key'] = riders['hldg_key'].apply(pd.to_numeric, errors='coerce')
    # Remove NAs and drop duplicates
    riders.dropna(inplace=True)
    riders = riders.drop_duplicates()
    return riders

def process_perm(perm: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            perm: pd.DataFrame of the raw Whole Life policy data
            riders: pd.DataFrame of the unique riders for each policy

        Returns:
            pd.DataFrame of Whole Life policy features
    '''
    # Set column data types
    perm[["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'perm_face_amount']] = perm[
        ["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'perm_face_amount']].apply(pd.to_numeric, errors='coerce')
    perm['issue_date'] = perm['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Process premium--------------------------------------------------------------
    perm['premium'] = np.where(perm['premium'] == 0, float('NaN'), perm['premium'])
    perm['prem_sparc'] = np.where(perm['issue_date'] > dt.datetime(2018, 12, 31), float('NaN'), perm['prem_sparc'])
    perm['prem_siera'] = np.where(perm['issue_date'] <= dt.datetime(2018, 12, 31), float('NaN'), perm['prem_siera'])
    perm[["premium", "prem_hist", "prem_sparc", "prem_siera"]] = perm[
        ["premium", "prem_hist", "prem_sparc", "prem_siera"]].apply(pd.to_numeric, errors='coerce')
    perm['perm_premium'] = perm[['premium', 'prem_hist', 'prem_sparc', 'prem_siera']].bfill(axis=1).iloc[:, 0]

    # Minor product features-------------------------------------------------------
    perm['perm_product'] = map_perm_product(perm, 'prod_typ_nme')
    perm.dropna(subset=['perm_product'], inplace=True)

    # Map risk classes -----------------------------------------------
    perm['perm_risk'] = map_perm_risk(perm, 'risk')

    # Riders ---------------------------------------------------------------------------
    riders['hldg_key'] = riders['hldg_key'].astype(int)
    perm_riders = pd.DataFrame(perm['hldg_key'].unique(), columns=['hldg_key']).merge(riders, on='hldg_key', how='left')
    perm_riders['value'] = 1
    perm_riders['prod_cd'] = perm_riders['prod_cd'].str.lower()
    perm_riders = pd.pivot_table(perm_riders, index='hldg_key', columns='prod_cd', values='value', aggfunc='sum', fill_value=0)
    perm_riders.columns = ['perm_rider_' + str(col) for col in perm_riders.columns]

    if sum(perm_riders.columns.str.contains('perm_rider_ltc')) > 1:
        raise ValueError("Additional LTC Riders found")

    perm_riders.reset_index(inplace=True)
    perm_riders = perm_riders[['hldg_key', 'perm_rider_wp', 'perm_rider_gir', 'perm_rider_rtr1', 'perm_rider_alir',
                               'perm_rider_ltcr', 'perm_rider_lisr']]
    perm = perm.merge(perm_riders, on='hldg_key', how='left')

    # Create features to be aggregated ----------------------------------------------------
    perm['perm_product'] = perm['perm_product'].str.lower().str.replace(" ", "_")
    # Create pivot table for 'perm_prod_ind'
    perm_prod_ind = pd.pivot_table(perm.assign(value=1), index='agreement_id', columns='perm_product',
                                   values='value', aggfunc='sum', fill_value=0)
    # Add a prefix to the column names
    perm_prod_ind.columns = ['perm_prod_' + col for col in perm_prod_ind.columns]
    perm_prod_ind.reset_index(inplace=True)

    perm = perm.merge(perm_prod_ind, on='agreement_id', how='left')

    selected_columns = ['agreement_id', 'perm_face_amount', 'perm_premium', 'perm_risk'] + \
                        perm.columns[perm.columns.str.startswith('perm_rider_')].tolist() + \
                        perm.columns[perm.columns.str.startswith('perm_prod_')].tolist()

    return perm[selected_columns]

def process_term(term: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            term: pd.DataFrame of the raw Term Life policy data
            riders: pd.DataFrame of the unique riders for each policy

        Returns:
            pd.DataFrame of Term Life policy features
    '''
    # Set column data types
    term[["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'term_face_amount']] = term[
        ["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'term_face_amount']].apply(pd.to_numeric, errors='coerce')
    term['issue_date'] = term['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Process premium--------------------------------------------------------------
    term['premium'] = np.where(term['premium'] == 0, float('NaN'), term['premium'])
    term['prem_sparc'] = np.where(term['issue_date'] > dt.datetime(2018, 12, 31), float('NaN'), term['prem_sparc'])
    term['prem_siera'] = np.where(term['issue_date'] <= dt.datetime(2018, 12, 31), float('NaN'), term['prem_siera'])
    term[["premium", "prem_hist", "prem_sparc", "prem_siera"]] = term[
        ["premium", "prem_hist", "prem_sparc", "prem_siera"]].apply(pd.to_numeric, errors='coerce')
    term['term_premium'] = term[['premium', 'prem_hist', 'prem_sparc', 'prem_siera']].bfill(axis=1).iloc[:, 0]
    term['term_premium'] = np.where(term['term_premium'] < 0, 0, term['term_premium'])
    term['term_premium'] = term['term_premium'].fillna(0)

    # Minor product features-------------------------------------------------------
    term['term_product'] = map_term_product(term, 'prod_typ_cde')
    term['term_prod_ecp'] = map_term_ecp(term['prod_typ_nme'])
    term.dropna(subset=['term_product'], inplace=True)

    # Map risk classes -----------------------------------------------
    term['term_risk'] = map_term_risk(term, 'risk')

    # Riders ---------------------------------------------------------------------------
    riders['hldg_key'] = riders['hldg_key'].astype(int)
    term_riders = pd.DataFrame(term['hldg_key'].unique(), columns=['hldg_key']).merge(riders, on='hldg_key', how='left')
    term_riders['value'] = 1
    term_riders['prod_cd'] = term_riders['prod_cd'].str.lower()
    term_riders = pd.pivot_table(term_riders, index='hldg_key', columns='prod_cd', values='value', aggfunc='sum', fill_value=0)
    term_riders.columns = ['term_rider_' + str(col) for col in term_riders.columns]

    term_riders.reset_index(inplace=True)
    term_riders = term_riders[['hldg_key', 'term_rider_wp']]
    term = term.merge(term_riders, on='hldg_key', how='left')

    # Create features to be aggregated ----------------------------------------------------
    term['term_product'] = term['term_product'].str.lower().str.replace(" ", "_")
    # Create pivot table for 'term_prod_ind'
    term_prod_ind = pd.pivot_table(term.assign(value=1), index='agreement_id', columns='term_product',
                                   values='value', aggfunc='sum', fill_value=0)
    # Add a prefix to the column names
    term_prod_ind.columns = ['term_prod_' + col for col in term_prod_ind.columns]
    term_prod_ind.reset_index(inplace=True)

    term = term.merge(term_prod_ind, on='agreement_id', how='left')

    selected_columns = ['agreement_id', 'term_face_amount', 'term_premium', 'term_risk'] + \
                        term.columns[term.columns.str.startswith('term_rider_')].tolist() + \
                        term.columns[term.columns.str.startswith('term_prod_')].tolist()

    return term[selected_columns]


def process_ntl(ntl: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            ntl: pd.DataFrame of the raw Non-Trad Life policy data
            riders: pd.DataFrame of the unique riders for each policy

        Returns:
            pd.DataFrame of Non-Trad Life policy features
    '''
    # Set column data types
    ntl[["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'ntl_face_amount']] = ntl[
        ["hldg_key", "premium", "prem_hist", "prem_sparc", "prem_siera", 'ntl_face_amount']].apply(pd.to_numeric,
                                                                                                    errors='coerce')
    ntl['issue_date'] = ntl['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Process premium--------------------------------------------------------------
    ntl['premium'] = np.where(ntl['premium'] == 0, float('NaN'), ntl['premium'])
    ntl['prem_sparc'] = np.where(ntl['issue_date'] > dt.datetime(2018, 12, 31), float('NaN'), ntl['prem_sparc'])
    ntl['prem_siera'] = np.where(ntl['issue_date'] <= dt.datetime(2018, 12, 31), float('NaN'), ntl['prem_siera'])
    ntl[["premium", "prem_hist", "prem_sparc", "prem_siera"]] = ntl[
        ["premium", "prem_hist", "prem_sparc", "prem_siera"]].apply(pd.to_numeric, errors='coerce')
    ntl['ntl_premium'] = ntl[['premium', 'prem_hist', 'prem_sparc', 'prem_siera']].bfill(axis=1).iloc[:, 0]
    ntl['ntl_premium'] = np.where(ntl['ntl_premium'] < 0, 0, ntl['ntl_premium'])
    ntl['ntl_premium'] = ntl['ntl_premium'].fillna(0)

    # Minor product features-------------------------------------------------------
    ntl['ntl_prod_vl_nonprop'] = np.where((ntl['minor_prod_cde'] == "NONPROP") | (ntl['minor_prod_cde'] == "ONPROP"), 1, 0)
    ntl['ntl_prod_vl_prop'] = np.where(ntl['minor_prod_cde'].str.contains("^STRAT|^SVUL|^VL"), 1, 0)
    ntl['ntl_prod_ul'] = np.where(ntl['minor_prod_cde'] == "UL", 1, 0)
    ntl['any_prod'] = ntl['ntl_prod_ul'] + ntl['ntl_prod_vl_nonprop'] + ntl['ntl_prod_vl_prop']
    ntl = ntl.loc[ntl['any_prod'] > 0]

    # Map risk classes -----------------------------------------------
    ntl['ntl_risk'] = map_ntl_risk(ntl, 'risk')

    # Riders ---------------------------------------------------------------------------
    riders['hldg_key'] = riders['hldg_key'].astype(int)
    ntl_riders = riders[riders['prod_cd'].isin(["VULIII WSP", "VULIII WMC", "DBR"])]
    ntl_prop = ntl[ntl['ntl_prod_vl_nonprop'] == 0]
    ntl_prop = ntl_prop[['agreement_id', 'hldg_key']].drop_duplicates()
    ntl_riders = pd.DataFrame(ntl_prop, columns=['agreement_id', 'hldg_key']).merge(ntl_riders, on='hldg_key', how='left')
    ntl_riders['value'] = 1
    ntl_riders = pd.pivot_table(ntl_riders, index=['agreement_id', 'hldg_key'], columns='prod_cd', values='value', aggfunc='sum',
                                 fill_value=0)
    ntl_riders.rename(columns={"DBR": "ntl_rider_dbr", "VULIII WSP": "ntl_rider_wsp", "VULIII WMC": "ntl_rider_wmc"},
                      inplace=True)
    ntl_riders.reset_index(inplace=True)
    ntl = ntl.merge(ntl_riders, on=['agreement_id', 'hldg_key'], how='left')

    # Set NA for columns for non-prop VL as they are unavailable -------------------
    ntl['ntl_face_amount'] = np.where(ntl['ntl_prod_vl_nonprop'] == 1, float('NaN'), ntl['ntl_face_amount'] )
    ntl['ntl_premium'] = np.where(ntl['ntl_prod_vl_nonprop'] == 1, float('NaN'), ntl['ntl_premium'])

    selected_columns = ['agreement_id', 'ntl_face_amount', 'ntl_premium', 'ntl_risk'] + \
                       ntl.columns[ntl.columns.str.startswith('ntl_rider_')].tolist() + \
                       ntl.columns[ntl.columns.str.startswith('ntl_prod_')].tolist()

    return ntl[selected_columns]

def process_di(di: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            di: pd.DataFrame of the raw Disability Income Insurance policy data
            riders: pd.DataFrame of the unique riders for each policy

        Returns:
            pd.DataFrame of DI policy features
    '''
    # Set column data types
    di[["hldg_key", "di_premium", "di_ben"]] = di[["hldg_key", "di_premium", "di_ben"]].apply(pd.to_numeric, errors='coerce')
    di['issue_date'] = di['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # De-duplicate by agreement ID
    di.drop_duplicates(subset=["agreement_id"], keep="first", inplace=True)

    # Map products ----------------------------------------------
    di['di_product'] = map_di_product(di, "prod_typ_nme")
    di.dropna(subset=['di_product'], inplace=True)

    # Map risk classes -------------------------------------------
    di['di_risk'] = map_di_risk(di, "occupation_class")

    # DI Riders --------------------------------------------------
    di_riders = pd.DataFrame(di['hldg_key'].drop_duplicates())
    di_riders = di_riders.merge(riders, on='hldg_key', how='left')
    di_riders = di_riders.loc[di_riders['prod_cd'].isin(["PARTIAL", "CAT", "COLA", "OPTION", "OWN OCC", "RETIRE"])]
    di_riders['value'] = 1
    di_riders = pd.pivot_table(di_riders, index='hldg_key', columns='prod_cd', values='value', aggfunc='sum', fill_value=0)
    di_riders.columns = di_riders.columns.str.lower().str.replace(" ", "_")
    di_riders.columns = ['di_rider_' + col for col in di_riders.columns]
    di_riders.reset_index(inplace=True)
    di = di.merge(di_riders, on="hldg_key", how="left")

    # DI Products --------------------------------------------------
    di_prod_ind = di.copy()
    di_prod_ind["value"] = 1
    di_prod_ind = pd.pivot_table(di_prod_ind, index='agreement_id', columns='di_product', values='value', aggfunc='sum', fill_value=0)
    di_prod_ind.columns = ['di_prod_' + col for col in di_prod_ind.columns]
    di_prod_ind.reset_index(inplace=True)
    di = di.merge(di_prod_ind, on='agreement_id', how='left')

    selected_columns = ['agreement_id', 'di_risk', 'di_premium', 'di_ben'] + \
                       di.columns[di.columns.str.startswith('di_rider_')].tolist() + \
                       di.columns[di.columns.str.startswith('di_prod_')].tolist()

    return di[selected_columns]

def process_fa(fa: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            fa: pd.DataFrame of the raw Fixed Annuity policy data

        Returns:
            pd.DataFrame of Fixed Annuity policy features
    '''
    # Set column data types
    fa[["hldg_key", "premium", "fa_init_pay", "accum_val_net_prem_amt"]] = \
        fa[["hldg_key", "premium", "fa_init_pay", "accum_val_net_prem_amt"]].apply(pd.to_numeric, errors='coerce')
    fa['issue_date'] = fa['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Process FA Products -----------------------------------------------
    fa['fa_product'] = map_fa_product(fa, "prod_typ_nme")
    fa.dropna(subset=['fa_product'], inplace=True)
    fa_prod_ind = fa.copy()
    fa_prod_ind["value"] = 1
    fa_prod_ind = pd.pivot_table(fa_prod_ind, index='agreement_id', columns='fa_product', values='value', aggfunc='sum',
                                 fill_value=0)
    fa_prod_ind.columns = ['fa_prod_' + col for col in fa_prod_ind.columns]
    fa_prod_ind.reset_index(inplace=True)
    fa = fa.merge(fa_prod_ind, on='agreement_id', how='left')

    selected_columns = ['agreement_id', 'fa_init_pay'] + fa.columns[fa.columns.str.startswith('fa_prod_')].tolist()

    return fa[selected_columns]

def process_va(va: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            va: pd.DataFrame of the raw Variable Annuity policy data

        Returns:
            pd.DataFrame of Variable Annuity policy features
    '''
    # Set column data types
    va[["hldg_key", "premium", "va_init_pay", "accum_val_net_prem_amt"]] = \
        va[["hldg_key", "premium", "va_init_pay", "accum_val_net_prem_amt"]].apply(pd.to_numeric, errors='coerce')
    va['issue_date'] = va['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    # Process VA Minor Products -----------------------------------------------
    va['va_prod_prop'] = np.where(va['prod_typ_cde'].str.contains("^VA NP"), 0, 1)
    va['va_prod_nonprop'] = np.where(va['prod_typ_cde'].str.contains("^VA NP"), 1, 0)

    # Map Single Premium ------------------------------------------------------
    va['va_single_prem'] = map_single_premium(va, "payment_mode")

    va = va[['agreement_id', 'va_init_pay', 'va_single_prem'] + va.columns[va.columns.str.startswith('va_prod_')].tolist()]
    va['va_init_pay'] = np.where(va['va_prod_nonprop'] == 1, None, va['va_init_pay'])
    va['va_single_prem'] = np.where(va['va_prod_nonprop'] == 1, None, va['va_single_prem'])

    return va

def create_model_data(client, fp_client, permdata, termdata, ntldata, didata, fadata, vadata, parameters: Dict) -> pd.DataFrame:
    '''
        Args:
            client: pd.DataFrame of the processed client demographics data
            fp_client: pd.DataFrame of all financial planning clients
            permdata: pd.DataFrame of the processed Whole Life demographics data
            termdata: pd.DataFrame of the processed Term Life demographics data
            ntldata: pd.DataFrame of the processed Non-Trad Life demographics data
            didata: pd.DataFrame of the processed DI data
            fadata: pd.DataFrame of the processed Fixed Annuity data
            vadata: pd.DataFrame of the processed Variable Annuity data
            parameters: Dict of model data parameters (max_date and target product)

        Returns:
            pd.DataFrame of model features
    '''
    # Parameters --------------------------------
    d = parameters['max_date']
    max_date = dt.datetime(d.year, d.month, d.day)
    min_date = max_date - relativedelta(years=6)
    target_product = parameters['target_product']
    # ----------------------------------------------------------------------

    client = client.merge(pd.concat([pd.DataFrame(termdata['agreement_id']),
                                    pd.DataFrame(permdata['agreement_id']),
                                    pd.DataFrame(ntldata['agreement_id']),
                                    pd.DataFrame(didata['agreement_id']),
                                    pd.DataFrame(fadata['agreement_id']),
                                    pd.DataFrame(vadata['agreement_id'])]), on='agreement_id', how='inner')

    client['major_product'] = client['major_product'].str.lower().str.replace(" ", "_")
    client['is_protect'] = 1

    # Create purchase history:
    pur_hist = client[['owner_ssn', 'agreement_id', 'major_product', 'is_protect', 'issue_date', 'term_date']]
    pur_hist['issue_date'] = pur_hist['issue_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')
    pur_hist['term_date'] = pur_hist['term_date'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')
    pur_hist = pur_hist.loc[pur_hist['issue_date'] >= dt.datetime(2010, 1, 1)]

    model_times = pd.DataFrame(pur_hist['owner_ssn']).drop_duplicates()
    model_times['snapshot_time'] = max_date
    model_times['outcome_time'] = model_times['snapshot_time'] + pd.offsets.DateOffset(years=1)
    pur_hist = pur_hist.merge(model_times, how='inner', on='owner_ssn')

    # Apply inclusion/exclusion criteria ----------------------------------------------
    # Filter by issue date before snapshot date
    purchased = pur_hist.loc[pur_hist['issue_date'] <= pur_hist['snapshot_time']]
    purchased['inforce'] = (purchased['term_date'] > purchased['snapshot_time']).astype(int)
    purchased = purchased[['owner_ssn', 'agreement_id', 'major_product', 'is_protect', 'inforce', 'issue_date',
                           'snapshot_time', 'outcome_time']]

    # Clients with Inforce product/account:
    client_inforce = purchased.loc[purchased['inforce'] == 1]
    client_inforce = pd.DataFrame(client_inforce['owner_ssn']).drop_duplicates()

    # Clients who purchased at least 1 product/account after min_date:
    client_recent = purchased.loc[purchased['issue_date'] >= min_date]
    client_recent = pd.DataFrame(client_recent['owner_ssn']).drop_duplicates()

    # Exclude clients who already have inforce target product:
    client_exclude = purchased.loc[(purchased['major_product'] == target_product) & (purchased['inforce'] == 1)]
    client_exclude = pd.DataFrame(client_exclude['owner_ssn']).drop_duplicates()

    # Apply all criteria
    purchased = purchased.merge(client_inforce, how='inner', on='owner_ssn')
    purchased = purchased.merge(client_recent, how='inner', on='owner_ssn')
    # Anti join to client_exclude:
    outer = purchased.merge(client_exclude, how='outer', on='owner_ssn', indicator=True)
    purchased = outer[(outer._merge == 'left_only')].drop('_merge', axis=1)

    # Outcome Variables ---------------------------------
    outcome = pur_hist.loc[(pur_hist['issue_date'] > pur_hist['snapshot_time']) & (pur_hist['issue_date'] <= pur_hist['outcome_time'])]
    outcome = outcome[['owner_ssn', 'major_product']].drop_duplicates()
    outcome['value'] = 1
    outcome = pd.pivot_table(outcome, index='owner_ssn', columns='major_product', values='value', aggfunc='sum', fill_value=0)

    outcome = pd.DataFrame(purchased['owner_ssn']).drop_duplicates().merge(outcome, on='owner_ssn', how='left')
    outcome.fillna(0, inplace=True)
    outcome[['dis_inc', 'fa', 'va', 'term', 'perm', 'ntl']] = outcome[['dis_inc', 'fa', 'va', 'term', 'perm', 'ntl']].astype(int)

    # Demographic Features ---------------------------------
    demo = pd.DataFrame(client[['owner_ssn', 'owner_dob', 'owner_gender', 'zip_income']])\
        .merge(pd.DataFrame(purchased[['owner_ssn', 'snapshot_time']]).drop_duplicates(), how='inner', on='owner_ssn')
    demo = demo[['owner_ssn', 'snapshot_time', 'owner_dob', 'owner_gender', 'zip_income']].drop_duplicates()
    demo['owner_dob'] = demo['owner_dob'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

    demo = demo.sort_values(by=['owner_ssn', 'owner_gender', 'zip_income'])
    demo.drop_duplicates(subset=["owner_ssn"], keep="first", inplace=True)
    demo['age'] = ((demo['snapshot_time'] - demo['owner_dob']) / dt.timedelta(days=365)).round(0).astype(int)
    demo = demo[['owner_ssn', 'snapshot_time', 'owner_gender', 'zip_income', 'age']]

    purchased['years'] = ((purchased['snapshot_time'] - purchased['issue_date']).dt.days / 365).round(1)
    client_years = purchased.groupby('owner_ssn').agg(
        client_years=('years', 'max'),
        recent_purchase=('years', 'min')
    ).reset_index()
    purchased = purchased[['owner_ssn', 'agreement_id', 'major_product', 'is_protect', 'inforce', 'issue_date',
                           'snapshot_time', 'outcome_time']]

    # Financial Planning client features:
    fp_client['first_plan'] = fp_client['first_plan'].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')
    fp_client_ind = demo[['owner_ssn', 'snapshot_time']].drop_duplicates().merge(fp_client, left_on='owner_ssn',
                                                                                 right_on='hashed_gov_id')
    fp_client_ind['fp_client'] = np.where(fp_client_ind['first_plan'] < fp_client_ind['snapshot_time'], 1, 0)

    demo = demo.merge(client_years, how='inner', on='owner_ssn').merge(fp_client_ind[['owner_ssn','fp_client']],
                                                                       how='inner', on='owner_ssn')

    # Life Policy Features ---------------------------------
    temp = purchased.loc[purchased['inforce'] == 1]
    temp = temp[['owner_ssn', 'agreement_id', 'issue_date']]

    term = temp.merge(termdata, how='inner', on='agreement_id')
    perm = temp.merge(permdata, how='inner', on='agreement_id')
    ntl = temp.merge(ntldata, how='inner', on='agreement_id')

    term = life_features(term, "term")
    perm = life_features(perm, "perm")
    ntl = life_features(ntl, "ntl")

    # Fixed Annuity Features ---------------------------------
    fa = temp.merge(fadata, how='inner', on='agreement_id')
    fa_max = fa.groupby('owner_ssn').agg({
        col: lambda column: column.max(skipna=True) for col in fa.columns if col.startswith("fa_prod")
    }).reset_index()
    fa_sum = fa.groupby('owner_ssn').agg({
        "fa_init_pay": 'sum'
    }).reset_index()

    fa = fa_max.merge(fa_sum, how='inner', on='owner_ssn')

    # Variable Annuity Features ---------------------------------
    va = temp.merge(vadata, how='inner', on='agreement_id')
    va_max = va.groupby('owner_ssn').agg({
        col: lambda column: column.max(skipna=True) for col in va.columns if (col.startswith("va_prod")) or
                                                                             (col.startswith("va_single_"))
    }).reset_index()
    va_sum = va.groupby('owner_ssn').agg({
        "va_init_pay": 'sum'
    }).reset_index()

    va = va_max.merge(va_sum, how='inner', on='owner_ssn')

    # Purchase Indicators -------------------------------------
    pur_ind = purchased.loc[purchased['inforce'] == 1]
    pur_ind = pur_ind[['owner_ssn', 'major_product']].drop_duplicates()
    pur_ind['value'] = 1
    pur_ind = pd.pivot_table(pur_ind, index='owner_ssn', columns='major_product', values='value', aggfunc='sum', fill_value=0)
    pur_ind.columns = [col + '_pur' for col in pur_ind.columns]
    pur_ind.reset_index(inplace=True)

    # Join Model Data -----------------------------------------
    model_data = outcome[['owner_ssn', target_product]].merge(demo, how='left').merge(pur_ind, how='left').\
        merge(term, how='left').merge(perm, how='left').merge(ntl, how='left').merge(fa, how='left').merge(va, how='left')

    # Additional pre-processing ------------------------------

    # Ignore clients with recent purchase
    model_data = model_data.loc[model_data['recent_purchase'] > 0.25]

    # Remove Long Term Care products
    model_data = model_data.loc[(model_data['perm_rider_ltcr'] == 0) | (model_data['perm_rider_ltcr'].isna())]

    model_data.drop(columns=['owner_ssn', 'owner_gender', 'va_pur', 'ntl_pur', 'fa_pur', 'perm_pur', 'term_pur',
                             'perm_rider_ltcr', 'snapshot_time'], inplace=True)

    # Combine product features
    vars_list = {
        "va_prod": ["va_prod_nonprop", "va_prod_prop"],
        "perm_prod": ["perm_prod_65", "perm_prod_hecv", "perm_prod_20_pay", "perm_prod_15_pay", "perm_prod_12_pay",
                      "perm_prod_10_pay", "perm_prod_100"],
        "term_prod": ["term_prod_art", "term_prod_10", "term_prod_15", "term_prod_20", "term_prod_25", "term_prod_30",
                      "term_prod_3or5"],
        "fa_prod": ["fa_prod_index", "fa_prod_odyssey", "fa_prod_retireease", "fa_prod_voyage"],
        "ntl_prod": ["ntl_prod_vl_nonprop", "ntl_prod_ul","ntl_prod_vl_prop"]
    }
    model_data = combine_product_all(vars_list, model_data)
    columns_to_remove = [column for sublist in vars_list.values() for column in sublist]
    model_data = model_data.drop(columns=columns_to_remove)

    # Rename the target variable
    model_data = model_data.rename(columns={target_product:'target'})

    # Convert indicator columns to integers ------------
    indicator_columns = ['target', 'term_rider_wp', 'term_prod_ecp',  'perm_rider_wp', 'perm_rider_gir',
                         'perm_rider_rtr1', 'perm_rider_alir', 'perm_rider_lisr', 'ntl_rider_wmc', 'ntl_rider_wsp',
                         'va_single_prem'] #'ntl_rider_dbr',
    model_data[indicator_columns] = model_data[indicator_columns].astype('Int64')

    # Convert character features to factors ------------
    model_data[model_data.select_dtypes('object').columns] = model_data[model_data.select_dtypes('object').columns].astype("category")

    return model_data





