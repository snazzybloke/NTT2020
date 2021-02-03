#!/usr/bin/env python3
# coding: utf-8

####################################################
# Author: Ante Bilic                               #
# Since: May 19, 2020                              #
# Copyright: The PLC Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

"""Summary:
   -------
   Collection of functions for IR Derivatives P/L attribution using SENSI method.
"""

import pathlib
import argparse
import re
import numpy as np
import pandas as pd
from dateutil.parser import parse
from FXO.common_futils_lcy import get_selection2, get_idxminmax2, human_format, sign_it,\
                                  get_idxminmax0, lst_unpack, highlight, highlight_ser, get_dtd_pl
from FXO.fxo_xl import pl_by_components
from FXO.enquire5 import *
from datetime import date
from openpyxl.styles import Font
from config.configs import getLogger
from config import configs
env = configs.env
logger = getLogger("irvega_xl.py")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.mode.chained_assignment = None


def get_sensi(ddate, files, entity="DBSSG", node_id="IRD"):
    """
    Summary:
    Reads the SENSI file in the format (i) obtained from the SQL query, named e.g.,
    20200319_DBSSG_IRD.csv.gz, or (ii) PLSensi_GFM...date.csv from iWork dashboard,
    named e.g. PLSensi_GFM_19-03-2020_IRD.csv.gz

    Parameters:
    -----------
    ddate: str          the business date as a string (e.g., '20190412')
    files: str          the path to the folder with the input files
    entity:  str        the trading entity (e.g., 'DBSSG', 'DBSHK', 'DBSSYD', etc)
    node_id: str        the trading desk (e.g., 'FX Options', 'CREDIT TRADING')

    Return:
    --------
    pf_sensi: DataFrame        the relevant/filtered rows (and columns) from the *SENSI*TXT file
    """
    sensi_file = files + ddate + '_' + entity + '_' + node_id + '.csv.gz'
    try:
        ucols = ['Sensi Date', 'Node', 'Entity', 'Component', 'Product Type', 'PL Currency',
                 'Val Group', 'Instrument', 'LCY', 'Raw Component', 'Product Group', 'Issuer Curve',
                 'Issuer Name', 'Seniority', 'Underlying', 'Yield Curve', 'Tenor',
                 'Underlying Maturity', 'Quotation', 'Multiplier', 'Type', 'PL Explain', 'PL Sensi',
                 'Strike', 'Mkt (T)', 'Mkt (T-1)', 'Sensi Risk (OCY)', 'DTD LCY', 'DTD SGD']
        pf_sensi = pd.read_csv(sensi_file, low_memory=False, usecols=ucols)
        pf_sensi = pf_sensi.assign(Rate_move=(pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"]))
    except FileNotFoundError:
        ddt = parse(ddate)
        # now take only the date part, make a string, split it, reverse the list, and join:
        newdate = '-'.join(str(ddt.date()).split('-')[::-1])
        sensi_file = files + 'PLSensi_GFM_' + newdate + '_' + node_id + '.csv.gz'
        ucols = ['Date', 'Node', 'Entity', 'Component', 'Product Type', 'Currency',
                 'ValGroup', 'Instrument', 'LCY', 'Raw Component', 'Product Group', 'Issuer Curve',
                 'Issuer Name', 'Seniority', 'Underlying', 'Yield Curve', 'Tenor',
                 'Underlying Maturity', 'Quotation', 'Multiplier', 'Type', 'PL Explain', 'PL Sensi',
                 'Strike', 'Mkt (T)', 'Mkt (T-1)', 'Sensi Risk (OCY)', 'DTD LCY', 'DTD SGD']
        pf_sensi = pd.read_csv(sensi_file, low_memory=False, usecols=ucols)
        rn_dict = {'Currency': 'PL Currency', 'ValGroup': 'Val Group', 'Date': 'Sensi Date'}
        pf_sensi.rename(rn_dict, axis=1, inplace=True)
        pf_sensi = pf_sensi.assign(Rate_move=(pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"]) * 100)

    pf_sensi = pf_sensi.drop(columns=['Entity', 'Product Group', 'Issuer Curve',
                                      'Issuer Name', 'Seniority', 'Multiplier',
                                      'PL Explain', 'PL Sensi'])
    return pf_sensi


def get_tenors_ordered(pf_, t_col='Tenor'):
    """
    Summary:
    --------
    Rank/order the standard Tenors (i.e., not the maturity dates) by their natural order
    according to the list below, and generate the new column 'ord' to the SENSI table,
    which is simply the index of a tenor in this list.

    Parameters:
    -----------
    pf_ (DataFrame):  a SENSI table (or a pre-filtered part of it), with 'Tenor' column

    Returns
    -----------
    pf_ (DataFrame):  the SENSI table with Tenors rank in the new 'ord' column
    """

    tenor_list = ['O/N', 'T/N', 'S/N', '1W', '2W',
                  '1M', '2M', '3M', '1M/4M', '4M', '2M/5M', '5M', '3M/6M', '6M',
                  '7M', '8M', '6M/9M', '9M', '7M/10M', '10M', '8M/11M', '11M', '9M/12M',
                  '1Y', '1Y3M', '1Y6M', '1Y9M', '2Y', '2Y3M', '2Y6M', '2Y9M',
                  '3Y', '3Y3M', '3Y6M', '3Y9M', '4Y', '4Y3M', '4Y6M', '4Y9M',
                  '5Y', '5Y3M', '5Y6M', '5Y9M', '6Y', '6Y3M', '6Y6M', '6Y9M',
                  '7Y', '7Y3M', '7Y6M', '7Y9M', '8Y', '8Y3M', '8Y6M', '8Y9M',
                  '9Y', '9Y3M', '9Y6M', '9Y9M', '10Y', '10Y3M', '10Y6M', '10Y9M',
                  '11Y', '11Y3M', '11Y6M', '11Y9M', '12Y', '12Y3M', '12Y6M', '12Y9M',
                  '13Y', '13Y3M', '13Y6M', '13Y9M', '14Y', '14Y3M', '14Y6M', '14Y9M',
                  '15Y', '15Y3M', '15Y6M', '15Y9M', '16Y', '16Y3M', '16Y6M', '16Y9M',
                  '17Y', '17Y3M', '17Y6M', '17Y9M', '18Y', '18Y3M', '18Y6M', '18Y9M',
                  '19Y', '19Y3M', '19Y6M', '19Y9M', '20Y', '20Y3M', '20Y6M', '20Y9M',
                  '21Y', '21Y3M', '21Y6M', '21Y9M', '22Y', '22Y3M', '22Y6M', '22Y9M',
                  '23Y', '23Y3M', '23Y6M', '23Y9M', '24Y', '24Y3M', '24Y6M', '24Y9M',
                  '25Y', '25Y3M', '25Y6M', '25Y9M', '26Y', '26Y3M', '26Y6M', '26Y9M',
                  '27Y', '27Y3M', '27Y6M', '27Y9M', '28Y', '28Y3M', '28Y6M', '28Y9M',
                  '29Y', '29Y3M', '29Y6M', '29Y9M', '30Y', '35Y', '40Y', '50Y']

    # pf_sensi[t_col].replace({'nan': ''}, inplace=True)
    pf_[t_col].replace({'12M': '1Y'}, inplace=True)
    pf_[t_col].replace({'18M': '1Y6M'}, inplace=True)
    pf_[t_col].replace({str(i) + 'WK': str(i) + 'W' for i in range(1, 10)}, inplace=True)
    get_tenor_idx = lambda x: tenor_list.index(x) + 1 if x in tenor_list else 0
    pf_['ord'] = pf_[t_col].map(get_tenor_idx).where(cond=pf_[t_col].notna(), other=0)


def get_risk(at_risk):
    """
    Summary:
    -----------
    Returns risk postion, long/short depending on the sign of the amount at risk (at_risk).
    """
    if at_risk > 0:
        return 'long'
    elif at_risk < 0:
        return 'short'


def capswp_comment(d_list):
    """
    Summary:
    --------
    Inserts the values provided in the d_list into the Caplet/Swaption commentary template.
    Once the commentary is complete, it is added as the final value to d_list, which
    will be converted into a dictionary and appended to the df_results DataFrame.

    Parameters:
    -----------
    d_list (list):  the values associated with dcol column values
                    of df_results

    Returns
    -----------
    d_list (list):  the list is updated with the commentary (the final value)
    """
    # Need to establish the correct number of decimal places j for the format.
    # The $ PnL is provided in d_list[2]. Set the criterion as a function of j:
    dtd_tot = d_list[2] + d_list[4]
    brit1 = lambda j: round(dtd_tot, j) == 0
    j = 0
    while brit1(j):
        j += 1
    comment = human_format(dtd_tot, j) if j > 0 else human_format(dtd_tot)
    comment += f" {d_list[0]} IR {d_list[1]} Vol"
    # Need to establish the correct number of decimal places i for the rate min/max formats.
    # The rates min & max are provided at indices 8 and 9 of d_list.
    # Set the 3 criteria as functions of i (# of decimal places):
    # (1) The larger and smaller rate appear equal with this i (# of decimal places):
    crit1 = lambda i: round(min(d_list[8:]), i) == round(max(d_list[8:]), i)
    # (2) The smaller rate appears equal to zero with this i:
    crit2 = lambda i: round(min(d_list[8:]), i) == 0
    # (2) The larger rate appears equal to zero with this i:
    crit3 = lambda i: round(max(d_list[8:]), i) == 0
    # If either of the 2 Rates is zero, get i (# of decimal places) from the one that isn't:
    if any(x==0 for x in d_list[8:]):
        i = abs(np.log10(max(abs(d_list[8]), abs(d_list[9]))))
        i = int(max(np.ceil(i), np.floor(i)))
    # Otherwise keep looping over i (# of decimal places) until ALL 3 criteria un-satisfied:
    else:
        i = 1
        while crit1(i) or crit2(i) or crit3(i):
            i += 1
            if i >= 6:
                break
    # the overall Rate_move is based on the signs of $ PnL and $ At Risk:
    if (d_list[2] > 0) & (d_list[3] > 0):
        comment += f" increased by up to {abs(d_list[9]):.{i}f} vol points"
    elif (d_list[2] < 0) & (d_list[3] < 0):
        comment += f" increased by up to {abs(d_list[9]):.{i}f} vol points"
    elif (d_list[2] < 0) & ( d_list[3] > 0):
        comment += f" decreased by up to {abs(d_list[8]):.{i}f} vol points"
    elif (d_list[2] > 0) & (d_list[3] < 0):
        comment += f" decreased by up to {abs(d_list[8]):.{i}f} vol points"
    # the Tenors
    comment += f" on tenors {d_list[5]}"
    comment += f" on underlying maturity {d_list[7]}" if d_list[7] else ""
    what_risk = get_risk(d_list[3])
    comment += f" on {what_risk} positions" if what_risk else ""
    d_list.append(comment)


def impute_pfolio(c1u, cy_vega):
    """
    Summary:
    --------
    For the currency ccy and its pivot table c1u, split on "Underlying" values, the
    portfolio origin (either MUREX or Minisys) is evaluated based on the associated
    "Product Type" values and stored in a new 'p_folio' column.

    Parameters:
    -----------
    c1u (DataFrame):      a pivot table of cy_vega split on "Underlying" column values
    cy_vega (DataFrame):  a SENSI table after Component selection ("IR Vega") and
                          currency ccy selection.
    Returns
    -----------
    c1u (DataFrame):      the pivot table with portfolio in the new 'p_folio' column
    """
    # The Underlying values coming from MUREX or Minisys portfolios?
    # Adding the new column 'p_folio':
    c1u.insert(c1u.shape[1], 'p_folio', None)
    # The "standard" Product Types, coming from MUREX:
    standard_pt = {'Interest rate swaps', 'Swaptions', 'Caps/floors'}
    # the "non-standard" Product Types, coming from Minisys:
    nonstandard_pt = {"Interest rate future options", "FXNote", "FXTimeOption",
                      "QuantoCMS CdsKO", "QuantoMulti", "QuantoCMS CdsKO", "QuantoMulti",
                      "FXTimeOption", "CMS Callable", "CMSCallable CdsKO", "FXNote",
                      "FXTimeOption", "Hybrid 12",  "Hybrid 19 Smile", "QuantoCMS CdsKO",
                      "QuantoMulti", 'ZeroCouponIRS', 'CMS Callable', 'CMSCallable CdsKO',
                      "QuantoCMS CdsKO", "QuantoMulti", "Hybrid"}
    for u_ing in c1u.index:
        u_ing_pts = set(cy_vega.loc[cy_vega['Underlying'] == u_ing, 'Product Type'])
        # if no intersection with standard PT, this Underlying must be from Minisys:
        if not standard_pt.intersection(u_ing_pts):
            c1u.loc[u_ing, 'p_folio'] = 'Minisys'
        # if no intersection with non-standard PT, this Underlying must be from MUREX:
        elif not nonstandard_pt.intersection(u_ing_pts):
            c1u.loc[u_ing, 'p_folio'] = 'MUREX'
        # having intersection with both standard and non-standard PT, hence MUREX & Minisys:
        else:
            c1u.loc[u_ing, 'p_folio'] = 'MUREX/Minisys'
        # the 'Total' has no PT, rectify the assignement:
        c1u.loc['Total', 'p_folio'] = None


def swapcap_tot(c1u, cu2pt, c3urc, cy_vega):
    """
    Summary:
    --------
    Calculating the partial $ contributions from all Caplets and all Swaptions.
    The MUREX Underlying item(s) typically have both Caplets & Options, revealed
    by the cu2pt pivot table. The Minisys Caplets & Swaptions are evident from c1u and c3urc.
    Combine MUREX Swaptions with the Minisys Swaptions in the 1st pivot table c1u
    and MUREX Caps/floors with Minisys Caplets in the 2nd pivot table cu2pt.
    Similarly, combine MUREX Caplets with the Minisys Caplets in the 2nd pivot table cu2pt
    and MUREX Caps/floors with Minisys Caplets in the 3rd pivot table c3urc.

    Parameters:
    -----------
    c1u (DataFrame):      a pivot table of cy_vega split on "Underlying" column values
    cu2pt (DataFrame):    a pivot table of the MUREX part(s) in c1u, split on Product Type values
    c3urc (DataFrame):    a pivot table of cy_vega split on "Underlying" & "Raw Component" values
    cy_vega (DataFrame):  a SENSI table after Component selection ("IR Vega") and
                          currency ccy selection.
    Returns
    -----------
    tot_list (list):      listing of "Caplet", "Swaption" contributions, useful for MUREX p'folio
    """
    # Combine MUREX Swaptions with the Minisys Swaptions in the 1st pivot table
    # and MUREX Caps/floors with Minisys Caplets in the 3rd pivot table.
    # First, the Swaptions:
    try:
        tot_swaptions = cu2pt.loc['Swaptions', 'DTD LCY']
        tot_list = ["Swaptions"]
    except KeyError:
        tot_swaptions = 0
        tot_list = []
    # Look up the 1st pivot table for its PnL and add it up:
    # Extract the Swaption row (ignore case):
    tmp = c1u.filter(regex=re.compile('swaption', re.I), axis=0)
    # then add them up:
    try:
        tot_swaptions += tmp['DTD LCY'].iloc[0]
    except IndexError:
        pass
    # Second, the Caps/floors from the MUREX Underlying:
    try:
        tot_capfloors = cu2pt.loc['Caps/floors', 'DTD LCY']
        tot_list += ["Caps/floors"]
    except KeyError:
        tot_capfloors = 0
        tot_list += []
    # Look up the 3rd pivot table for its PnL and add it up:
    # Extract the Caplets row (ignore case):
    tmp = c3urc.filter(regex=re.compile('caplet', re.I), axis=0)
    # get its Underlying value from the Multi-index:
    try:
        tmp_u = tmp.index[0][0]
        # It may also have 'IR Vega' Raw Component, so check its Underlying Maturity values
        # to decide if it is a Caplet (nan) or Swaption (with regular values, e.g. '1Y' etc):
        if cy_vega.loc[cy_vega['Underlying'] == tmp_u,
                       'Underlying Maturity'].unique().tolist() == [np.nan]:
            # IR Vega is a Caplet, add the WHOLE tmp_u contribution from the 1st pivot table:
            tot_capfloors += c1u.loc[tmp_u, 'DTD LCY']
        else:
            # IR Vega is a Swaption, only add the Caplet part, not the 'IR Vega' part:
            tot_capfloors += tmp.loc[tmp_u, 'DTD LCY']
            # The 'IR Vega' contribution then should be added to tot_swaptions? (TO-DO...)
    except IndexError:
        pass
    return tot_list


def irvega_type(c1u, c3urc, cy_vega, murund, tot_list, xclwrt, wsheet):
    """
    Summary:
    --------
    Prints the type (Caplet vs Swaption vs ...) first for the MUREX Underlying
    (typically missing/'blank') IR Vega, which is usually a combination of Caplets & Swaptions.
    This is followed by the IR Vega type of the other Underlying items (usually from Minisys)

    Parameters:
    -----------
    c1u (DataFrame):      a pivot table of cy_vega split on "Underlying" column values
    c3urc (DataFrame):    a pivot table of cy_vega split on "Underlying" & "Raw Component" values
    cy_vega (DataFrame):  a SENSI table after Component selection ("IR Vega") and
    murund (str):         the MUREX Underlying (typically 'blank', i.e. no Underlying values)
    tot_list (list):      the type(s) of murund IR Vega (usually both ['Caps/floors', 'Swaptions'])
    xclwrt (XlsWriter):   an Excel Writer object
    wsheet (Worksheet):   the worksheet for the selected Currency
    """
    # Extract the MUREX Raw Component from the Multi-index:
    try:
        murund_rc = c3urc.loc[murund, 'DTD LCY'].index[0]
        print((murund, murund_rc), ': ', c1u.loc[murund, 'p_folio'], tot_list)
        tmp_dic = {'Underlying': [f"{murund}"],
                   'Raw Component': [f"{murund_rc}"],
                   'p_folio': [f"{c1u.loc[murund, 'p_folio']}"],
                   'type(s)': [f"{tot_list}"]}
    except IndexError:
        tmp_dic = {'Underlying': [],
                   'Raw Component': [],
                   'p_folio': [],
                   'type(s)': []}
    # What about other IR Vegas from c2urc? Inspect their Underlying Maturity:
    othervgs = c3urc.index.sortlevel(0, ascending=True)[0].drop(murund)
    irv_idx = [(uu, 'IR Vega') for uu, rc in othervgs if rc == 'IR Vega']
    for i, uu_irv in enumerate(irv_idx, 1):
        print(uu_irv, ': ', end='')
        tmp_dic['Underlying'].append(uu_irv[0])
        tmp_dic['Raw Component'].append('IR Vega')
        tmp_dic['p_folio'].append(c1u.loc[uu_irv[0], 'p_folio'])
        if cy_vega.loc[cy_vega['Underlying'] == uu_irv[0],
                       'Product Type'].unique().tolist() == ["Interest rate future options"]:
            print(c1u.loc[uu_irv[0], 'p_folio'], "-")
            tmp_dic['type(s)'].append('')
        elif cy_vega.loc[cy_vega['Underlying'] == uu_irv[0],
                       'Underlying Maturity'].unique().tolist() == [np.nan]:
            print(c1u.loc[uu_irv[0], 'p_folio'], "Caplet")
            tmp_dic['type(s)'].append('Caplet')
        else:
            print(c1u.loc[uu_irv[0], 'p_folio'], "Swaption")
            tmp_dic['type(s)'].append('Swaption')
    print()
    wsheet["K15"] = "Assigning types to IR Vega:"
    pd.DataFrame(tmp_dic).\
            to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=15, startcol=10)


def rank_umatur(cy_vega):
    """
    Summary:
    --------
    Rank/order the Underlying Maturity by their natural order
    according the number before 'Y', akin to the Tenors, and generate
    the new column 'um_ord' to the SENSI table,

    Parameters:
    -----------
    cy_vega (DataFrame):  a SENSI table after Component selection ("IR Vega")

    Returns
    -----------
    cy_vega (DataFrame):  the SENSI table with Underlying Maturity rank in the new 'um_ord' column
    """
    pattern = "([0-9]+)([A-Za-z]+)"
    # pandas does not recognise empty string as Nan. Replace them:
    cy_vega['Underlying Maturity'].replace('', np.nan, inplace=True)
    cy_vega['Underlying Maturity'].replace('-', np.nan, inplace=True)
    # sort using the number that appears before 'Y', akin to the Tenors:
    um_list = sorted(cy_vega['Underlying Maturity'].dropna().unique().tolist(),
                     key=lambda x: int(re.match(pattern, x).groups()[0]))
    # insert the missing/blank at the start of the list:
    um_list.insert(0, 'blank')
    # Ranking 'um_ord' is provided by the index in the sorted list:
    get_um_idx = lambda x: um_list.index(x) + 1 if x in um_list else 0
    cy_vega['um_ord'] = cy_vega['Underlying Maturity'].map(get_um_idx).\
                        where(cond=cy_vega['Underlying Maturity'].notna(), other=0)


def get_strikes(pf_):
    """
    Summary:
    --------
    Extracts the Strike range contributing to 90-110% of the PnL.
    The standard procedure does it: evaluate the pivot table split across the Strike values,
    aggregating the associated $ PnL subtotals; add the 'frac' column showing the fractional
    contributions of the subtotals to the grand Total $PnL; then select the contiguous range
    of Strike values making up the key contributions to it.

    Parameters:
    -----------
    pf_ (DataFrame):  a SENSI table after Component ("IR Vega") and Ccurrency selection.

    Returns
    -----------
    ... (list):       the selected range of values from the Strike column
    """
    pt_ = pf_.pivot_table(index='Strike',
                          fill_value=0,
                          dropna=False,
                          values='DTD LCY',
                          aggfunc=np.sum,
                          margins=True,
                          margins_name='Total')
    tot_pl = pt_.loc['Total', 'DTD LCY']
    pt_.drop('Total', axis=0, inplace=True)
    pt_.reset_index(inplace=True)
    pt_ = pt_.assign(frac = pt_['DTD LCY'] / tot_pl)
    # to select sequentially: pt_ = pt_.sort_values('Strike', ascending=True).reset_index(drop=True)
    # pt_ = pt_.sort_values('frac', ascending=False).reset_index(drop=True)
    pt_.sort_values('frac', inplace=True, ignore_index=True, ascending=False)
    # get the number of rows with a negative fraction/percentage:
    i_roll = pt_.query("frac < 0").shape[0]
    pt_rolled = pd.DataFrame(columns=pt_.columns, data=np.roll(pt_.values, i_roll, axis=0))
    pt_rolled['frac'] = pd.to_numeric(pt_rolled['frac'])
    select = get_idxminmax2(pt_rolled, 'Strike', xrac='frac', min_xrac=0.9, max_xrac=1.1)
    return select.tolist()


def work_2dflipped(pfs, tots, strikes, xclwrt, wsheet):
    """
    Summary:
    --------
    The purpose of this function is simply to reduce the cognitive complexity
    (evaluated by Sonar) of the get_2dflipped() function below. This function
    simply continues the processing started by get_2dflipped() which calls it.
    When finished, it returns the 8 output values to it. Essentially, here it
    extracts the sflip range contributing to 90-110% of the PnL for each Strike value.
    A single sflip value comprises a range of Tenors with the same $ PnL sign.
    So, we extract the contiguous groups of Tenor contrubutions to the 90-110%
    of the Strike's $ PnL subtotal.

    Parameters:
    -----------
    pfs (list):          a list of SENSI pivot tables, one for each Strike value
    tots (list):         a list of the Total $ PnL, one for each Strike value
    strikes (list):      the list of the Strike values
    xclwrt (XlsWriter):  an Excel Writer object
    wsheet (Worksheet):  the worksheet for the selected Currency

    Returns
    -----------
    ... (9-tuple): the min & max Tenor rank, 2 empty strings (no Underlying Maturity for captions),
                   the min and max Rate_move, the $ total PnL and $ at risk from Caplets.
                   These are the value used to fill out the df_results table, Caplets row.
                   The last member (mini_dtd) is the selected fraction of Minisys Caps/floors.
    """
    # Initialize the appropriate number of new pivot tables, one for each Strike value...
    pt2 = [pd.DataFrame() for _ in range(len(strikes))]
    # ... and the associated $ PnL subtotals, both as lists:
    pt2tot = [0 for _ in range(len(strikes))]
    # Fill the lists with actual tables and their $ subtotals:
    for i in range(len(strikes)):
        if tots[i] == 0:
            continue
        # split the pfs[i] tables across their sign-flip values and aggregate $ PnL and $ at risk:
        pt2[i] = pfs[i].pivot_table(index='sflip',
                                    values=['DTD LCY', 'Sensi Risk (OCY)'],
                                    aggfunc=np.sum,
                                    margins=True,
                                    margins_name='Total')
        pt2tot[i] = pt2[i].loc['Total', 'DTD LCY']
        pt2[i].drop('Total', axis=0, inplace=True)
        pt2[i].reset_index(inplace=True)
    # Initialize 6 (out of 8) output values (the other 2 will be simply '') with dummy values:
    ord_min = 1e6
    ord_max = -1e6
    rate_min = 1.0e6
    rate_max = -1.0e6
    pnl = 0
    at_risk = 0
    mini_dtd = 0.0
    # Now extract the sflip contiguous ranges contributing to 90-110% of the Strike $ subtotals:
    i_row = 3
    icol = 20
    rn_dic = {"sflip": "Sign-flip Category"}
    bold_font = Font(bold=True, size=12)
    for i in range(len(strikes)):
        # skip the empty tables:
        if pt2[i].empty:
            continue
        # the fractional contributions:
        pt2[i]= pt2[i].assign(frac=pt2[i]['DTD LCY'] / pt2tot[i])
        # For a single-row table, simply extract its sflip value:
        if pt2[i].shape[0] == 1:
            sel_idx = pt2[i]['sflip'].astype('int').tolist()
        # For others use the get_idxminmax0, initially devised for Tenor selection:
        else:
            sel_idx = get_idxminmax0(pt2[i], 'sflip', min_xrac=0.90, max_xrac=1.10).\
                        astype('int').tolist()
        print(f"\nSELECTED Strike: {strikes[i]}\nsflips:\n{sel_idx}")
        df_tmp = pfs[i].query('sflip in @sel_idx')
        pfs[i].drop(columns=['Strike', 'ord'], inplace=True)
        pfs[i] = pfs[i].assign(Selected=["Yes" if s in sel_idx else "No" for s in pfs[i]['sflip']])
        print(f"\n{df_tmp}")
        # dtenors = df_tmp['Tenor'].unique().tolist()  # can pass these to highlight() below
        # to evaluate the selected fraction of Minisys Caplets:
        if np.isinf(strikes[i]):
            mini_dtd = pt2[i].loc[pt2[i]['sflip'].isin(sel_idx), 'frac'].sum()
        #@ df_tmp.to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=i_row, startcol=icol)
        
        pfs[i].rename(rn_dic, axis=1).style.applymap(highlight(lst=["Yes"])).\
                to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=i_row, startcol=icol)
        dstring = lst_unpack("Selected Strike:", [strikes[i]], ',')\
                            .replace('-inf', 'Missing MUREX')\
                            .replace('inf', 'Mini')
        dstring += lst_unpack(" sign-flip:", sel_idx)
        wsheet["U" + str(i_row)] = dstring
        wsheet["U" + str(i_row)].font = bold_font
        #@ i_row += (df_tmp.shape[0] + 2)
        i_row += (pfs[i].shape[0] + 2)
        # Update the 4 output values with better values, if found:
        ord_min = min(ord_min, df_tmp['ord'].min())
        ord_max = max(ord_max, df_tmp['ord'].max())
        rate_min = min(rate_min, df_tmp['Rate_move'].min())
        rate_max = max(rate_max, df_tmp['Rate_move'].max())
        # Aggregate the $ output values:
        pnl += df_tmp['DTD LCY'].sum()
        at_risk += df_tmp['Sensi Risk (OCY)'].sum()
    return ord_min, ord_max, '', '', rate_min, rate_max, pnl, at_risk, mini_dtd


def get_2dflipped(pf_, strikes, xclwrt, wsheet):
    """
    Summary:
    --------
    This 2D-version (for Caplets, i.e., using Strike-Tenor variables) is designed
    for selecting the contiguous range of Tenor groups with the same $ PnL sign.
    To this end a new column 'sflip' is introduced which gets values e.g. -1, 0, 1, 2,...
    Each new value flags indicates that the sign of $ PnL subtotal has changed from this Tenor on.
    Hence the number of 'sflip' values is the number of sign changes.
    Previously, individual adjacent Tenors were selected, and here Tenor GROUPS (with a common
    sflip value) are being selected. The processing continues in the work_2dflipped()
    above to select the 6 output values for the Caplet row df_results (plus two
    empty strings '' for Underlying Maturity, which don't exist for Caplets).
    When finished, it returns the 8 output values to it. Essentially, here it
    extracts the sflip range contributing to 90-110% of the PnL for each Strike value.
    A single sflip value comprises a range of Tenors with the same $ PnL sign.
    So, we extract the contiguous groups of Tenor contrubuting to the 90-110%
    of the Strike's $ PnL subtotal.

    Parameters:
    -----------
    pf_ (DataFrame):      SENSI table for Caplets (MUREX + Minisys)
    strikes (list):       the list of the Strike values
    xclwrt (XlsWriter):   an Excel Writer object
    wsheet (Worksheet):   the worksheet for the selected Currency

    Returns
    -----------
    ... (8-tuple): the min & max Tenor rank, 2 empty strings (no Underlying Maturity for captions),
                   the min and max Rate_move, the $ total PnL and $ at risk from Caplets.
                   These are the value used to fill out the df_results table, Caplets row.
    """
    # Initialize the lists of pivot_tables and their $ PnL subtotals (one index for each Strike value):
    pfs = [pd.DataFrame() for _ in range(len(strikes))]
    tots = [0 for _ in range(len(strikes))]
    # The 2Ds are Strike & Tenor (with their Rate_move):
    idx_l = ['Strike', 'ord', 'Tenor', 'Rate_move']
    tot_tup = ('Total', '', '', '')
    # Fill  the actual pivot tables and their $ PnL subtotals:
    for i in range(len(strikes)):
        si = strikes[i]
        pfs[i] = pf_.query("Strike == @si").\
                pivot_table(index=idx_l,
                            values=['DTD LCY', 'Sensi Risk (OCY)'],
                            aggfunc=np.sum,
                            margins=True,
                            margins_name='Total').sort_values(['Strike', 'ord'])
        tots[i] = pfs[i].loc[tot_tup, 'DTD LCY']
        # skip the zero $PnL Strikes:
        if tots[i] == 0:
            continue
        print(f"Total for Strike {si}: {tots[i]}")
        pfs[i].drop('Total', axis=0, inplace=True)
        # A Series with True value if the row's $ PnL is zero, False otherwise:
        zero_dtd = (pfs[i]['DTD LCY'] == 0)
        # Drop the rows with True (i.e., zero $ PnL) values:
        pfs[i].drop(zero_dtd.index[zero_dtd], axis=0, inplace=True)
        # For the rest of the rows initialize the 'sflip' according to its $ PnL sign:
        pfs[i].reset_index(inplace=True)
        pfs[i] = pfs[i].assign(sflip=np.sign(pfs[i]['DTD LCY']))

    # Now, the actual 'sflip' values are being assigned:
    for i in range(len(strikes)):
        # skip this Strike if zero $ PnL:
        if tots[i] == 0:
            continue
        # Evaluate the difference as either 0 or 1 between this and previous row's sflip:
        pfs[i].loc[1:, 'sflip'] = abs(pfs[i]['sflip'].diff(1) / 2)
        # Now all the rows have sflip either 0 or 1 (except perhaps the 0 row with -1 if $ PnL< 0).
        # If the current row's sflip is 0, then it has the same cumsum(sflip) as the previous row.
        # If the current row's sflip is 1, then its cumsum(sflip) will increase by 1:
        pfs[i]['sflip'] = pfs[i]['sflip'].cumsum()
    results_8tuple = work_2dflipped(pfs, tots, strikes, xclwrt, wsheet)
    return results_8tuple


def tenor_flipped(pfs_mini, tots_mini, tenors_mini):
    """
    Summary:
    --------
    This is the part of Swaption processing that concerns only the Minisys Swaptions.
    The purpose of this function is simply to reduce the cognitive complexity
    (evaluated by Sonar) of the get_3dflipped() function below. This function
    assigns a different 'tflip' value for each Tenor's group of Underlying Maturity
    values with the common $ PnL sign. The group of Tenors that make up the 90-110% range
    of the $ PnL is is selected and subsequently returned to get_3dflipped().

    Parameters:
    -----------
    pfs_mini (list):     a list of Minisys pivot tables, one for each Tenor value
    tots_mini (list):    a list of the Total $ PnL, one for each Tenor value
    tenors_mini (list):  the list of the Tenor values

    Returns
    -----------
    tmp_pf (list):           the updated substitute for pfs_mini 
    tmp_tot (list):          the updated substitute for tots_mini
    miniswap_tenors (list):  the updated substitute for tenors_mini
    """
    # Don't consider Tenor=='blank', i.e., ord==0:
    j_start = 2 if 0 in tenors_mini else 1
    # Copy the initial tflip values into a new temporary column "t_col":
    for j in range(j_start - 1, len(tenors_mini)):
        pfs_mini[j] = pfs_mini[j].assign(t_col= pfs_mini[j]['tflip'])
    for j in range(j_start, len(tenors_mini)):
        # Skip the Tenors with zero $ PnL:
        if tots_mini[j] == 0:
            continue
        # Unlike in the sflip case, we cannot use diff(1) and cumsum()
        # because the rows are not adjacent, the Tenors are.
        # The difference between the current Tenor's tflip and that of the previous Tenor:
        tdiff = abs(pfs_mini[j]['t_col'].iloc[0] - pfs_mini[j-1]['t_col'].iloc[0])
        # Update the tflip by adding a half of this value (i.e, either by adding 0 or 1):
        pfs_mini[j].loc[:, 'tflip'] = pfs_mini[j-1]['tflip'].iloc[0] + tdiff / 2
    # Drop the temporary t_col column. Starting with an empty table, join the tables together:
    miniswap = pd.DataFrame(columns = pfs_mini[j_start - 1].columns[:-1])
    for j in range(j_start - 1, len(tenors_mini)):
        pfs_mini[j].drop(columns='t_col', inplace=True)
        miniswap = miniswap.append(pfs_mini[j])
    # Split the combined table across tflip values and evaluate their the $ PnL and Risk subtotals:
    pt_miniswap = miniswap.pivot_table(index='tflip',
                                       values=['DTD LCY', 'Sensi Risk (OCY)'],
                                       aggfunc=np.sum,
                                       margins=True,
                                       margins_name='Total')
    miniswp_tot = pt_miniswap.loc['Total', 'DTD LCY']
    pt_miniswap.drop('Total', axis=0, inplace=True)
    pt_miniswap.reset_index(inplace=True)
    # Fractional PnL contributions:
    pt_miniswap = pt_miniswap.assign(frac=pt_miniswap['DTD LCY'] / miniswp_tot)
    # Get the contiguous tflip selection making up 90-110% of the Total:
    tflip_lst = get_idxminmax0(pt_miniswap, 'tflip', min_xrac=0.90, max_xrac=1.10).tolist()
    # Selected Minisys Tenors (or, rather, their ranks):
    miniswap_tenors = miniswap.loc[miniswap['tflip'].isin(tflip_lst), 'ord'].unique().tolist()
    # Weed out those not selected:
    tmp_pf = []
    tmp_tot = []
    for j in range(len(tenors_mini)):
        # The Tenor ord (rank) of the index j:
        t = tenors_mini[j]
        if t in miniswap_tenors:
            tmp_pf.append(pfs_mini[j])
            tmp_tot.append(tots_mini[j])
    # Return the updated the Minisys Swaption Tenor ranks, their subtotals and pivot tables:
    return tmp_pf, tmp_tot, miniswap_tenors


def extra_3dflipped(pfs, tenors, strikes, pt2, pt2tot, xclwrt, wsheet):
    """
    Summary:
    --------
    The purpose of this function is simply to reduce the cognitive complexity
    (evaluated by Sonar) of the work_3dflipped() function below. This function
    simply continues the processing started by get_3dflipped(), followed by
    work_3dflipped(), which calls it. When finished, it returns the 8 output values to it.
    It extracts the sflip range contributing to 90-110% of the PnL for each Strike value.
    A single sflip value comprises a range of Tenors with the same $ PnL sign.
    So, we extract the contiguous groups of Tenor contrubutions to the 90-110%
    of the Strike's $ PnL subtotal. Additionally, for each Tenor value the tflip
    values show the Underlying Maturity values with the common sign of the $ PnL
    subtotal. So, here we perform a "NESTED SELECTION": based on sflip a group
    of adjacent Tenors is chosen and, based on tflip a group of Underlying
    Maturities is selected.

    Parameters:
    -----------
    pfs (list):     a list of SENSI pivot tables, one for each Strike value
    tenors (list):  a list of the lists of Tenors, one for each Strike value
    strikes (list): the list of the Strike values
    pt2 (list):     a list of lists of SENSI pivot tables, one for each Strike/Tenor values
    pt2tot (list):  a list of list of the Total $ PnL, one for each Strike/Tenor values

    Returns
    -----------
    ... (9-tuple): the min & max Tenor rank, the Underlying Maturity min & max rank,
                   the min and max Rate_move, the $ total PnL and $ at risk from Swaptions.
                   These are the value used to fill out the df_results table, Swaptions row.
                   The last member (mini_dtd) is the selected fraction of Minisys Swaptions.
    """
    # Initialize 8 output values with dummy values:
    ord_min = 1e6
    ord_max = -1e6
    um_min = 1e6
    um_max = -1e6
    rate_min = 1e6
    rate_max = -1e6
    pnl = 0
    at_risk = 0
    # to evaluate the selected fraction of Minisys swaptions:
    mini_dtd = 0.0
    # mini_swp_tot = sum(t_j for t_j in pt2tot[-1])
    # Now extract the sflip contiguous ranges contributing to 90-110% of the Strike $ subtotals:
    i_row = 3
    i_col = 32
    bold_font = Font(bold=True, size=12)
    rn_dic = {"sflip": "Sign-flip Category"}
    for i in range(len(strikes)):
        for j in range(len(tenors[i])):
            if pt2[i][j].empty:
                continue
            pt2[i][j] = pt2[i][j].assign(frac=pt2[i][j]['DTD LCY'] / pt2tot[i][j])
            # For a single-row table, simply extract its sflip value:
            if pt2[i][j].shape[0] == 1:
                sel_idx = pt2[i][j]['sflip'].astype('int').tolist()
            # For others use the get_idxminmax0, initially devised for Tenor selection:
            else:
                sel_idx = get_idxminmax0(pt2[i][j], 'sflip', min_xrac=0.90, max_xrac=1.10).\
                            astype('int').tolist()
            print(f"\nSELECTED Strike: {strikes[i]}, Tenor rank: {tenors[i][j]}\nsflips: {sel_idx}")
            df_tmp = pfs[i][j].query('sflip in @sel_idx')
            dtenor = df_tmp['Tenor'].unique().tolist()  # should yield a single value
            pfs[i][j] = pfs[i][j]\
                    .assign(Selected=["Yes" if s in sel_idx else "No" for s in pfs[i][j]['sflip']])
            pfs[i][j].drop(columns=['Strike', 'ord', 'um_ord', 'tflip'], inplace=True)
            print(f"\n{df_tmp}")
            if np.isinf(strikes[i]):
                mini_dtd += pt2[i][j].loc[pt2[i][j]['sflip'].isin(sel_idx), 'DTD LCY'].sum()
                df_tmp.drop(columns='Strike', inplace=True)
            pfs[i][j].rename(rn_dic, axis=1).style.applymap(highlight(lst=["Yes"])).\
                    to_excel(xclwrt, sheet_name=wsheet.title,
                            index=False, startrow=i_row, startcol=i_col)
            dstring = lst_unpack("Selected Strike:", [strikes[i]], ',')\
                                .replace('-inf', 'Missing MUREX')\
                                .replace('inf', 'Mini')
            dstring += lst_unpack(" Tenor:", dtenor, ',')
            dstring += lst_unpack(" sign-flip:", sel_idx)
            wsheet["AG" + str(i_row)] = dstring
            wsheet["AG" + str(i_row)].font = bold_font
            #@ i_row += (df_tmp.shape[0] + 2)
            i_row += (pfs[i][j].shape[0] + 2)
            # Aggregate the $ output values:
            pnl += df_tmp['DTD LCY'].sum()
            at_risk += df_tmp['Sensi Risk (OCY)'].sum()
            # Update the 6 output values with better values, if found:
            ord_min = min(ord_min, df_tmp['ord'].min())
            ord_max = max(ord_max, df_tmp['ord'].max())
            um_min = min(um_min, df_tmp['um_ord'].min())
            um_max = max(um_max, df_tmp['um_ord'].max())
            rate_min = min(rate_min, df_tmp['Rate_move'].min())
            rate_max = max(rate_max, df_tmp['Rate_move'].max())
    # mini_dtd /= mini_swp_tot
    return ord_min, ord_max, um_min, um_max, rate_min, rate_max, pnl, at_risk, mini_dtd


def work_3dflipped(pfs, tots, tenors, strikes, xclwrt, wsheet):
    """
    Summary:
    --------
    The purpose of this function is simply to reduce the cognitive complexity
    (evaluated by Sonar) of the get_3dflipped() function below. This function
    simply continues the processing started by get_3dflipped() which calls it.
    When finished, it returns the 8 output values to it. Essentially, here it
    extracts the sflip range contributing to 90-110% of the PnL for each Strike value.
    A single sflip value comprises a range of Tenors with the same $ PnL sign.
    So, we extract the contiguous groups of Tenor contrubutions to the 90-110%
    of the Strike's $ PnL subtotal. Additionally, for each Tenor value the tflip
    values show the Underlying Maturity values with the common sign of the $ PnL
    subtotal. So, here we perform a "NESTED SELECTION": based on sflip a group
    of adjacent Tenors is chosen and, based on tflip a group of Underlying
    Maturities is selected.

    Parameters:
    -----------
    pfs (list):     a list of SENSI pivot tables, one for each Strike value
    tots (list):    a list of the Total $ PnL, one for each Strike value
    tenors (list):  a list of the lists of Tenors, one for each Strike value
    strikes (list): the list of the Strike values

    Returns
    -----------
    ... (9-tuple): the min & max Tenor rank, the Underlying Maturity min & max rank,
                   the min and max Rate_move, the $ total PnL and $ at risk from Swaptions.
                   These are the value used to fill out the df_results table, Swaptions row.
                   The last member (mini_dtd) is the selected amount of Minisys Swaptions.
    """
    # Initialize the appropriate number of new pivot tables, one for each Strike value...
    pt2 = []
    # ... and the associated $ PnL subtotals, both as lists:
    pt2tot = []
    for i in range(len(strikes)):
        pt2.append([pd.DataFrame() for _ in range(len(tenors[i]))])
        pt2tot.append([0 for _ in range(len(tenors[i]))])

    # Fill the lists with actual tables and their $ subtotals:
    for i in range(len(strikes)):
        for j in range(len(tenors[i])):
            if tots[i][j] == 0:
                continue
            # split the pfs tables across their sflip values and aggregate $ PnL and $ at risk:
            pt2[i][j] = pfs[i][j].pivot_table(index='sflip',
                                              values=['DTD LCY', 'Sensi Risk (OCY)'],
                                              aggfunc=np.sum,
                                              margins=True,
                                              margins_name='Total')
            pt2tot[i][j] = pt2[i][j].loc['Total', 'DTD LCY']
            pt2[i][j].drop('Total', axis=0, inplace=True)
            pt2[i][j].reset_index(inplace=True)
    the_8tup = extra_3dflipped(pfs, tenors, strikes, pt2, pt2tot, xclwrt, wsheet)
    return the_8tup


def get_tranks(pf_, strike):
    """
    Summary:
    --------
    For Swaptions, given their Strike value, extract the associated Tenor rank list.

    Parameters:
    -----------
    pf_ (DataFrame):  SENSI table for Caplets (MUREX + Minisys), chosen for the Strike value below
    strike (float):   the Strike value

    Returns
    -----------
    ... (list):      the tenor rank list
    """
    idx_l = ['ord', 'Tenor']
    tot_tup = ('Total', '')
    # Fill  the actual pivot tables and their $ PnL subtotals:
    pfs = pf_.pivot_table(index=idx_l,
                          values='DTD LCY',
                          aggfunc=np.sum,
                          margins=True,
                          margins_name='Total')
    # skip the zero $PnL Strikes:
    pfs.drop('Total', axis=0, inplace=True)
    # A Series with True value if the row's $ PnL is zero, False otherwise:
    zero_dtd = (pfs['DTD LCY'] == 0)
    # Drop the rows with True (i.e., zero $ PnL) values:
    pfs.drop(zero_dtd.index[zero_dtd], axis=0, inplace=True)
    # Drop the row with missing Tenors, if any:
    try:
        pfs.drop(0, axis=0, inplace=True)
    except KeyError:
        pass
    # For the rest of the rows initialize the 'sflip' according to their $ PnL sign:
    pfs.reset_index(inplace=True)
    pfs = pfs.assign(sflip=np.sign(pfs['DTD LCY']))

    # Now, the actual 'sflip' values are being assigned:
    # Evaluate the difference as either 0 or 1 between this and previous row's sflip:
    pfs.loc[1:, 'sflip'] = abs(pfs['sflip'].diff(1) / 2)
    # Now all the rows have sflip either 0 or 1 (except perhaps the 0 row with -1 if $ PnL< 0).
    # If the current row's sflip is 0, then it has the same cumsum(sflip) as the previous row.
    # If the current row's sflip is 1, then its cumsum(sflip) will increase by 1:
    pfs['sflip'] = pfs['sflip'].cumsum()
    # split the pfs table across their sign-flip values and aggregate $ PnL and $ at risk:
    pt2 = pfs.pivot_table(index='sflip',
                          values='DTD LCY',
                          aggfunc=np.sum,
                          margins=True,
                          margins_name='Total')
    pt2tot = pt2.loc['Total', 'DTD LCY']
    pt2.drop('Total', axis=0, inplace=True)
    pt2.reset_index(inplace=True)
    # Now extract the sflip contiguous ranges contributing to 90-110% of the Strike $ subtotal:
    # the fractional contributions:
    pt2 = pt2.assign(frac=pt2['DTD LCY'] / pt2tot)
    # For a single-row table, simply extract its sflip value:
    if pt2.shape[0] == 1:
        sel_idx = pt2['sflip']
    # For others use the get_idxminmax0, initially devised for Tenor selection:
    else:
        sel_idx = get_idxminmax0(pt2, 'sflip', min_xrac=0.90, max_xrac=1.10)
    print(f"\nSELECTED sflips:\n{sel_idx}")
    df_tmp = pfs.query('sflip in @sel_idx')
    print(f"\n{df_tmp}")
    # Update the 4 output values with better values, if found:
    return sorted(df_tmp['ord'].unique().tolist())


def get_3dflipped(pf_, strikes, xclwrt, wsheet):
    """
    Summary:
    --------
    This 3D-version (for Swaptions, i.e., using Strike-Tenor-UnderMaturity variables) is
    designed for selecting the contiguous range of Tenor groups with the same $ PnL sign.
    To this end a new column 'sflip' is introduced which gets values e.g. -1, 0, 1, 2,...
    Each new value flags that the sign of $ PnL subtotal has changed from this Tenor on.
    Hence the number of 'sflip' values is the number of sign changes.
    Previously, individual adjacent Tenors were selected, and here Tenor GROUPS (with common
    sflip value) are being selected. The processing continues in the work_3dflipped()
    above to select the 8 output values for the Swaptions row df_results
    When finished, it returns the 8 output values to it. Essentially, here it
    extracts the sflip range contributing to 90-110% of the PnL for each Strike value.
    A single sflip value comprises a range of Tenors with the same $ PnL sign.
    So, we extract the contiguous GROUPS of Tenor contrubuting to the 90-110%
    of the Strike's $ PnL subtotal. Additionally, for each Tenor value the tflip
    values show the Underlying Maturity values with the common sign of the $ PnL
    subtotal. So, here we perform a "NESTED SELECTION": based on sflip a group
    of adjacent Tenors is chosen and, based on tflip a group of Underlying
    Maturities is selected.

    Parameters:
    -----------
    pf_ (DataFrame):  the SENSI table for Swaptions (MUREX & Minisys), prefiltered for a Currency
    strikes (list):   the list of the Strike values

    Returns
    -----------
    ... (8-tuple): the min & max Tenor rank, 2 empty strings (no Underlying Maturity for captions),
                   the min and max Rate_move, the $ total PnL and $ at risk from Caplets.
                   These are the value used to fill out the df_results table, Caplets row.
    """
    # Initialize the lists of pivot_tables and their $ PnL subtotals (one index for each Strike value):
    pfs = []
    tots = []
    # The 3Ds are Strike, Tenor and Underlying Maturity (with their Rate_move):
    idx_l = ['Strike', 'ord', 'Tenor', 'um_ord', 'Underlying Maturity', 'Rate_move']
    tot_tup = ('Total', '', '', '', '', '')
    # Initialize an empty list to add Tenors for each Strike:
    tenors = [ []  for _ in range(len(strikes))]
    # Fill  the actual pivot tables and their $ PnL subtotals:
    for i in range(len(strikes)):
        si = strikes[i]
        if np.isinf(si):
            mini_swp_tot = pf_.query("(Strike == @si) & (Tenor != 'blank')")['DTD LCY'].sum()
        # tenors[i].extend(sorted(pf_.query("Strike == @si")['ord'].unique().tolist()))
        tenors[i].extend(get_tranks(pf_.query('Strike == @si'), si))
        pfs.append([pd.DataFrame() for _ in range(len(tenors[i]))])
        tots.append([0 for _ in range(len(tenors[i]))])
        for j in range(len(tenors[i])):
            tij = tenors[i][j]
            pfs[i][j] = pf_.query("(Strike == @si) & (ord == @tij)").\
                pivot_table(index=idx_l,
                            values=['DTD LCY', 'Sensi Risk (OCY)'],
                            aggfunc=np.sum,
                            margins=True,
                            margins_name='Total')
            tots[i][j] = pfs[i][j].loc[tot_tup, 'DTD LCY']
            # skip the zero $PnL Strikes:
            if tots[i][j] == 0:
                continue
            tenor_ij = pf_.loc[pf_['ord'] == tij, 'Tenor'].iloc[0]
            print(f"Total for Strike {si}, Tenor {tij} {tenor_ij}: {tots[i][j]}")
            pfs[i][j].drop('Total', axis=0, inplace=True)
            # A Series with True value if the row's $ PnL is zero, False otherwise:
            zero_dtd = (pfs[i][j]['DTD LCY'] == 0)
            # Drop the rows with True (i.e., zero $ PnL) values:
            pfs[i][j].drop(zero_dtd.index[zero_dtd], axis=0, inplace=True)
            # For the rest of the rows initialize the 'sflip' according to its $ PnL sign:
            pfs[i][j] = pfs[i][j].assign(sflip = np.sign(pfs[i][j]['DTD LCY']))
            pfs[i][j] = pfs[i][j].assign(tflip = np.sign(tots[i][j]))
            pfs[i][j].reset_index(inplace=True)

    # Now, the actual 'sflip' values are being assigned:
    for i in range(len(strikes)):
        for j in range(len(tenors[i])):
            # skip this Strike-Tenor if zero $ PnL:
            if tots[i][j] == 0:
                continue
            # Evaluate the difference as either 0 or 1 between this and previous row's sflip:
            pfs[i][j].loc[1:, 'sflip'] = abs(pfs[i][j]['sflip'].diff(1) / 2)
            # Now all the rows have sflip either 0 or 1 (except perhaps the 0 row with -1 if $ PnL< 0).
            # If the current row's sflip is 0, then it has the same cumsum(sflip) as the previous row.
            # If the current row's sflip is 1, then its cumsum(sflip) will increase by 1:
            pfs[i][j]['sflip'] = pfs[i][j]['sflip'].cumsum()
            tij = tenors[i][j]
            tenor_ij = pf_.loc[pf_['ord'] == tij, 'Tenor'].iloc[0]

    # Minisys Strike index (if any):
    mst = -1
    try:
        mst = strikes.index(np.infty)
        # Get the Tenor subset that counts towards the Total $ PnL and update the lists at this index:
        pfs[mst], tots[mst], tenors[mst] = tenor_flipped(pfs[mst], tots[mst], tenors[mst])
    except Exception:
        pass
    # Finish the Swaption job:
    res_8tup = work_3dflipped(pfs, tots, tenors, strikes, xclwrt, wsheet)
    if mst >= 0:
        # update the Mini (from selected $ amount to selected fraction):
        res_8tup = *res_8tup[:8], res_8tup[8]/mini_swp_tot 
    return res_8tup


def l2to(lst):
    """A helper function to combine a range/pair of values into the form of "first to last"
       or simply "first" if only a single value given or identical to last:
    """
    # More than a single unique element:
    if (len(lst) > 1):
        if (lst[0] != lst[-1]):
            return f"{lst[0]} to {lst[-1]}"
    # Non-empty list (i.e., a single element):
    elif lst:
        return f"{lst[0]}"
    else:
        return "N/A"


def get_their_type(pf_vega):
    """
    pf_ vega(DataFrame):  the SENSI table filtered for IR Vega component,
                          gets updated with 2 extra cols: IRVega_Type and Source
    """
    path = pathlib.Path().cwd()
    df_types = pd.read_csv(path / "Minisys.csv")
    if pf_vega[pf_vega['Underlying'].isna()].empty:
        # SENSI coming from SQL (empty strings, not missing values), match them:
        df_types['Underlying'].fillna("", inplace=True)
    pf_vega = pd.merge(pf_vega, df_types, on="Underlying", how="left")
    if not pf_vega[pf_vega['Source'].isna()].empty:
        print("Warning: some rows have missing 'Source' values.")
    if not pf_vega[pf_vega['Source'] == 'MUREX'].equals(pf_vega[pf_vega['IRVega_Type'].isna()]):
        print("Warning: MUREX rows and missing 'IRVega_Type' rows not matching.")
    pf_vega['IRVega_Type'].fillna(pf_vega['Product Type'], inplace=True)
    return pf_vega


def doit_doit(pf_sensi, xclwrt=None):
    """The common engine of irvega_demo() and irvega_db() below.
    Parameters:
    -----------
    pf_ sensi(DataFrame):  the IRD desk SENSI table
    xclwrt (XlsWriter):    an Excel Writer object

    Returns
    -----------
    df_results (DataFrame) the table with the summary of results
    """
    str_lst = ["1. Obtain the SENSI table from iWork."]
    tmp_str = "2. Evaluate the DTD LCY breakdown of the SENSI table by Currencies"
    tmp_str += " and select the key Currencies contributing to 80-120% of the Total."
    tmp_str += " Similarly, evaluate the breakdown by nodes (unrequired for further workflow)."
    tmp_str += " (the Sensi IR Vega tab)."
    str_lst.append(tmp_str)
    tmp_str = "3. For each of the selected Currencies (the CCY Sensi IR Vega tab) carry"
    tmp_str += " out the following steps:"
    str_lst.append(tmp_str)
    tmp_str = "     (a) DTD LCY breakdown by Underlying values to ascertain their portfolio;"
    str_lst.append(tmp_str)
    tmp_str = "     (b) for the MUREX portfolio, DTD LCY breakdown by the Product Type"
    tmp_str += " to ascertain the type (Swaption or Caplet);"
    str_lst.append(tmp_str)
    tmp_str = "     (c) 2-d DTD LCY breakdown across both Underlying and Raw Component values. "
    str_lst.append(tmp_str)
    tmp_str = "4. After identifying the type of all Underlying contributions, divide them"
    tmp_str += " accordingly into Caps/floors and Swaptions. Then"
    str_lst.append(tmp_str)
    tmp_str = "     (a) for Caps/floors select the Strike values making up 90-110% of the PnL and"
    tmp_str += " for each Strike select the continuous Tenor groups with common DTD LCY signs"
    tmp_str += " (i.e., common sflip values) contributing to 90-110% of the subtotal."
    str_lst.append(tmp_str)
    tmp_str = "     (a) for Swaptions, additionally, for each Tenor select the groups of"
    tmp_str += " Underlying Maturity values with common DTD LCY signs" 
    tmp_str += " contributing to 90-110% of the subtotal."
    str_lst.append(tmp_str)
    pd.DataFrame({'Workflow': str_lst}).\
            to_excel(xclwrt, sheet_name='IRVega_Workflow', index=False, startrow=0, startcol=0)

    # Extract the local currency for the current entity:
    loc_ccy = pf_sensi['LCY'].mode().iloc[0]
    # Filter the IR Vega Component and get its Total DTD LCY:
    pf_vega = pf_sensi.query("Component == 'IR Vega'")
    pf_vega = get_their_type(pf_vega)
    # SQL returns empty string for missing values, can confuse Caplets/Swaptions
    pf_vega['Underlying Maturity'].replace('', np.nan, inplace=True)
    # The breakdown by the Currency:
    p_ccy = pf_vega.pivot_table(index='PL Currency',
                                values='DTD LCY',
                                margins=True,
                                margins_name='Total',
                                fill_value=0,
                                aggfunc=np.sum).sort_values('DTD LCY')
    tot_dtd = p_ccy.loc['Total', 'DTD LCY']
    p_ccy.drop('Total', axis=0, inplace=True)
    p_ccy.reset_index(inplace=True)
    th=[0.8, 0.1, 0.8, 1.2, 0.0]
    # The list of selected Currencies:
    my_ccys = get_selection2(p_ccy, tot_dtd, 'PL Currency', th=th)
    print(my_ccys, '\n')
    p_ccy.style.applymap(highlight(lst=my_ccys)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name="Sensi IR Vega", index=False, startrow=1, startcol=1)
    wsheet = xclwrt.book["Sensi IR Vega"]
    wsheet['B1'] = "DTD LCY breakdown by Currencies:"
    bold_font = Font(bold=True, size=12)
    wsheet["B" + str(p_ccy.shape[0] + 4)] = lst_unpack("Selected currencies:", my_ccys)
    wsheet["B" + str(p_ccy.shape[0] + 4)].font = bold_font

    # Have been shown only how to handle "USD" IR Vega, hence:
    ##my_ccys = set(my_ccys).intersection({'USD'})
    # The breakdown by the Node/Portfolio (eventually, not needed):
    # p_nod = pf_vega.pivot_table(index='Node',
    #                             values='DTD LCY',
    #                             margins=True,
    #                             margins_name='Total',
    #                             fill_value=0,
    #                             aggfunc=np.sum).sort_values('DTD LCY')
    # p_nod.drop('Total', axis=0, inplace=True)
    # p_nod.reset_index(inplace=True)
    # The list of selected nodes/portfolios:
    # my_nods = get_selection2(p_nod, tot_dtd, 'Node', th=th)
    # wsheet["G1"] = "DTD LCY breakdown by portfolios:"
    # p_nod.style.applymap(highlight(lst=my_nods)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
    #         to_excel(xclwrt, sheet_name="Sensi IR Vega", index=False, startrow=1, startcol=6)
    # print(my_nods, '\n')
    # wsheet["G" + str(p_nod.shape[0] + 4)] = lst_unpack("Selected portfolios:", my_nods)
    # wsheet["G" + str(p_nod.shape[0] + 4)].font = bold_font

    # column names for the DataFrame df_results which will hold the output
    dcols = ['Ccy', 'cap_tion', 'DTD_LCY', 'At_Risk', 'Restore', 'Tenors',
             'Strikes', 'UMaturity', 'Rate_min', 'Rate_max', 'Comment']
    df_results = pd.DataFrame(columns=dcols)
    tmp_val = ['All', '', tot_dtd, '', '', '', '', '', '', '',
               f"{sign_it(tot_dtd, loc_ccy)} from IR Vega"]
    tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
    df_results = df_results.append(tmp_dic, ignore_index=True)
    # a useful criterion to consider tables based on DTD _LCY vs Type "PL Restore"
    crit_dtd = lambda x: abs(x.loc[x['Type'] != "PL Restore - imported", 'DTD LCY'].sum()) > 0.0
    for ccy in my_ccys:
        cy_vega = pf_vega.query("`PL Currency` == @ccy")
        # if reading a csv table:
        cy_vega['Underlying'].fillna('blank', inplace=True)
        # if receiving it from a SQL query:
        cy_vega['Underlying'].replace('', 'blank', inplace=True)

        # The 1st pivot table: the breakdown by Underlying values for this Currency.
        c1u = cy_vega.pivot_table(index='Underlying',
                                  values='DTD LCY',
                                  fill_value=0,
                                  margins=True,
                                  margins_name='Total',
                                  aggfunc=np.sum).sort_values('DTD LCY')
        impute_pfolio(c1u, cy_vega)
        # temporarily add a boolean column (True for the "Total" row only, else False):
        c1u = c1u.assign(bool_key=c1u.index == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(c1u)
        print('\n')
        c1u.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=1, startcol=1)
        wsheet = xclwrt.book[f"{ccy} Sensi IR Vega"]
        wsheet["A1"] = f"DTD LCY Breakdown by Underlying values for {ccy}:"

        # The 2nd pivot table: for the MUREX Underlying value, the breakdown by the Product Type;
        murund = c1u.index[c1u['p_folio'] == "MUREX"]
        murund = murund[0] if len(murund) == 1 else murund.tolist()
        cy_vegamur = cy_vega.query("Underlying in @murund")
        cu2pt = cy_vegamur.pivot_table(index='Product Type',
                                       values='DTD LCY',
                                       fill_value=0,
                                       margins=True,
                                       margins_name='Total',
                                       aggfunc=np.sum).sort_values('DTD LCY')
        # temporarily add a boolean column (True for the "Total" row only, else False):
        cu2pt = cu2pt.assign(bool_key=cu2pt.index == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(cu2pt)
        print('\n')
        cu2pt.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=10, startcol=1)
        wsheet["A10"] = f"DTD LCY breakdown by Product Type values for MUREX {ccy}:"

        # The 3rd pivot table gives the breakdown across both Underlying and Raw Component values:
        c3urc = cy_vega.pivot_table(index=['Underlying', 'Raw Component'],
                                    values='DTD LCY',
                                    fill_value=0,
                                    margins=True,
                                    margins_name='Total',
                                    aggfunc=np.sum).sort_values('DTD LCY')
        # temporarily add a boolean column (True for the "Total" row only, else False):
        # c3urc = c3urc.assign(bool_key=[ix[0] == 'Total' for ix in c3urc.index])\  # OK, but...
        c3urc = c3urc.assign(bool_key=c3urc.index.droplevel(1) == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(c3urc)
        cc2 = pd.merge(c3urc, c1u, left_index=True, right_index=True, how='left').reset_index()
        # Minisys Caplet(s):
        crit_mincap = "(Source == 'Minisys') & (IRVega_Type == 'Caplet')"
        try:
            c_mincap = cy_vega.query(crit_mincap)['Underlying'].unique().tolist()
        except Exception:
            c_mincap = []
        crit_minswp = "(Source == 'Minisys') & (IRVega_Type == 'Swaption')"
        try:
            c_minswp = cy_vega.query(crit_minswp)['Underlying'].unique().tolist()
        except Exception:
            c_minswp = []
        print('\n')
        c3urc.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=1, startcol=10)
        wsheet["J1"] = f"DTD LCY breakdown by Underlying and Raw Component values for {ccy}:"

        tot_list = swapcap_tot(c1u, cu2pt, c3urc, cy_vega)
        irvega_type(c1u, c3urc, cy_vega, murund, tot_list, xclwrt, wsheet)

        mur_cap = cy_vega.query("`Product Type` == 'Caps/floors'")
        mur_cap['Strike'].fillna(-np.infty, inplace=True)
        try:
            strik_cap = get_strikes(mur_cap)
            print(f"Selected Caplet Strikes: {strik_cap}\n")
            wsheet["S1"] = "Caps/floors selection"
            wsheet["S2"] = lst_unpack("Selected Caplet Strikes:", strik_cap).replace('inf', 'Missing')
            wsheet["S2"].font = bold_font
        except Exception:
            strik_cap = []
        mini_cap = cy_vega.query("Underlying in @c_mincap")
        mini_cap['Strike'] = np.infty
        strik_cap.append(np.infty)
        all_cap = mur_cap.append(mini_cap)
        if (not all_cap.empty) & crit_dtd(all_cap):
            cap_tup = get_2dflipped(all_cap, strik_cap, xclwrt, wsheet)
            print(f"\nCaplet output values:\n{cap_tup}")
            # Convert the Tenor ranks 'ord' to actual Tenors:
            tenor_lst = [all_cap.query("ord == @cap_tup[0]")['Tenor'].iloc[0],
                         all_cap.query("ord == @cap_tup[1]")['Tenor'].iloc[0]]
            cap_rest = all_cap.query("Type.str.contains('Restore', case=False)",
                                     engine='python')['DTD LCY'].sum() * cap_tup[-1]
            tmp_val = [ccy, 'Caplet', cap_tup[6], cap_tup[7], cap_rest, l2to(tenor_lst),
                       l2to(strik_cap[:-1]), l2to(cap_tup[2:4]), cap_tup[4], cap_tup[5]]
            # Generate the Caplet comment string and append to this list:
            capswp_comment(tmp_val)
            # Prepend the comment with the LCY symbol:
            tmp_val[-1] = loc_ccy + ' ' + tmp_val[-1]
            tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
            df_results = df_results.append(tmp_dic, ignore_index=True)

        rank_umatur(cy_vega)
        mur_swp = cy_vega.query("`Product Type` == 'Swaptions'")
        mur_swp['Strike'].fillna(-np.infty, inplace=True)
        if (not mur_swp.empty) & crit_dtd(mur_swp):
            strik_swp = get_strikes(mur_swp)
            print(f"Selected Swaption Strikes: {strik_swp}\n")
            wsheet["AE1"] = "Swaption selection"
            wsheet["AE2"] = lst_unpack("Selected Swaption Strikes:", strik_swp).replace('inf', 'Missing')
            wsheet["AE2"].font = bold_font
        else:
            strik_swp = []
        # Finally, for USD SWAPTION we are going to use the rows with missing Underlying Maturity
        # values, so we fill those with 'blank':
        cy_vega['Underlying Maturity'].fillna('blank', inplace=True)
        # The corresponding Tenors can have many values (shown below), but we replace those with
        # 'blank', too, and Rate_moves with zero (akin to the table in the Excel sheet):
        # cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Tenor'].unique()
        cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Tenor'] = 'blank'
        cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Rate_move'] = 0
        # Only one more thing left: the *Sensi Risk (OCY)* values happened to be missing for the
        # 'blank' (Underlying Maturity, Tenor) pairs, which then (a bug?) makes the pivot table skip
        # their contributions to **BOTH** DTD LCY and Sensi Risk (OCY) when calculating the Total.
        # Impute zero to these 13 rows as a workaround:
        mini_swp = cy_vega.query("Underlying in @c_minswp")
        if (not mini_swp.empty) & crit_dtd(mini_swp):
            mini_swp['Strike'] = np.infty
            all_swp = mur_swp.append(mini_swp)
            strik_swp.append(np.infty)
        else:
            all_swp = mur_swp
        if (not all_swp.empty) & crit_dtd(all_swp):
            swp_tup = get_3dflipped(all_swp, strik_swp, xclwrt, wsheet)
            print(f"\nSwaption output values:\n{swp_tup}")
            tenor_lst = swp_tup[:2]
            # Convert the Tenor ranks 'ord' to actual Tenors:
            tenor_lst = [all_swp.query("ord == @swp_tup[0]")['Tenor'].iloc[0],
                         all_swp.query("ord == @swp_tup[1]")['Tenor'].iloc[0]]
            um_lst = [all_swp.query("um_ord == @swp_tup[2]")['Underlying Maturity'].iloc[0],
                      all_swp.query("um_ord == @swp_tup[3]")['Underlying Maturity'].iloc[0]]
            swp_rest = all_swp.query("Type.str.contains('Restore', case=False)",
                                     engine='python')['DTD LCY'].sum() * swp_tup[-1]
            tmp_val = [ccy, 'Swaption', swp_tup[6], swp_tup[7], swp_rest, l2to(tenor_lst),
                       l2to(strik_swp[:-1]), l2to(um_lst), swp_tup[4], swp_tup[5]]
            # Generate the Swaption comment string and append to this list:
            capswp_comment(tmp_val)
            # Prepend the comment with the LCY symbol:
            tmp_val[-1] = loc_ccy + ' ' + tmp_val[-1]
            tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
            df_results = df_results.append(tmp_dic, ignore_index=True)
    df_results.drop(columns='Strikes', inplace=True)
    return df_results


def irvega_demo(node_id='IRD', ddate='20200203', entity='DBSSG', pub_holiday=None, files=None):
    """
    Summary:
    -----------
    Executes the workflow of P&L attribution for IR Vega of the IRD trading desk.
    First, it obtains the pre-cleaned and pre-filtered tables pf_sensi, plexp,
    and pf_plva from get_clean_portfolio(), which reads this information either
    from a SQL query or the SQL-generated files YYYYMMDD_sensi.entity.TXT,
    YYYYMMDD_IWORK_PLVA.TXT and YYYYMMDD_PLVA.entity.TXT.
    Results are returned as the table df_results and commentary.

    Parameters:
    -----------
    node_id (str):    the trading desk (should be 'FX Options' here)
    ddate (str):      the Business Date in the string format, e.g. '20190430'
    entity (str):     the trading DBS entity (typically 'DBSSG')
    pub_holiday (*):  if supplied (i.e. not None), it flags the previous day was a public holiday
    files (str):      the path to the input files

    Returns:
    -----------
    df_results (DataFrame):  the table with up to 3 rows (for Total, Caplets and Swaptions) and
                             8 columns (Ccy, cap_tion, DTD_LCY, At_Risk... Commentary)
    """

    pf_sensi = get_sensi(ddate, files)
    get_tenors_ordered(pf_sensi)
    path = pathlib.Path().cwd() / "meta_data"
    xclfile = f"{parse(ddate).strftime('%Y-%m-%d')}_IRD.xlsx"
    xclwrt = pd.ExcelWriter(path / xclfile, mode="w", engine='openpyxl')
    bold_font = Font(bold=True, size=12)

    df_results = doit_doit(pf_sensi, xclwrt)
    print('\n', df_results)
    df_results.to_excel(xclwrt, sheet_name='IRD_Commentary', index=False, startrow=3, startcol=1)
    wsheet = xclwrt.book["IRD_Commentary"]
    wsheet["B1"] = "SUMMARY"
    wsheet["B1"].font = bold_font
    wsheet["B3"] = "Selection Table"
    # Generate the IR Vega final commentary from the last column:
    comment = df_results.iloc[0, -1] + " where"
    # Iterate over the rows of Series df_results.iloc[1:, -1]:
    for i_com in df_results.iloc[1:, -1].iteritems():
        comment += " \n(" + str(i_com[0]) + ") "
        comment += str(i_com[1])
    comment += "."
    print('\n', comment, '\n')
    wsheet["B" + str(df_results.shape[0] + 8)] = "COMMENTARY"
    wsheet["B" + str(df_results.shape[0] + 8)].font = bold_font
    wsheet["B" + str(df_results.shape[0] + 9)] = comment
    wsheet["B" + str(df_results.shape[0] + 9)].font = bold_font
    xclwrt.close()
    comm_dic = {'commentary': [comment], 'file_name': xclfile, 'components': ['IR Vega']}
    return df_results, comm_dic


def irvega_db(ddate, npth=None, tup5db=None):
    """
    Summary:
    -----------
    Same as irvega_demo() above, but relying on iWork DB to extract the SENSI data.

    Parameters:
    -----------
    ddate (str):    the Business Date in the string format, e.g. '2019-04-30'
    npth (str):     the trading desk (should be 'Interest Rate Derivatives & Structuring' here)
    tup5db (tuple): the 5-tuple carrying the user, password, IP address, port and DB name details


    Returns:
    -----------
    df_results (DataFrame):  the table with up to 4 rows (for MUREX/Minisys Caplets/Swaptions) and
                             12 columns (Ccy, Underlying, p_folio, cap_tion,... Commentary)
    """
    from Common_Flow.commentary_funtions import common_function
    if not npth:
        npth = 'GFM>DBS>DBS Singapore>DBSSG>Treasury>Sales and Trading>Interest Rate Desk>'
        npth += 'Interest Rate Derivatives & Structuring>'
    entity, node_id = np.array(npth.split('>'))[[3, -2]]
    mydb = PyMyDB(tup5db)
    mysql = TheQueries(mydb, ddate, entity, node_id)
    mysql.get_node_tree_path()
    dtd_pl = mysql.get_dtd_pl()
    pf_sensi = mysql.get_sensi()
    get_tenors_ordered(pf_sensi)
    path = pathlib.Path().cwd() / "meta_data"
    xclfile = f"{parse(ddate).strftime('%Y-%m-%d')}_IRD.xlsx"
    xclwrt = pd.ExcelWriter(path / xclfile, mode="w", engine='openpyxl')
    bold_font = Font(bold=True, size=12)
    if pf_sensi.empty:
        xclwrt.save()
        return {"commentary": [], "file_name": xclfile, "components": []}

    pd.DataFrame().to_excel(xclwrt, sheet_name='IRD_SENSI', startrow=0, startcol=0)
    wsheet = xclwrt.book["IRD_SENSI"]
    selected = pl_by_components(pf_sensi, dtd_pl, "The SENSI way", xclwrt, wsheet)
    print(selected)

    i_com = [0]
    comment = ""
    comments = []
    i_row = 1
    pd.DataFrame().to_excel(xclwrt, sheet_name='IRD_Commentary', index=False, startrow=3, startcol=1)
    wsheet = xclwrt.book["IRD_Commentary"]
    if "IR Vega" in selected:
        df_results = doit_doit(pf_sensi, xclwrt)
        print('\n', df_results)
        df_results.to_excel(xclwrt, sheet_name='IRD_Commentary', index=False, startrow=3, startcol=1)
        wsheet = xclwrt.book["IRD_Commentary"]
        wsheet["B1"] = "SUMMARY"
        wsheet["B1"].font = bold_font
        wsheet["B3"] = "Selection Table"
        # Generate the IR Vega final commentary from the last column:
        comment = df_results.iloc[0, -1] + " where"
        # Iterate over the rows of Series df_results.iloc[1:, -1]:
        for i_com in df_results.iloc[1:, -1].iteritems():
            comment += " \n[" + str(i_com[0]) + "] "
            comment += i_com[1]
        comment += "."
        comments = [comment]
        i_row = df_results.shape[0] + 8

    # extract prominent Components from other desks for Default Logic:
    extras = set(selected).difference(set(["IR Vega"]))
    for i, comp in enumerate(extras, i_com[0] + 1):
        sensi_nc = pf_sensi.query("Component == @comp")
        pl_comp = sensi_nc['DTD LCY'].sum()
        tmp_com, xclwrt = common_function(sensi=sensi_nc, component=comp, pl_com=pl_comp,
                                          writer=xclwrt, date=ddate, nodepath=npth)
        comments.append(tmp_com)

    print('\n', comment, '\n')
    wsheet["B" + str(i_row + 8)] = "COMMENTARY"
    wsheet["B" + str(i_row + 8)].font = bold_font
    wsheet["B" + str(i_row + 9)] = ' '.join(comments)
    wsheet["B" + str(i_row + 9)].font = bold_font
    xclwrt.close()
    comm_dic = {'commentary': comments, 'file_name': xclfile, 'components': selected}
    return comm_dic


def irvega_deflog(pf_sensi, xclwrt):
    """
    Summary:
    -----------
    Similar to irvega_demo() above, but for Default Logic (in case IR Vega features as a prominent
    component at another, non-IRD, trading desk).

    Parameters:
    -----------
    pf_sensi (DataFrame):  the SENSI table from another trading desk
    xclwrt (XlsWriter):    an Excel Writer object

    Returns:
    -----------
    df_results (DataFrame):  the table with up to 4 rows (for MUREX/Minisys Caplets/Swaptions) and
                             12 columns (Ccy, Underlying, p_folio, cap_tion,... Commentary)
    """

    bold_font = Font(bold=True, size=12)
    str_lst = ["1. Obtain the SENSI table from the calling function."]
    tmp_str = "2. Evaluate the DTD LCY breakdown of the SENSI table by Currencies"
    tmp_str += " and select the key Currencies contributing to 80-120% of the Total."
    tmp_str += " Similarly, evaluate the breakdown by nodes (unrequired for further workflow)."
    tmp_str += " (the Sensi IR Vega tab)."
    str_lst.append(tmp_str)
    tmp_str = "3. For each of the selected Currencies (the CCY Sensi IR Vega tab) carry"
    tmp_str += " out the following steps:"
    str_lst.append(tmp_str)
    tmp_str = "     (a) DTD LCY breakdown by Underlying values to ascertain their portfolio;"
    str_lst.append(tmp_str)
    tmp_str = "     (b) for the MUREX portfolio, DTD LCY breakdown by the Product Type"
    tmp_str += " to ascertain the type (Swaption or Caplet);"
    str_lst.append(tmp_str)
    tmp_str = "     (c) 2-d DTD LCY breakdown across both Underlying and Raw Component values. "
    str_lst.append(tmp_str)
    tmp_str = "4. After identifying the type of all Underlying contributions, divide them"
    tmp_str += " accordingly into Caps/floors and Swaptions. Then"
    str_lst.append(tmp_str)
    tmp_str = "     (a) for Caps/floors select the Strike values making up 90-110% of the PnL and"
    tmp_str += " for each Strike select the continuous Tenor groups with common DTD LCY signs"
    tmp_str += " (i.e., common sflip values) contributing to 90-110% of the subtotal."
    str_lst.append(tmp_str)
    tmp_str = "     (a) for Swaptions, additionally, for each Tenor select the groups of"
    tmp_str += " Underlying Maturity values with common DTD LCY signs"
    tmp_str += " contributing to 90-110% of the subtotal."
    str_lst.append(tmp_str)
    pd.DataFrame({'Workflow': str_lst}).\
            to_excel(xclwrt, sheet_name='IRVega_Workflow', index=False, startrow=0, startcol=0)

    # Extract the local currency for the current entity:
    loc_ccy = pf_sensi['LCY'].mode().iloc[0]

    pf_sensi.replace({r"\N": np.nan}, inplace=True)
    pf_sensi['DTD LCY'] = pd.to_numeric(pf_sensi['DTD LCY'])
    pf_sensi['Sensi Risk (OCY)'] = pd.to_numeric(pf_sensi['Sensi Risk (OCY)'])
    if 'Rate_move' not in pf_sensi.columns:
        pf_sensi["Mkt (T)"] = pd.to_numeric(pf_sensi["Mkt (T)"])
        pf_sensi["Mkt (T-1)"] = pd.to_numeric(pf_sensi["Mkt (T-1)"])
        pf_sensi = pf_sensi.assign(Rate_move=pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"])
    get_tenors_ordered(pf_sensi)
    pf_sensi['Rate_move'] = pf_sensi['Rate_move'] * 100

    # Filter the IR Vega Component and get its Total DTD LCY:
    pf_vega = pf_sensi.query("Component == 'IR Vega'")
    pf_vega = get_their_type(pf_vega)
    # SQL returns empty string for missing values, can confuse Caplets/Swaptions
    pf_vega['Underlying Maturity'].replace('', np.nan, inplace=True)
    # The breakdown by the Currency:
    p_ccy = pf_vega.pivot_table(index='PL Currency',
                                values='DTD LCY',
                                margins=True,
                                margins_name='Total',
                                fill_value=0,
                                aggfunc=np.sum).sort_values('DTD LCY')
    tot_dtd = p_ccy.loc['Total', 'DTD LCY']
    p_ccy.drop('Total', axis=0, inplace=True)
    p_ccy.reset_index(inplace=True)
    th=[0.8, 0.1, 0.8, 1.2, 0.0]
    # The list of selected Currencies:
    my_ccys = get_selection2(p_ccy, tot_dtd, 'PL Currency', th=th)
    print(my_ccys, '\n')
    p_ccy.style.applymap(highlight(lst=my_ccys)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name="Sensi IR Vega", index=False, startrow=1, startcol=1)
    wsheet = xclwrt.book["Sensi IR Vega"]
    wsheet['B1'] = "DTD LCY breakdown by Currencies:"
    wsheet["B" + str(p_ccy.shape[0] + 4)] = lst_unpack("Selected currencies:", my_ccys)
    wsheet["B" + str(p_ccy.shape[0] + 4)].font = bold_font

    # The breakdown by the Node/Portfolio (eventually, not needed):
    # p_nod = pf_vega.pivot_table(index='Node',
                                # values='DTD LCY',
                                # margins=True,
                                # margins_name='Total',
                                # fill_value=0,
                                # aggfunc=np.sum).sort_values('DTD LCY')
    # p_nod.drop('Total', axis=0, inplace=True)
    # p_nod.reset_index(inplace=True)
    # The list of selected nodes/portfolios:
    # my_nods = get_selection2(p_nod, tot_dtd, 'Node', th=th)
    # wsheet["G1"] = "DTD LCY breakdown by portfolios:"
    # p_nod.style.applymap(highlight(lst=my_nods)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            # to_excel(xclwrt, sheet_name="Sensi IR Vega", index=False, startrow=1, startcol=6)
    # print(my_nods, '\n')
    # wsheet["G" + str(p_nod.shape[0] + 4)] = lst_unpack("Selected portfolios:", my_nods)
    # wsheet["G" + str(p_nod.shape[0] + 4)].font = bold_font

    # column names for the DataFrame df_results which will hold the output
    dcols = ['Ccy', 'cap_tion', 'DTD_LCY', 'At_Risk', 'Restore', 'Tenors',
             'Strikes', 'UMaturity', 'Rate_min', 'Rate_max', 'Comment']
    df_results = pd.DataFrame(columns=dcols)
    tmp_val = ['All', '', tot_dtd, '', '', '', '', '', '', '',
               f"{sign_it(tot_dtd, loc_ccy)} from IR Vega"]
    tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
    df_results = df_results.append(tmp_dic, ignore_index=True)
    # Have been shown only how to handle "USD" IR Vega, hence:
    # my_ccys = set(my_ccys).intersection({'USD'})
    crit_dtd = lambda x: abs(x.loc[x['Type'] != "PL Restore - imported", 'DTD LCY'].sum()) > 0.0
    for ccy in my_ccys:
        cy_vega = pf_vega.query("`PL Currency` == @ccy")
        cy_vega['Underlying'].fillna('blank', inplace=True)
        cy_vega['Underlying'].replace('', 'blank', inplace=True)

        # The 1st pivot table: the breakdown by Underlying values for this Currency.
        c1u = cy_vega.pivot_table(index='Underlying',
                                  values='DTD LCY',
                                  fill_value=0,
                                  margins=True,
                                  margins_name='Total',
                                  aggfunc=np.sum).sort_values('DTD LCY')
        impute_pfolio(c1u, cy_vega)
        # temporarily add a boolean column (True for the "Total" row only, else False):
        c1u = c1u.assign(bool_key=c1u.index == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(c1u)
        print('\n')
        c1u.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=1, startcol=1)
        wsheet = xclwrt.book[f"{ccy} Sensi IR Vega"]
        wsheet["A1"] = f"DTD LCY breakdown by Underlying values for {ccy}"

        # The 2nd pivot table: for the MUREX Underlying value, the breakdown by the Product Type;
        murund = c1u.index[c1u['p_folio'] == "MUREX"]
        murund = murund[0] if len(murund) == 1 else murund.tolist()
        cy_vegamur = cy_vega.query("Underlying in @murund")
        cu2pt = cy_vegamur.pivot_table(index='Product Type',
                                       values='DTD LCY',
                                       fill_value=0,
                                       margins=True,
                                       margins_name='Total',
                                       aggfunc=np.sum).sort_values('DTD LCY')
        # temporarily add a boolean column (True for the "Total" row only, else False):
        cu2pt = cu2pt.assign(bool_key=cu2pt.index == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(cu2pt)
        print('\n')
        cu2pt.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=10, startcol=1)
        wsheet["A10"] = f"DTD LCY breakdown by Product Type values for MUREX {ccy}"

        # The 3rd pivot table gives the breakdown across both Underlying and Raw Component values:
        c3urc = cy_vega.pivot_table(index=['Underlying', 'Raw Component'],
                                    values='DTD LCY',
                                    fill_value=0,
                                    margins=True,
                                    margins_name='Total',
                                    aggfunc=np.sum).sort_values('DTD LCY')
        # temporarily add a boolean column (True for the "Total" row only, else False):
        c3urc = c3urc.assign(bool_key=c3urc.index.droplevel(1) == 'Total')\
                .sort_values(['bool_key', 'DTD LCY'], ascending=[True, False])\
                .drop('bool_key', axis=1)
        print(c3urc)
        cc2 = pd.merge(c3urc, c1u, left_index=True, right_index=True, how='left').reset_index()
        # Minisys Caplet(s):
        crit_mincap = "(Source == 'Minisys') & (IRVega_Type == 'Caplet')"
        try:
            c_mincap = cy_vega.query(crit_mincap)['Underlying'].unique().tolist()
        except Exception:
            c_mincap = []
        crit_minswp = "(Source == 'Minisys') & (IRVega_Type == 'Swaption')"
        try:
            c_minswp = cy_vega.query(crit_minswp)['Underlying'].unique().tolist()
        except Exception:
            c_minswp = []
        print('\n')
        c3urc.to_excel(xclwrt, sheet_name=f"{ccy} Sensi IR Vega", startrow=1, startcol=10)
        wsheet["J1"] = f"DTD LCY breakdown by Underlying and Raw Component values for {ccy}:"

        tot_list = swapcap_tot(c1u, cu2pt, c3urc, cy_vega)
        irvega_type(c1u, c3urc, cy_vega, murund, tot_list, xclwrt, wsheet)

        mur_cap = cy_vega.query("`Product Type` == 'Caps/floors'")
        mur_cap['Strike'].fillna(-np.infty, inplace=True)
        try:
            strik_cap = get_strikes(mur_cap)
        except Exception:
            strik_cap = []
        mini_cap = cy_vega.query("Underlying in @c_mincap")
        if not mini_cap.empty:
            mini_cap['Strike'] = np.infty
            strik_cap.append(np.infty)
        print(f"Selected Caplet Strikes: {strik_cap}\n")
        wsheet["S1"] = "Caps/floors selection"
        wsheet["S2"] = lst_unpack("Selected Caplet Strikes:", strik_cap).replace('inf', 'Missing')
        wsheet["S2"].font = bold_font
        all_cap = mur_cap.append(mini_cap)
        if (not all_cap.empty) & crit_dtd(all_cap):
            cap_tup = get_2dflipped(all_cap, strik_cap, xclwrt, wsheet)
            print(f"\nCaplet output values:\n{cap_tup}")
            # Convert the Tenor ranks 'ord' to actual Tenors:
            tenor_lst = [all_cap.query("ord == @cap_tup[0]")['Tenor'].iloc[0],
                         all_cap.query("ord == @cap_tup[1]")['Tenor'].iloc[0]]
            cap_rest = all_cap.query("Type.str.contains('Restore', case=False)",
                                     engine='python')['DTD LCY'].sum() * cap_tup[-1]
            tmp_val = [ccy, 'Caplet', cap_tup[6], cap_tup[7], cap_rest, l2to(tenor_lst),
                       l2to(strik_cap[:-1]), l2to(cap_tup[2:4]), cap_tup[4], cap_tup[5]]
            # Generate the Caplet comment string and append to this list:
            capswp_comment(tmp_val)
            # Prepend the comment with the LCY symbol:
            tmp_val[-1] = loc_ccy + ' ' + tmp_val[-1]
            tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
            df_results = df_results.append(tmp_dic, ignore_index=True)

        rank_umatur(cy_vega)
        mur_swp = cy_vega.query("`Product Type` == 'Swaptions'")
        mur_swp['Strike'].fillna(-np.infty, inplace=True)
        if (not mur_swp.empty) & crit_dtd(mur_swp):
            strik_swp = get_strikes(mur_swp)
        else:
            strik_swp = []
        # Finally, for USD SWAPTION we are going to use the rows with missing Underlying Maturity
        # values, so we fill those with 'blank':
        cy_vega['Underlying Maturity'].fillna('blank', inplace=True)
        # The corresponding Tenors can have many values (shown below), but we replace those with
        # 'blank', too, and Rate_moves with zero (akin to the table in the Excel sheet):
        # cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Tenor'].unique()
        cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Tenor'] = 'blank'
        cy_vega.loc[cy_vega['Underlying Maturity'] == 'blank', 'Rate_move'] = 0
        # Only one more thing left: the *Sensi Risk (OCY)* values happened to be missing for the
        # 'blank' (Underlying Maturity, Tenor) pairs, which then (a bug?) makes the pivot table skip
        # their contributions to **BOTH** DTD LCY and Sensi Risk (OCY) when calculating the Total.
        # Impute zero to these 13 rows as a workaround:
        mini_swp = cy_vega.query("Underlying == @c_minswp")
        if (not mini_swp.empty) & crit_dtd(mini_swp):
            mini_swp['Strike'] = np.infty
            all_swp = mur_swp.append(mini_swp)
            strik_swp.append(np.infty)
        else:
            all_swp = mur_swp
        print(f"Selected Swaption Strikes: {strik_swp}\n")
        wsheet["AE1"] = "Swaption selection"
        wsheet["AE2"] = lst_unpack("Selected Swaption Strikes:", strik_swp).replace('inf', 'Missing')
        wsheet["AE2"].font = bold_font
        if (not all_swp.empty) & crit_dtd(all_swp):
            swp_tup = get_3dflipped(all_swp, strik_swp, xclwrt, wsheet)
            print(f"\nSwaption output values:\n{swp_tup}")
            tenor_lst = swp_tup[:2]
            # Convert the Tenor ranks 'ord' to actual Tenors:
            tenor_lst = [all_swp.query("ord == @swp_tup[0]")['Tenor'].iloc[0],
                         all_swp.query("ord == @swp_tup[1]")['Tenor'].iloc[0]]
            um_lst = [all_swp.query("um_ord == @swp_tup[2]")['Underlying Maturity'].iloc[0],
                      all_swp.query("um_ord == @swp_tup[3]")['Underlying Maturity'].iloc[0]]
            swp_rest = all_swp.query("Type.str.contains('Restore', case=False)",
                                     engine='python')['DTD LCY'].sum() * swp_tup[-1]
            tmp_val = [ccy, 'Swaption', swp_tup[6], swp_tup[7], swp_rest, l2to(tenor_lst),
                       l2to(strik_swp[:-1]), l2to(um_lst), swp_tup[4], swp_tup[5]]
            # Generate the Swaption comment string and append to this list:
            capswp_comment(tmp_val)
            # Prepend the comment with the LCY symbol:
            tmp_val[-1] = loc_ccy + ' ' + tmp_val[-1]
            tmp_dic = {c: v for  c, v in zip(dcols, tmp_val)}
            df_results = df_results.append(tmp_dic, ignore_index=True)

    df_results.drop(columns='Strikes', inplace=True)
    print('\n', df_results)
    df_results.to_excel(xclwrt, sheet_name='IRD_Commentary', index=False, startrow=3, startcol=1)
    wsheet = xclwrt.book["IRD_Commentary"]
    wsheet["B1"] = "SUMMARY"
    wsheet["B1"].font = bold_font
    wsheet["B3"] = "Selection Table"
    # Generate the IR Vega final commentary from the last column:
    comment = df_results.iloc[0, -1] + " where"
    # Iterate over the rows of Series df_results.iloc[1:, -1]:
    for i_com in df_results.iloc[1:, -1].iteritems():
        comment += " \n(" + str(i_com[0]) + ") "
        comment += str(i_com[1])
    comment += "."
    print('\n', comment, '\n')
    wsheet["B" + str(df_results.shape[0] + 8)] = "COMMENTARY"
    wsheet["B" + str(df_results.shape[0] + 8)].font = bold_font
    wsheet["B" + str(df_results.shape[0] + 9)] = comment
    wsheet["B" + str(df_results.shape[0] + 9)].font = bold_font
    wsheet.sheet_state = 'hidden'
    xclwrt.save()
    return comment, df_results


if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--entity', '-e', default='DBSSG',
                    required=False, help="The trading DBS entity, typically 'DBSSG'")
    AP.add_argument('--date', '-d',
                    required=False, help="date in this format '20190430'")
    AP.add_argument('--node_id', '-n', default='IRD', choices=['IRD', 'CREDIT TRADING',
                                                               'FX Options'],
                    required=False, help="Trading Desk (e.g. 'IRD', 'Credit Trading', 'Fx Options'")
    AP.add_argument('--pub_holiday', '-p', default=False,
                    required=False, help="Day after a public holiday?")
    AP.add_argument('--files', '-f',
                    required=False, help="Path to the folder with the input files")
    ARG = vars(AP.parse_args())

    if ARG['date']:
        # _, cdic = irvega_demo(ARG['node_id'], ARG['date'], ARG['entity'], ARG['pub_holiday'], ARG['files'])
        cdic = irvega_db(ARG['date'])
        print(cdic)
    else:
        sensi = pd.read_csv(ARG['files'])
        xclwrt = pd.ExcelWriter(f"IRVega_{ARG['files']}.xlsx", engine='openpyxl')
        print(irvega_deflog(sensi, xclwrt))
