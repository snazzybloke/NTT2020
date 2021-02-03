#!/usr/bin/env python3
# coding: utf-8

####################################################
# Author: Ante Bilic                               #
# Since: Apr 29, 2020                              #
# Copyright: The PLC Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

"""Summary:
   -------
   Collection of functions for FX Options P/L attribution using SENSI method.
"""

import pathlib
import argparse
import re
import numpy as np
import pandas as pd
from pandas_ods_reader import read_ods
from dateutil.parser import parse
from FXO.common_futils_lcy import get_clean_portfolio, get_dtd_pl, pivtab_best, diag_sensi_vs_pl,\
                                  sign_it, human_format, get_idxminmax2, get_selection2,\
                                  lst_unpack, highlight, highlight_ser
from FXO.enquire5 import *
from string import ascii_uppercase
from openpyxl.styles import Font
from config.configs import getLogger
from config import configs
env = configs.env
logger = getLogger("fxo_xl.py")
# pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def get_vega_tenors_rates(pf_, tot_tenor):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Called only from Vega_bdown() for the selected ccy Quotations to select
    their Tenors. Handles the type of Tenors from the tenor_list below.
    The rows of the pf_ table are assigned order 'ord', percentage 'frac',
    and the partial variance 'vrac'. From the get_idxMinMax() the tenor
    range is evaluated. Then, to evaluate the associated avg_ratemove,
    the list of tenors in that range is compiled.

    Parameters:
    -----------
    pf_ (DataFrame):       a SENSI table pre-filtered for the specific component
                           (typically Fx Vega)
    tot_tenor (number):    the actual P/L from the ccy pair Quotation

    Returns:
    -----------
    pf_ (DataFrame):       the processed SENSI table, Tenor rank-sorted
    tenor_min (str):       the shortest tenor contributing to this Quotation
    tenor_max (str):       the longest tenor contributing to this Quotation
    avg_ratemove (float):  the average of the Rate_move across the tenor list
    the_amount (float):    the $ amount at risk
    """
    tenor_list = ['O/N', 'T/N', '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M',
                  '10M', '11M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '11Y',
                  '12Y', '13Y', '14Y', '15Y', '16Y', '17Y', '18Y', '19Y', '20Y', '30Y', '40Y', '50Y'
                  ]
    # The function get_tenor_idx returns the Tenor rank, i.e., the index in tenor_list:
    get_tenor_idx = lambda x: tenor_list.index(x) + 1 if x in tenor_list else 0
    # Replace 'WK' with 'W' and '12M' with '1Y':
    pf_['Tenor'].replace({'12M': '1Y'}, inplace=True)
    pf_['Tenor'].replace({str(i) + 'WK': str(i) + 'W' for i in range(1, 10)}, inplace=True)
    # Set the default to None for the four return values:
    tenor_min = None
    tenor_max = None
    avg_ratemove = None
    the_amount = None
    # Safe to drop the rows with PnL close to zero:
    pf_.drop(pf_.query("abs(`DTD LCY`) < 1.e-5").index, axis=0, inplace=True)
    # Ensure that Rate_move is numeric (as opposed to string sometimes):
    pf_['Rate_move'] = pf_['Rate_move'].astype('float')
    # If a Tenor appears multiple times, aggregate its Rate_move, DTD PnL and Sensi_Risk:
    if (pf_.groupby('Tenor')['DTD LCY'].count() > 1).any():
        pf_ = pf_.groupby(['Quotation', 'Tenor'], as_index=False).\
                agg({'Rate_move': 'mean', 'DTD LCY': 'sum', 'Sensi Risk (OCY)': 'sum'})
    # The column 'ord' with Tenor ranking, also used for sorting the rows:
    pf_['ord'] = pf_['Tenor'].map(get_tenor_idx).where(cond=pf_['Tenor'].notna(), other=0)
    pf_ = pf_.sort_values('ord', ascending=True).reset_index()
    # Introducing the column 'frac' (fractional contribution to the Total DTD PnL):
    pf_ = pf_.assign(frac=pf_['DTD LCY'].div(tot_tenor))
    n_size = pf_['frac'].shape[0]
    mu = tot_tenor / n_size
    # Introducing the column 'vrac' (partial contribution to the DTD PnL variance):
    pf_ = pf_.assign(vrac=(pf_['DTD LCY'] - mu)**2 / (n_size * pf_['DTD LCY'].var(ddof=0)))
    pf_.drop(columns='index', inplace=True)

    # Get the Tenor selection that makes the 90-110% of the Total DTD LCY:
    df_tenors = get_idxminmax2(pf_, 'Tenor', min_xrac=0.904, max_xrac=1.10)
    # The 1st and last Tenor selected:
    tenor_min = df_tenors.iloc[0]
    tenor_max = df_tenors.iloc[-1]

    # Get the indices of the 1st and last Tenor selected in the tenor_list:
    start_index = 0
    end_index = -1
    for index, value in enumerate(tenor_list):
        if (value == tenor_min) & (value == tenor_max):
            # a single tenor found, it's both min & max
            start_index = index
            end_index = index
        elif value == tenor_min:
            start_index = index
        elif value == tenor_max:
            end_index = index

    # From the indices get the actual Tenors:
    sel_tenor_lst = []
    for i in range(len(tenor_list)):
        if (i >= start_index and i <= end_index):
            sel_tenor_lst.append(tenor_list[i])

    # Evaluate the_amount $ at risk and avg_ratemove:
    df_tenor_select = pf_.query('Tenor in @sel_tenor_lst')
    sum_ratemove = df_tenor_select['Rate_move'].sum()
    the_amount = round(df_tenor_select['Sensi Risk (OCY)'].sum(), 0)
    cnt_tenor = df_tenor_select['Tenor'].nunique()
    avg_ratemove = round(abs(sum_ratemove / cnt_tenor * 100), 2)
    pf_.drop(columns=['frac', 'vrac'], inplace=True)
    return pf_, (tenor_min, tenor_max, avg_ratemove, the_amount)


def pl_by_components(pf_, dtd_pl, headline, xclwrt, wsheet):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Called after get_clean_portfolio(), get_dtd_pl() and diag_sensi_vs_pl().
    Provides the breakdown of PLVA or SENSI by their Components and returns
    the selection of the most prominent Component(s) for the day.

    Parameters:
    -----------
    pf_ (DataFrame):      either PLVA or SENSI table
    dtd_pl (float):       the $ amount from get_dtd_pl().
                          for SENSI approach based on diag_sensi_vs_pl()
    headline (str):       to print which pf_ is being broken down
    xclwrt (XlsWriter):   an Excel Writer object
    wsheet (Worksheet):   the current Excel worksheet

    Returns:
    -----------
    Comp_selection (list):   the selected items from Component column of pf_
    """
    print(headline)
    # Evaluate the pivot table showing the contribution of each Component to DTD PnL:
    pl_by_comp = pf_.pivot_table(index='Component',
                                 values='DTD LCY',
                                 aggfunc=np.sum,
                                 margins=True,
                                 margins_name='Total').\
                                         sort_values('DTD LCY', ascending=(dtd_pl < 0))
    print("\nThe PL Components:")
    print(pl_by_comp)
    # Extract the "grand total" DTD PnL from the table:
    gtotal = pl_by_comp.loc['Total', 'DTD LCY']
    pl_by_comp.drop('Total', axis=0, inplace=True)
    pl_by_comp.reset_index(inplace=True)
    # If it is not a "good" SENSI day, ignore the "grand total" and use the provided dtd_pl amount:
    if 'not good' in headline:
        gtotal = dtd_pl
        comp_selection = get_selection2(pl_by_comp, gtotal, 'Component',
                                        th=[0.75, 0.2, 0.8, 1.2, 0.005], top3=False)
    # If it is a "good" SENSI day, the "grand total" $ amount is reliable to use:
    else:
        comp_selection = get_selection2(pl_by_comp, gtotal, 'Component',
                                        th=[0.75, 0.2, 0.8, 1.2, 0.005], top3=False)
    wsheet["B1"] = "DTD PnL breakdown by Components:"
    pl_by_comp.style.applymap(highlight(lst=comp_selection)).\
            format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=1, startcol=1)
    # to unpack the list in the output string:
    wsheet["B" + str(pl_by_comp.shape[0] + 5)] = lst_unpack("Selected components:", comp_selection)
    bold_font = Font(bold=True, size=12)
    wsheet["B" + str(pl_by_comp.shape[0] + 5)].font = bold_font
    return comp_selection


def comp_by_(pf_plva, dtd_pl, idx, top_comp, top_contrib, th=None, top_pt=None):
    """
    Summary:
    -----------
    A helper function to select the top Product Type and Instrument items
    from pf_PLVA or its subset.

    Parameters:
    -----------
    pf_plvA (DataFrame): the PLVA table
    dtd_pl (number):     the amount read by get_dtd_pl(). Or only its SIGN.
    idx (str):           column name, 'Product Type' or 'Instrument'
    top_comp (str):      typically 'New deals'
    top_contrib (float): the Total $ amount from top_comp
    th (list):           5 thresholds (floats) for get_selection()
    top_pt (list):       optional, so the Instrument selection is across all
                         Product Types aggregated or only their top subset

    Returns:
    -----------
    top_idx (list):      the selection of top Product Types or Instruments
    """
    if top_pt:
        top_plva = pf_plva.query("`Product Type` in @top_pt")
    else:
        top_plva = pf_plva
    # Evaluate the pivot table showing the contribution of each Product Type/Instrument to DTD PnL:
    comp_by_idx = top_plva.query("Component == @top_comp").\
            pivot_table(index=idx,
                        values='DTD LCY',
                        aggfunc=np.sum,
                        margins=True,
                        margins_name="Total").\
                                sort_values('DTD LCY', ascending=(dtd_pl < 0))
    print("\nPLVA ", idx, ":")
    print(comp_by_idx.head())
    print(comp_by_idx.tail())
    print("\n")
    comp_by_idx.drop('Total', axis=0, inplace=True)
    comp_by_idx.reset_index(inplace=True)
    # Get the Product Type/Instrument selection making the desired range (90-110%) of top_contrib:
    top_idx = get_selection2(comp_by_idx, top_contrib, idx, th=th)
    # If any found, print them:
    if top_idx:
        print("\nThe top " + idx + ":\n" + str(top_idx))
    # Otherwise, nothing to show:
    else:
        print("\nNo substantial " + idx + " items found for " + top_comp + " commentary.")
        # Optional info: what are the idx values that contribute primarily to the top_comp?
        idx_best_comp = pivtab_best(pf_plva, the_idx=idx, the_col='Component', dtd_pl=dtd_pl)
        bestc = idx_best_comp.query("Component == @top_comp")
        if not bestc.empty:
            print("\n", idx, "(s) whose highest contribution are to the " + top_comp)
            print(bestc)
            bc_list = bestc[idx].tolist()
            print("\nThe total contribution from these ", idx)
            print(pf_plva.query("`{0}` in @bc_list".format(idx))['DTD LCY'].sum())
    return top_idx


def plva_bdown(pf_plva, dtd_pl, loc_ccy='SGD', plva_comps=None, df_results=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Carries out the breakdown of PLVA by Component, Instrument, PL Type.
    First, the pivot table of pf_PLVA is evaluated to select the top compinent.
    This may happen to be one of the "Greeks" (IR/Fx Delta, Gamma, Vega) for
    which one of the SENSI breakdown functions (below) is appropriate. Hence,
    the pivot table, PLVA_Comps, is filtered for PLVA_Comps, which excludes the
    "Greeks", but may include New Deals for which PLVA-style breakdown is right.
    If the latter is the case, the top Component (usually New Deals) only
    survives in PLVA_Comps and then the top Product Types are selected, followed
    by top Instruments. If the list of top Product Types is not empty, again
    we re-evaluate the top Instruments from the filtered subset, with only top
    Product Type contributions inside. The table with results, df_results is
    then updated with the PLVA/New Deals information and returned.

    Parameters:
    -----------
    pf_plvA (DataFrame):     the PLVA table
    dtd_pl (number):         the amount read by get_dtd_pl(). Or only its SIGN.
    loc_ccy (str):           the local currency for the DBS entity, usually 'SGD'
    plva_comps (list)        what's left from PL_by_Components(pf_sensi,...)
                             once the "Greeks" are removed from its output
                             (typically either [] or ['New deals'])
    df_results (DataFrame):  the 5-column table that holds all the results

    Returns:
    -----------
    df_results (DataFrame):  the 5-column table updated (not if PLVA_Comps==[])
    """
    # Evaluate the pivot table showing the contribution of each Component to PLVA DTD PnL:
    plva_by_comp = pf_plva.pivot_table(index='Component',
                                       values='DTD LCY',
                                       aggfunc=np.sum,
                                       margins=True,
                                       margins_name='Total').\
                                               sort_values('DTD LCY', ascending=(dtd_pl < 0))
    # Extract the "grand total":
    gtotal = plva_by_comp.loc['Total', 'DTD LCY']
    plva_by_comp.drop('Total', axis=0, inplace=True)
    plva_by_comp = plva_by_comp.reset_index()
    # Get the top DTD PnL contributing Component:
    i_max = plva_by_comp['DTD LCY'].div(gtotal).idxmax()
    top_comp = plva_by_comp.iloc[i_max, 0]
    # Get its DTD PnL contributing $ amount:
    print("\nThe top component based on PLVA: " + top_comp)
    top_contrib = plva_by_comp.iloc[i_max, -1]
    print("Its total DTD LCY contribution: " + str(top_contrib))
    # Now focus only on the contributions from plva_comps (typically ['New deals'])
    plva_by_comp = plva_by_comp.query("Component in @plva_comps")
    plva_by_comp.reset_index(inplace=True, drop=True)
    if not plva_by_comp.empty:
        # Set the default values to '' for the df_results columns:
        tmp_dict = {'Component': '', 'Selected': '', 'PnL': '',
                    'Ccy/Instrument': '', 'Product_Type': '', 'Comment': ''}
        # Get the top DTD PnL contributing Component:
        i_max = plva_by_comp['DTD LCY'].div(gtotal).idxmax()
        plva_comp = plva_by_comp.iloc[i_max, 0]
        # In case it's different from the top_comp above, show it:
        if plva_comp != top_comp:
            print("\nThe top PLVA component: " + plva_comp)
            plva_contrib = plva_by_comp.iloc[i_max, -1]
            print("Its total DTD LCY contribution: " + str(plva_contrib))
        # Otherwise, assign the previously evaluated values:
        else:
            plva_comp = top_comp
            plva_contrib = top_contrib
        # Assign the Component name and its DTD PnL to output dictionary:
        if not plva_comp in df_results['Component']:
            tmp_dict['Component'] = plva_comp
            tmp_dict['PnL'] = plva_contrib
        # Start making the string for the PLVA Commentary:
        comment = f"{sign_it(tmp_dict['PnL'], loc_ccy)} from {tmp_dict['Component']}"
        comment += " where largest contribution(s) are from the following "
        print("\nProduct Type for PLVA aggregated across Instruments:")
        # Get the selection of 80-120% $-contributing Product Types:
        top_pt = comp_by_(pf_plva, dtd_pl, 'Product Type', plva_comp, plva_contrib,
                          [0.8, 0.1, 0.8, 1.2, 0.0])
        # Convert the list into a string:
        tmp_dict['Product_Type'] = str.join(', ', top_pt)
        # Get the selection of 80-120% $-contributing Instruments:
        top_ins = comp_by_(pf_plva, dtd_pl, 'Instrument', plva_comp, plva_contrib,
                           [0.8, 0.1, 0.8, 1.2, 0.0])
        if top_pt:
            comment += f"product(s): {tmp_dict['Product_Type']}, and "
            print("\nSelecting Instruments for PLVA Product Types found in: ", top_pt)
            # For the selected Product Types get the selection of 80-120% contributing Instruments:
            top_ins = comp_by_(pf_plva, dtd_pl,'Instrument', plva_comp, plva_contrib,
                               [0.8, 0.1, 0.8, 1.2, 0.0], top_pt)
        # Convert the list into a string:
        tmp_dict['Ccy/Instrument'] = str.join(', ', top_ins)
        comment += f"instrument(s): {tmp_dict['Ccy/Instrument']}."
        tmp_dict['Comment'] = comment
        df_results = df_results.append(tmp_dict, ignore_index=True)
        return df_results


def delta_bdown(pf_, idx='PL Currency', comp='Fx Delta', dtd_pl=None, df_results=None, tup4=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Carries out the Delta-style breakdown of a SENSI table, typically splitting
    on PL Currency or Product Type. First, the pivot table of pf_ is
    evaluated to select the top contributing items.

    Parameters:
    -----------
    pf_ (DataFrame):         the SENSI table
    idx (str)                the column name to make the breakdown on
    comp (str)               the Component choice, typically Fx Delta
    dtd_pl (number):         the amount read by get_dtd_pl(). Or only its SIGN.
    df_results (DataFrame):  the 5-column table that holds all the results
    tup4 (4-tuple):          xclwrt (XlsWriter), wsheet (Worksheet), i_row (int), i_col (int)

    Returns:
    -----------
    df_results (DataFrame):  the 5-column table updated
    top_dict (dict):         the selected PL Currencies and the associated
                             $ amount at risk
    i_col (int):             the column index for another table (in outsource())
    """
    # Evaluate the pivot table showing the contribution of each Currency/Product Type to DTD PnL:
    pfc = pf_.query("Component == @comp").\
            pivot_table(index=idx,
                        values=['Sensi Risk (OCY)', 'DTD LCY'],
                        aggfunc=np.sum,
                        dropna=False,
                        fill_value=0,
                        margins=True,
                        margins_name='Total').\
                                sort_values('DTD LCY', ascending=(dtd_pl < 0))
    print(pfc)
    xclwrt, wsheet, i_row, i_col = tup4
    bold_font = Font(bold=True, size=12)
    cell = ascii_uppercase[i_col] + "1"
    wsheet[cell] = f"DTD PnL and exposure breakdown by {idx}"
    # Extract the grand Total $:
    tot_amount = pfc.loc['Total', 'DTD LCY']
    print("\nTotal DTD LCY from " + comp + ": " + str(tot_amount))
    cell = ascii_uppercase[i_col] + str(pfc.shape[0] + 4)
    wsheet[cell] = f"Total DTD LCY from {comp}: {tot_amount:.2f}"
    wsheet[cell].font = bold_font
    pfc.drop('Total', axis=0, inplace=True)
    pfc_ = pfc.reset_index()
    print("\n" + idx + " largest contributions to the " + comp + ":\n")
    # For Currency breakdown get the selection of top 80-120% contributiing Currencies:
    if idx == 'PL Currency':
        top_bit = get_selection2(pfc_, tot_amount, idx, th=[0.8, 0.1, 0.8, 1.2, 0.0])
    # For Product Type breakdown get the selection of top 90-110% contributiing Product Types:
    else:  # Product Type
        top_bit = get_selection2(pfc_, tot_amount, idx, th=[0.9, 0.1, 0.9, 1.1, 0.0])
    pfc_.style.applymap(highlight(lst=top_bit)).format('{:.2f}', na_rep="-").\
            to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=i_row, startcol=i_col)
    print("\nSelected " + idx + ": " + str(top_bit))
    cell = ascii_uppercase[i_col] + str(i_row + pfc.shape[0] + 6)
    # to unpack the list in the output string:
    wsheet[cell] = lst_unpack(f"{idx} selection:", top_bit)
    wsheet[cell].font = bold_font
    # Set up the default values to '' for the df_results columns:
    tmp_dict = {'Component': '', 'Selected': '', 'PnL': '',
                'Ccy/Instrument': '', 'Product_Type': '', 'Comment': ''}
    # delta_bdown() is called twice: first with idx='PL Currency', which fills out the cols below:
    if not comp in df_results['Component'].tolist():
        tmp_dict['Component'] = comp
        tmp_dict['PnL'] = tot_amount
        if idx == 'PL Currency':
            tmp_dict['Ccy/Instrument'] = str.join(', ', top_bit)
        df_results = df_results.append(tmp_dict, ignore_index=True)
    # The 2nd call to delta_bdown() fills out only the 'Product Type' col:
    else:
        if idx == 'Product Type':
            prod_types = str.join(', ', top_bit)
            df_results.loc[df_results['Component'] == comp, 'Product_Type'] = prod_types
    # For each selected Currency make a record of their DTD PnL and $ amount at risk:
    top_dict = {ccy: (pfc.loc[ccy, 'DTD LCY'], pfc.loc[ccy, 'Sensi Risk (OCY)']) for ccy in top_bit}
    # Return both the df_results and this record for further processing
    i_col += pfc_.shape[1] + 5
    return df_results, top_dict, i_col


def gamma_bdown(pf_sensi, idx=None, col='Product Type', comp='Fx Gamma', dtd_pl=0, df_results=None,
                tup4=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Carries out the Gamma-style breakdown of a SENSI table, typically splitting
    on Instrument and Product Type. First, the pivot table of pf_sensi is
    evaluated to select the top contributing items. First it tries to identify
    the top Instrument items. If that returns nothing, then it goes Delta-style
    breakdown, by invoking Delta_bdown() for PL Currency and then Product Type.
    Otherwise, it filters only on the selected Instruments and from this subsest
    selects the Product Type items.

    Parameters:
    -----------
    pf_sensi (DataFrame):    the SENSI table
    idx (list/str)           the column names to make the breakdown on
    col (str)                the column name for an additional breakdown
    comp (str)               the Component choice, typically Fx Gamma
    dtd_pl (number):         the amount read by get_dtd_pl(). Or only its SIGN.
    df_results (DataFrame):  the 5-column table that holds all the results
    tup4 (4-tuple):          xclwrt (XlsWriter), wsheet (Worksheet), i_row (int), i_col (int)

    Returns:
    -----------
    df_results (DataFrame):  the 5-column table updated
    top_dict (dict):         the selected Instruments (or Currencies) and the
                             associated $ amount at risk
    i_col (int):             the column index for another table (in outsource())
    """
    # Evaluate the pivot table showing the breakdown by each Instrument/Product Type pair to DTD PnL
    pfc = pf_sensi.query("Component == @comp").\
            pivot_table(index=idx,
                        columns=col,
                        values=['Sensi Risk (OCY)', 'DTD LCY'],
                        aggfunc=np.sum,
                        dropna=False,
                        fill_value=0,
                        margins=True,
                        margins_name='-Total').\
                                sort_values(('DTD LCY', '-Total'), ascending=(dtd_pl < 0))
    # Extract the grand Total $:
    tot_amount = pfc.loc['-Total', ('DTD LCY', '-Total')].iloc[0]
    pfc.columns.name = None
    # Combine the Multi-index column names into a single index:
    pfc.columns = [x[0] + x[1] for x in pfc.columns]
    print("\nTotal DTD LCY from " + comp + ": " + str(tot_amount))
    # Drop the row with the grand Total and rows with missing DTD PnL
    pfc.drop(pfc.query("`PL Currency` == '-Total'").index, axis=0, inplace=True)
    pfc.dropna(subset=['DTD LCY-Total'], inplace=True)
    pfc_ = pfc.reset_index()
    idx1 = idx[-1] if isinstance(idx, list) else idx
    # Get the selection of Instruments making up 80-120% of the Total DTD PnL:
    top_ins = get_selection2(pfc_, tot_amount, idx1, 'DTD LCY-Total',
                             [0.8, 0.1, 0.8, 1.2, 0.0], True)
    xclwrt, wsheet, i_row, i_col = tup4
    bold_font = Font(bold=True, size=12)
    wsheet["B1"] = f"DTD LCY breakdown for {comp}"
    pfc.reset_index().style.applymap(highlight(lst=top_ins)).format("{:.2f}", na_rep="-").\
            to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=1, startcol=1)
    i_row = pfc.shape[0] + 4
    wsheet["B" + str(i_row)] = f"Total DTD LCY from {comp}: {tot_amount:.2f}"
    wsheet["B" + str(i_row)].font = bold_font

    # For each selected Instrument make a record of their DTD PnL and $ at risk:
    tmp_dict = {}
    for ins in top_ins:
        ccys = ins.split('/')
        if (ccys[0], ins) in pfc.index:
            tmp_dict.update({ins: (pfc.loc[(ccys[0], ins), 'DTD LCY-Total'],
                                   pfc.loc[(ccys[0], ins), 'Sensi Risk (OCY)-Total'])})
        elif (ccys[1], ins) in pfc.index:
            tmp_dict.update({ins: (pfc.loc[(ccys[1], ins), 'DTD LCY-Total'],
                                   pfc.loc[(ccys[1], ins), 'Sensi Risk (OCY)-Total'])})
    top_ins = tmp_dict
    # If no Instrument selected, one can try Delta-style breakdown by Currencies (incorrect!):
    if not top_ins:
        print("\nNo substantial " + idx1 + " items found for " + comp + " commentary.")
        print("Trying PL Currency (i.e., Fx Delta style) instead:\n")
        # try with the PL Currency only, i.e. the Fx Delta way:
        df_results, top_ins = delta_bdown(pf_sensi, 'PL Currency', 'Fx Gamma', dtd_pl, df_results)
        df_results, top_pt = delta_bdown(pf_sensi, 'Product Type', 'Fx Gamma', dtd_pl, df_results)
        top_pt = list(top_pt.keys())
        i_row += 3
        i_col += 2
    # Otherwise with the selected Instrument, evaluate the pivot table showing Product Type contribs
    else:
        print("\nSelected Instrument(s): " + str(list(top_ins.keys())))
        i_row += 2
        wsheet["B" + str(i_row)] = lst_unpack("Selected Instrument(s):", list(top_ins.keys()))
        wsheet["B" + str(i_row)].font = bold_font
        print(f"\nProduct Type for {comp} Instruments ", list(top_ins.keys()))
        i_row += 2
        wsheet["B" + str(i_row)] = lst_unpack(f"Product Types for {comp} Instruments",
                                              list(top_ins.keys()), ':')
        pftmp = pf_sensi.query("(Component == @comp) & (Instrument in @top_ins.keys())")
        piv_pt = pftmp.pivot_table(index='Product Type',
                                   values='DTD LCY',
                                   aggfunc=np.sum,
                                   margins=True,
                                   margins_name='Total').\
                                           sort_values('DTD LCY', ascending=(dtd_pl < 0))
        print(piv_pt)
        # Extact the Total
        totax = piv_pt.loc['Total', 'DTD LCY']
        piv_pt.drop('Total', axis=0, inplace=True)
        piv_pt.reset_index(inplace=True)
        # Get the selection of Product Types making up 90-110% the DTD PnL:
        top_pt = get_selection2(piv_pt, totax, 'Product Type', 'DTD LCY', [0.9, 0.1, 0.9, 1.1, 0.0])
        piv_pt.style.applymap(highlight(lst=top_pt)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
                to_excel(xclwrt, sheet_name=wsheet.title, index=False, startrow=i_row, startcol=1)
        # Set up the default values to '' for the df_results columns:
        tmp_dict = {'Component': '', 'Selected': '', 'PnL': '',
                    'Ccy/Instrument': '', 'Product_Type': '', 'Comment': ''}
        # Prefill four columns of the Fx Gamma row:
        if not comp in df_results['Component'].tolist():
            tmp_dict['Component'] = comp
            tmp_dict['PnL'] = tot_amount
            tmp_dict['Ccy/Instrument'] = str.join(', ', top_ins)
            tmp_dict['Product_Type'] = str.join(', ', top_pt)
            df_results = df_results.append(tmp_dict, ignore_index=True)
        i_row += piv_pt.shape[0] + 3
        i_col += piv_pt.shape[1] + 5
    print("\nSelected " + str(idx) + ":\n" + str(top_ins) + str(top_pt))
    wsheet["B" + str(i_row + 1)] = lst_unpack("Selected", idx, end=':')
    wsheet["B" + str(i_row + 1)].font = bold_font
    wsheet["B" + str(i_row + 2)] = lst_unpack(d_list=top_ins, end=',')
    wsheet["B" + str(i_row + 2)].font = bold_font
    wsheet["B" + str(i_row + 3)] = lst_unpack(d_list=top_pt)
    wsheet["B" + str(i_row + 3)].font = bold_font
    # Return both the df_results and top Instruments for further processing
    return df_results, top_ins, i_col


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


def vega_bdown(pf_sensi, drill_on='Quotation', comp='Fx Vega', dtd_pl=None, loc_ccy='SGD',
               df_results=None, xclwrt=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    The breakdown of SENSI by Component, Instrument, PL Type
    First, it makes use of the pf DataFrame to calculate a pivot table, split on
    the values of drill_on (typically ccy Quotation), from which the top
    contributions are selected to the top_Drill list. Then the loop over the
    list makes a pivot table split on ['Tenor', and 'Rate_move'], with the sum of
    'DTD LCY' values for each pair, named "piv_pfCQ". Then one invokes
    get_Vega_tenors_rates() to evalute the 4-tuple of results and update the
    dictionary tup_dict, which has each Quotation pairs as keys, while the
    values is the list [Tot_amount, 4-tuple]. The storage is needed so that
    the following for-loop can print the commentaries in the sorted $ order.
    Finally a piv_pt table, split on the 'Product Type', is evaluated and the
    key product types associated with the top_Drill selection evaluated.
    The table that holds all the results, df_results, is updated and returned.

    Parameters:
    -----------
    pf_sensi   (DataFrame):  the SENSI table
    drill_on (str):          the column from the SENSI table to choose from the top contributions
    comp (str):              the Component value to select ("Fx Vega" the only meaningful choice)
    dtd_pl (number):         the amount read by get_dtd_pl(). Or only its SIGN.
    loc_ccy (str):           the local currency for the DBS entity, usually 'SGD'
    df_results (DataFrame):  the 5-column table that holds all the results
    xclwrt (XlsWriter):      an Excel Writer object

    Returns:
    -----------
    df_results (DataFrame):  the 5-column table updated
    """

    # Evaluate the pivot table showing the breakdown by Quotation to DTD PnL
    pfcomp = pf_sensi.query("Component == @comp")
    pl_byquot = pfcomp.pivot_table(index=drill_on,
                                   values='DTD LCY',
                                   aggfunc=np.sum,
                                   margins=True,
                                   margins_name='Total').\
                                           sort_values('DTD LCY', ascending=(dtd_pl < 0))
    print(pl_byquot.head(10))
    # Extract the grand Total $:
    tot_amount = pl_byquot.loc['Total', 'DTD LCY']
    print("\nTotal DTD LCY from " + comp + ": " + str(tot_amount))
    pl_byquot.drop('Total', axis=0, inplace=True)
    pl_byquot.reset_index(inplace=True)
    # Get the selection of top Quotations making up 80-120% of the DTD PnL:
    top_drill = get_selection2(pl_byquot, tot_amount, drill_on, 'DTD LCY', [0.8, 0.1, 0.8, 1.2, 0.0])
    # If none selected, simply get the top 3 items:
    if not top_drill:
        top_drill = pl_byquot.query("`DTD LCY` * @dtd_pl > 0")['Quotation'].head(3).tolist()
    # Start making the string for the Fx Vega commentary:
    comment = f"{sign_it(tot_amount, loc_ccy)} from Fx Vega"
    print("\nSelected Quotation(s): " + str(top_drill))
    pl_byquot.style.applymap(highlight(lst=top_drill)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='Fx Vega', index=False, startrow=1, startcol=1)
    wsheet = xclwrt.book["Fx Vega"]
    wsheet["A1"] = f"DTD LCY breakdown for {comp} across {drill_on}:"
    bold_font = Font(bold=True, size=12)
    wsheet["H1"] = f"Total DTD LCY from {comp}: {tot_amount:.2f}"
    wsheet["H1"].font = bold_font
    wsheet["H3"] = lst_unpack("Selected Quotation(s):", top_drill)
    wsheet["H3"].font = bold_font
    comment += " where largest contribution"
    comment += "s are from:" if len(top_drill) > 1 else " is from"
    tup_dict = {t: None for t in top_drill}
    # For each selected Quotation
    i_row = 5
    for i, t in enumerate(top_drill):
        print("\n", t, ":")
        pfcq = pfcomp[pfcomp[drill_on] == t]
        # evaluate the pivot table showing the breakdown by Quotation/Tenor/Rate_move to DTD PnL
        piv_pfcq = pfcq.pivot_table(index=[drill_on, 'Tenor', 'Rate_move'],
                                    values=['Sensi Risk (OCY)', 'DTD LCY'],
                                    aggfunc=np.sum,
                                    margins=True,
                                    margins_name='Total').\
                                            sort_values('DTD LCY', ascending=(dtd_pl < 0))
        # extract the grand Total $:
        tot_tenors = piv_pfcq.loc[('Total', '', ''), 'DTD LCY']
        piv_pfcq.drop('Total', axis=0, inplace=True)
        piv_pfcq.reset_index(inplace=True)
        # piv_pfcq.drop(columns='index', inplace=True)
        print(piv_pfcq)
        print("\nTotal DTD LCY from " + t + " : " + str(tot_tenors))
        print("\nSelecting tenors, avg_rate, amount_at_risk:\n")
        # get the Tenor range, avg_ratemove and $ amount at risk:
        piv_pfcq, tup4 = get_vega_tenors_rates(piv_pfcq, tot_tenors)
        piv_pfcq.style.applymap(highlight(lst=tup4)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='Fx Vega', index=False, startrow=i_row, startcol=7)
        i_row += piv_pfcq.shape[0] + 2
        wsheet["H" + str(i_row + 1)] = f"Total DTD LCY from {t}: {tot_tenors:.2f}"
        print(tup4)
        tmp_str = f"Selected Tenor range: {tup4[0]}-{tup4[1]}, "
        tmp_str += f"average rate: {tup4[2]}, amount at risk: {tup4[3]:.0f}"
        wsheet["H" + str(i_row + 2)] = tmp_str
        wsheet["H" + str(i_row + 2)].font = bold_font
        i_row += 5
        # store the DTD PnL and four values associated with this Quotation:
        tup_dict.update({t: [tot_tenors, tup4]})
    # Now that we have all the tuples, sort according to their tot_tenors $ amount:
    tt_slist = sorted(tup_dict.items(), key=lambda x: x[1][0], reverse=(tot_amount > 0))
    # For each selected Quotation
    for i, ttt in enumerate(tt_slist, 1):
        tup4 = ttt[1][1]       # extract the 4-tuple
        tot_tenors = ttt[1][0] # extract the Tot_tenors amount
        # complete its part of the  Commentary string. First up or down?
        dec_inc = 'increased' if  tot_tenors > 0  else 'decreased'
        # itemize with (1)..., (2)... if more than 1:
        comment += f" \n({i})" if len(top_drill) > 1 else ''
        # Fill in the PnL $ amount, Quotation, up/down direction:
        comment += f" {loc_ccy} {human_format(tot_tenors)} {ttt[0]} as vols {dec_inc} by"
        # avg_ratemove and a single Tenor:
        single_tenor = f" {tup4[2] / 100.:.2f}% on {tup4[0]} tenor"
        # avg_ratemove and the Tenor range:
        tenor_range = f" an average of {tup4[2] / 100.:.2f}% across tenors {tup4[0]} to {tup4[1]}"
        comment += single_tenor if tup4[1] == tup4[0] else tenor_range
        # get the risk:
        what_risk = get_risk(tup4[3])
        if what_risk:
            comment += f" on {what_risk} positions,"
    # When finished, replace the final comma-space with the period:
    comment = re.sub(',$', '.', comment)
    # With selected Quotations evaluate the pivot table showing the breakdown by Product Types:
    print(f"\nProduct Types for the selected {top_drill}")
    wsheet["H" + str(i_row)] = lst_unpack("Product Types for the selected", top_drill, ':')
    tmp2piv = pfcomp[pfcomp[drill_on].isin(top_drill)]
    piv_pt = tmp2piv.pivot_table(index='Product Type',
                                 values='DTD LCY',
                                 aggfunc=np.sum,
                                 margins=True,
                                 margins_name='Total').\
                                         sort_values('DTD LCY', ascending=(dtd_pl < 0))
    print(piv_pt)
    piv_pt.drop('Total', axis=0, inplace=True)
    piv_pt.reset_index(inplace=True)
    # Get the selection of top Product Types making up the 90-110% range of DTD PnL:
    top_bit = get_selection2(piv_pt, tot_amount, 'Product Type', 'DTD LCY',
                             [0.9, 0.1, 0.9, 1.1, 0.0])
    piv_pt.style.applymap(highlight(lst=top_bit)).format({"DTD LCY": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='Fx Vega', index=False, startrow=i_row, startcol=7)
    print("\nSelected Product Type(s): " + str(top_bit))
    i_row += piv_pt.shape[0] + 3
    wsheet["H" + str(i_row)] = lst_unpack("Selected Product Type(s):", top_bit)
    wsheet["H" + str(i_row)].font = bold_font
    # Set the default values to '' for the df_results columns:
    tmp_dict = {'Component': '', 'Selected': '', 'PnL': '',
                'Ccy/Instrument': '', 'Product_Type': '', 'Comment': ''}
    # Fill in the whole Fx Vega row except the 'Selected' column:
    tmp_dict['Component'] = comp
    tmp_dict['PnL'] = tot_amount
    tmp_dict['Ccy/Instrument'] = str.join(', ', top_drill)
    tmp_dict['Product_Type'] = str.join(', ', top_bit)
    tmp_dict['Comment'] = comment
    df_results = df_results.append(tmp_dict, ignore_index=True)
    return df_results


def get_ccypair(dkey, dval, ccy, loc_ccy, at_risk, quot_list):
    """
    Summary:
    -----------
    Based on one of the 3 regex patterns (see dpat in get_rates() below for 'Currency',
    'Instrument' or 'Quotation') returns the Ccy pair formatted as Quotation.

    Parameters:
    -----------
    dkey (str):        one of the 3 regex pattern in dpat dictionary in get_rates()
    dval (str):        the value from dpat[dkey], Currency', Instrument' or 'Quotation'
    ccy (str):         currency (single or pair) symbol 
    loc_ccy (str):     local currency of the DBS entity
    at_risk (number):  $ amount exposure from this currency 
    quot_list (list):  exchange rate (Quotation) list

    Returns::
    -----------
    ccy_pair (str):   currency pair in the 'AB2-XYZ' format
    what_risk (str):  'long' or 'short'
    """
    # Get the risk (long/short):
    what_risk = get_risk(at_risk)
    # Currency case: extract the currency using the regex:
    if dval == 'Currency':
        mc0 = re.match(dkey, ccy)
        xxx = mc0.group().upper()
        # if it is the USD, the move is then relative to the local currency (usually SGD):
        if xxx == 'USD':
            ccy_pair = 'USD-' + loc_ccy
            # reverse the risk, accordingly
            what_risk = get_risk(-at_risk)
        # if it is NOT the USD, get its FX rate vs USD from the Quotations:
        elif 'USD-' + xxx in quot_list:
            ccy_pair = 'USD-' + xxx
        else:
            ccy_pair = xxx + '-USD'
    # Instrument case: replace the slash with dash, so it's a Quotation case:
    if dval == 'Instrument':
        ccy_pair = ccy.replace('/', '-').upper()
    # Quotation case: simply make sure it's the uppercase:
    elif dval == 'Quotation':
        ccy_pair = ccy.upper()

    return ccy_pair, what_risk


def rate_comment(pfcc_sensi, ccy, ccy_pair, ccy_pl, the_pl, what_risk, sdates=None):
    """
    Summary:
    -----------
    Extracts the currency Quotation for the given ccy_pair and evaluates the rate
    move and returns the associated string for the commentary.

    Parameters:
    -----------
    pfcc_sensi (DataFrame):  the SENSI table prefiltered for the Fx Gamma component and Currency
    ccy (str):               the Currency/Instrument/Quotation of interest
    ccy_pair (str):          the Currency pair of interest in the "ABC-XYZ" format
    ccy_pl (number):         DTD P/L $ from this ccy_pair
    what_risk (str)          "long" or "short"
    the_pl (number):         the total P/L amount from the_comp. Or only its SIGN.
    sdates (list):           for multi-Sensi Busniness Dates includes the 1st and last date

    Returns:
    -----------
    rate_str (str):         the part of the commentary for the ccy_pair
    """
    # For the given ccy_pair extract the Sensi Date, Mkt (T), Mkt (T-1) and Rate_move values
    xcy = pfcc_sensi.query("Quotation == @ccy_pair").\
            sort_values('DTD LCY', ascending=(the_pl < 0))
    xcy = xcy[['Sensi Date', 'Mkt (T)', 'Mkt (T-1)', 'Rate_move']].drop_duplicates(keep='first')
    # split the ccy_pair to the base and counter currency, respectively:
    base_counter = ccy_pair.split('-')
    if xcy.empty:
        return ''
    # a single Sensi date:
    if not sdates:
        mkt = xcy['Mkt (T)'].iloc[0]
        mkt_1 = xcy['Mkt (T-1)'].iloc[0]
        r_move = abs(xcy['Rate_move'].iloc[0])
    # Multi-Sensi dates, compare the rates on the last and first date:
    else:
        try:
            mkt = xcy.query("`Sensi Date` == @sdates[-1]")['Mkt (T)'].iloc[0]
        except IndexError:
            mkt = xcy.query("`Sensi Date` == @sdates[0]")['Mkt (T)'].iloc[0]
        try:
            mkt_1 = xcy.query("`Sensi Date` == @sdates[0]")['Mkt (T-1)'].iloc[0]
        except IndexError:
            mkt_1 = xcy.query("`Sensi Date` == @sdates[-1]")['Mkt (T-1)'].iloc[0]
        # rate move from the 1st to the latest date:
        r_move = abs(mkt - mkt_1)
    # Commence the comment string, to comment on rate move of ccy:
    rate_str = f" {human_format(ccy_pl)} {ccy} on "
    # If the rate_move is zero, no decimal places are needed:
    if r_move == 0:
        rate_str += f"{r_move:.0f}% "
    # Otherwise, for small non-zero rate_moves, find the number of decimal places to print i:
    else:
        # the criterion as a function of the number of decimal places i:
        crit = lambda i: round(abs(r_move / mkt_1) * 100, i) > 0
        i = 1
        while not crit(i):
            i += 1
        rate_str += f"{r_move / mkt_1 * 100:.{i}f}% "
    # if USD in the pair, express the move against USD:
    if base_counter[0] == 'USD':
        wway = 'appreciation' if mkt < mkt_1 else 'depreciation'
        # swap the base and counter currencies:
        ccy0 = base_counter[1]
        ccy1 = base_counter[0]  # i.e., USD
    else:
        wway = 'depreciation' if mkt < mkt_1 else 'appreciation'
        ccy0 = base_counter[0]
        ccy1 = base_counter[1]
    # Do not print e.g. "...from ID3 on ID2 0.33% appreciation..." Rather ommit ID2.
    # Test if the first 2 characters of ccy0 are in ccy, but the whole cc0 isn't:
    crits = (ccy0[:2] in ccy) & (ccy0 not in ccy)
    # If test positive, skip the currency cc0:
    rate_str += '' if crits else f"{ccy0} " 
    rate_str += wway + f" against {ccy1}"
    # Finally complete the string with the risk kind (if any)
    rate_str += f" on {what_risk} positions," if what_risk else ","
    return rate_str


def get_rates(pf_sensi, ccy_dict, the_comp, the_pl, loc_ccy="SGD", sdates=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    For Fx Delta or Gamma Component from the selected PL Currencies, Instruments
    or Quotations if it will find their current (Mkt (T)) and previous
    (Mkt (T-1)) market rates, evaluate the associated risk and generate the
    commentary. For PL Currency the move will be relative to USD, unless the
    currency IS already USD, in which case it will be relative to the loc_ccy.
    Note that it always needs to change the regex pattern to the Quotation style,
    i.e., "BaseCcy-CounterCcy", and find the corresponding market rates in the
    'Fx Gamma' Component rows of SENSI.

    Parameters:
    -----------
    pf_sensi (DataFrame):  the SENSI table
    ccy_dict (dict):       the selected Instruments/Currencies and their $ P/L,
                           $ amounts at risk, from Delta_bdown()/Gamma_bdown()
    the_comp (str):        the Component choice, usually Fx Delta or Fx Gamma
    the_pl (number):       the total P/L amount from the_comp.
    loc_ccy (str):         the local currency for the DBS entity, usually 'SGD'
    sdates (list):         for multi-Sensi Busniness Dates includes the 1st and last date

    Returns:
    -----------
    comment (str):         the commentary for df_Result["Comment"]
    """
    quot_pat = '^([A-Za-z]+[0-9]?)(\-)([A-Za-z]+[0-9]?)$'    # e.g., 'ID2-USD'
    # The dictionary with 3 regex patterns as keys for Currency, Instrument and Quotation values:
    dpat = {}
    dpat.update({'^([A-Za-z]+[0-9]?)$': 'Currency'})                          # e.g., 'ID2'
    dpat.update({'^([A-Za-z]+[0-9]?)(\/)([A-Za-z]+[0-9]?)$': 'Instrument'})   # e.g., 'USD/ID2'
    dpat.update({quot_pat: 'Quotation'})                                      # e.g., 'ID2-USD'
    # Commence the comment string for the FXO component:
    comment = f"{sign_it(the_pl, loc_ccy)} from {the_comp}"
    comment += " where largest contribution(s) are from:"
    # The rows whose Component is equal 'Fx Gamma' shows the correct FX rates in the Quotation col:
    quot_list = pf_sensi.loc[pf_sensi["Component"] == 'Fx Gamma', 'Quotation'].\
            unique().tolist()
    # sort the ccy_dict by the PnL $ amounts:
    ccy_slist = sorted(ccy_dict.items(), key=lambda x: x[1][0], reverse=(the_pl > 0))
    for i, ccc in enumerate(ccy_slist, 1):
        ccy_tup = ccc            # a tuple ('USD-SGD', (-62016.04, 576893.42))
        ccy = ccy_tup[0]         # Currency or pair
        ccy_pl = ccy_tup[1][0]   # DTD P/L $
        at_risk = ccy_tup[1][1]  # Exposure, the amount at risk
        if len(ccy_slist) > 1:
            comment += f" \n({i})"
        # Avoid this: [p for p in dpat if re.match(p, ccy)][0] because of if/for penalty by Sonar.
        # Similarly, avoid [re.match(p, ccy) for p in dpat if re.match(p, kl)][0].group().
        # Use filter() instead to find the pattern (Currency/Instrument/Quotation) that matches ccy:
        dkey = list(filter(lambda p: re.match(p, ccy), dpat.keys()))[0]
        # Get its value (Currency/Instrument/Quotation):
        dval = dpat[dkey]
        # Get the currency pair (in the Quotation format) and the risk kind:
        ccy_pair, what_risk = get_ccypair(dkey, dval, ccy, loc_ccy, at_risk, quot_list)
        # Break the pair into the base and counter currency, respectively:
        base_counter = ccy_pair.split('-')
        # Extract the rows that carry the FX rates (i.e., Component=='Fx Gamma') for these two:
        pfcc_sensi = pf_sensi.query("(Component == 'Fx Gamma') &\
                (`PL Currency` in @base_counter)")
        # First, check if the pair cannot be found in this table:
        if ccy_pair not in pfcc_sensi['Quotation'].unique().tolist():
            # if that's the case, first try the reverse (counter-base) ccy pair:
            tmp = re.match(quot_pat, ccy_pair)
            rev_pair = str.join('', tmp.groups()[::-1])
            # second, test if a digit is a part of the first Ccy name:
            xpat = '^([A-Za-z]+)([0-9])(\-)([A-Za-z]+)$'
            xmat = re.match(xpat, ccy_pair)
            # third, test if a digit is a part of the second Ccy name:
            ypat = '^([A-Za-z]+)(\-)([A-Za-z]+)([0-9])$'
            ymat = re.match(ypat, ccy_pair)
            # first case: the reverse ccy pair is available:
            if pfcc_sensi['Quotation'].str.match(rev_pair).any():
                ccy_pair = rev_pair
                base_counter = base_counter[::-1]
                # reverse the risk accordingly
                what_risk = get_risk(-at_risk)
            # second case: a digit is found in the base currency:
            elif xmat:
                # extract the 4 groups based on the xpat regex pattern:
                xgr = xmat.groups()
                # combine the first and last group into a new pattern matching ANY digit in the base
                xst = str.join('', ('(' + xgr[0] + '[0-9])', '-', '('+ xgr[3] + ')'))
                # extract the first row matching such a Quotation pattern:
                ccy_pair = pfcc_sensi.loc[pfcc_sensi['Quotation'].str.match(xst), 'Quotation'].iloc[0]
                # break this pair into the new base and counter currency:
                base_counter = ccy_pair.split('-')
            # third case: a digit is found in the counter currency:
            elif ymat:
                # extract the 4 groups based on the xpat regex pattern:
                ygr = ymat.groups()
                # combine the first and third group into a pattern matching ANY digit in the counter
                yst = str.join('', (ygr[0], '-', ygr[2], '([0-9])'))
                # extract the first row matching such a Quotation pattern:
                ccy_pair = pfcc_sensi.loc[pfcc_sensi['Quotation'].str.match(yst), 'Quotation'].iloc[0]
                # break this pair into the new base and counter currency:
                base_counter = ccy_pair.split('-')
        # append the local currency:
        comment += ' ' + loc_ccy 
        # get the rates part of the commentary:
        comment += rate_comment(pfcc_sensi, ccy, ccy_pair, ccy_pl, the_pl, what_risk, sdates)
    # Finally, substitute the final comma-EOL with period to complete the string commentary:
    comment = re.sub('\,$', '.', comment)
    print(comment)
    return comment


def ccy_convert(fxc, df_results):
    """
    Summary:
    -----------
    Convert the rounded $ PnL amount associated with the components Fx Delta and Gamma
    from string back to the actual number, required for further processing.

    Parameters:
    -----------
    fxc (str):               either Fx Delta or Fx Gamma
    df_results (DataFrame):  the table with 4 rows (for New deals, Fx Delta/Gamma/Vega) and 5
                             columns (Component, PL, Ccy/Instrument, Product Type, Commentary)
    Returns:
    -----------
    tmp_pl (number):         $ amount from either Fx Delta or Fx Gamma
    """
    # The regex pattern matching e.g. "+S$687" or "-S$0.987":
    pattern = "([+-])?S\$([0-9]*)(\.?)([0-9]*)"
    # Extract the the PnL as a "+/-S$8799.8798" string:
    fx_pl = df_results.query("Component == @fxc")['PnL'].iloc[0]
    tmp_pl = re.match(pattern, fx_pl)
    # Join the groups that match the string (no "S$") and evaluate as a number:
    tmp_pl = eval(''.join(tmp_pl.groups()))
    # Multiply the number by their order of magnitude suffix (if any):
    if fx_pl[-1].upper() == 'K':
        tmp_pl *= 1000
    elif fx_pl[-1].upper() == 'M':
        tmp_pl *= 1e6
    return tmp_pl


def outsourced(pf_sensi, fxc, top_fxc_ccy, dtd_pl, df_results, tup4):
    """
    Summary:
    -----------
    Deals with the "Greeks" and smile if they are passed as "fxc" arrgument.
    What follows is the calculation of top Product Types for each "Greek"
    Component in the selection (typically most relevant for Fx Delta, because
    Delta_bdown() needs to be called twice, first to select the key contributing
    Currencies, and then to select the key contributing Product Types for ALL
    Currencies aggregated, but not yet for the selected Currencies only).
    To this end, first we split the SENSI table (filtered for this Component) across
    all Currencies and report the top 2 Product Types for the top Currencies.
    The latter step is repeated if there is a list of selected Currencies from
    the FX Delta breakdown. The final selection of Product Types for Fx Delta
    takes place at the very end, with the SENSI table filtered on both the
    Component (i.e. Fx Delta) and selected Currency list. The evaluated
    Product Types are then used to update the corresponding Component row
    in the df_results.

    Parameters:
    -----------
    pf_sensi (DataFrame):    SENSI table for FXO (or another desk in Default Logic)
    fxc (str):               FX Options component (Fx Delta/Gamma/Vega/smiles)
    top_fxc_ccy (list):      the selected top PL Currencies/Instruments
    dtd_pl (number):         the $ amount from the fxc component
    df_results (DataFrame):  the table with rows for New deals, Fx Delta/Gamma/Vega/smiles
    tup4 (4-tuple):          xclwrt (XlsWriter), wsheet (Worksheet), i_row (int), i_col (int)

    Returns:
    -----------
    df_results (DataFrame):  the updated table
    """
    # Evaluate the pivot table showing the breakdown by the Currency/Product Type pairs of DTD PnL:
    fxc_ptypes = pf_sensi.query("Component == @fxc").\
            pivot_table(index='PL Currency',
                        columns='Product Type',
                        values='DTD LCY',
                        aggfunc=np.sum,
                        margins=True,
                        margins_name="Total").\
                                sort_values('Total', ascending=(dtd_pl < 0))
    fxc_ptypes.columns.name = None
    fxc_ptypes.reset_index(inplace=True)
    print(f"\n The Currency-Product Type breakdown for {fxc}:")
    xclwrt, wsheet, i_row, i_col = tup4
    bold_font = Font(bold=True, size=12)
    cell = ascii_uppercase[i_col] + "1"
    print(fxc_ptypes.head(10))
    if fxc == 'Fx Delta':
        wsheet[cell] = f"Currency-Product Type breakdown for {fxc}:"
    # Extract the top 2 Product Types for each Currency:
    fxc_best_ptype = pivtab_best(pf_sensi.query("Component == @fxc"),
                                 the_idx='PL Currency',
                                 the_col='Product Type',
                                 n_best=2,
                                 dtd_pl=dtd_pl)
    print(f"\n The top 2 Product Types for the top Currencies in {fxc}:")
    print(fxc_best_ptype.head())
    # Now filter only the the pre-selected Currencies:
    tmp = fxc_best_ptype.query("`PL Currency` in @top_fxc_ccy")
    if not tmp.empty:
        print(f"\n The top 2 Product Types for the selected Currencies in {fxc}:")
        print(tmp)
        cy_lst = tmp['PL Currency'].unique().tolist()
        # if a Currency with the sign opposite to DTD_PL was dropped by pivtab_best(), restore it:
        ccys = top_fxc_ccy.copy() if len(cy_lst) < len(top_fxc_ccy) else cy_lst
        # evaluate the pivot table showing the contribution of each Product Type to DTD PnL:
        pfccp = pf_sensi.query("(Component == @fxc) & (`PL Currency` in  @ccys)").\
            pivot_table(index='Product Type',
                        values='DTD LCY',
                        aggfunc=np.sum,
                        dropna=False,
                        fill_value=0,
                        margins=True,
                        margins_name='Total').\
                                sort_values('DTD LCY', ascending=(dtd_pl < 0))
        print('\n', ccys, ':\n', pfccp.head(10), '\n')
        # extract the grand Total DTD PnL:
        tot_amount = pfccp.loc['Total', 'DTD LCY']
        pfccp.drop('Total', axis=0, inplace=True)
        pfccp.reset_index(inplace=True)
        # get the selection of top Product Types contributing the 90-110% of the Total DTD PnL:
        print(f"\nProduct Type breakdown for {fxc} considering only the Currencies in {ccys}")
        top_cpt = get_selection2(pfccp, tot_amount, 'Product Type', 'DTD LCY',
                                 [0.9, 0.1, 0.9, 1.1, 0.0])
        fxc_ptypes.set_index('PL Currency').style.apply(highlight_ser, axis=1, lst=top_fxc_ccy).\
                apply(highlight_ser, axis=0, c='orange', lst=top_cpt).\
                format({"PnL": '{:.2f}'}, na_rep="-").\
                to_excel(xclwrt, sheet_name=wsheet.title, startrow=3, startcol=i_col)
        print("\n Selected Product Types for Currencies in {ccys}: " + str(top_cpt))
        cell = ascii_uppercase[i_col] + str(fxc_ptypes.shape[0] + 9)
        wsheet[cell] = lst_unpack(f"Selected Product Types for Currencies in {*ccys,}:", top_cpt)
        wsheet[cell].font = bold_font
        # convert the list into a string to fill in the "Product Type" column of df_results:
        df_results.loc[df_results['Component'] == fxc, 'Product_Type'] = str.join(', ', top_cpt)
    print('_' * 80)


def doit_doit(pf_sensi, pf_plva, dtd_pl, xclwrt):
    """The common engine of fxo_demo() and fxo_db() below.
    Parameters:
    -----------
    pf_ sensi(DataFrame):  the IRD desk SENSI table
    pf_ plva(DataFrame):   the PLVA table
    dtd_pl (float):        the day-to-day PnL $ amount
    xclwrt (XlsWriter):    an Excel Writer object

    Returns
    -----------
    df_results (DataFrame) the table with the summary of results
    """
    str_lst = ["1. Obtain the SENSI and PLVA tables from iWork and the DTD_PL amount."]
    tmp_str = "2. Evaluate the DTD LCY breakdown of the PLVA table by Components"
    tmp_str += " and select the key Components contributing to 80-120% of the Total"
    tmp_str += " (the PLVA tab). Not required for further workflow."
    str_lst.append(tmp_str)
    tmp_str = "3. Evaluate the DTD LCY breakdown of the SENSI table by Components"
    tmp_str += " and select the key Components contributing to 80-120% of the Total."
    tmp_str += " For the 'Fx' Components and 'New deals' show the Product Type breakdown"
    tmp_str += " (the SENSI tab)."
    str_lst.append(tmp_str)
    tmp_str = "4. For New deals select the key Product Types making 80-120% of its PnL."
    tmp_str += " For this subset select the key Instruments making 80-120% of the Pnl subtotal."
    str_lst.append(tmp_str)
    tmp_str = "5. For Fx Delta select the key Currencies making 80-120% of its PnL."
    tmp_str += " For this subset select the key Product Types with 90-110% of the Pnl subtotal."
    str_lst.append(tmp_str)
    tmp_str = "6. For Fx Gamma/Smiles select the key Instruments making 80-120% of its PnL."
    tmp_str += " For this subset select the key Product Types with 90-110% of the Pnl subtotal."
    str_lst.append(tmp_str)
    tmp_str = "7. For Fx Vega select the key Quotations making 80-120% of its PnL."
    tmp_str += " For each of these Quotations select the Tenor range with 90-110% of its Pnl total."
    tmp_str += " For this subset select the key Product Types with 90-110% of the Pnl subtotal."
    str_lst.append(tmp_str)
    pd.DataFrame({'Workflow': str_lst}).\
            to_excel(xclwrt, sheet_name='FXO_Workflow', index=False, startrow=0, startcol=0)

    # A dummy DataFrame, simply to start the new 'PLVA' worksheet:
    pd.DataFrame().to_excel(xclwrt, sheet_name='PLVA', startrow=0, startcol=0)
    wsheet = xclwrt.book["PLVA"]
    wsheet.sheet_state = 'hidden'

    # Extract the local currency for the current entity:
    loc_ccy = pf_sensi['LCY'].mode().iloc[0]
    # For multi-Sensi Business Dates, the rate move is from the earliest to latest date.
    sdates = pf_sensi['Sensi Date'].unique().tolist()
    # Associate dates with their datetimes for sorting:
    try:
        sdate_dict = {d: parse(d) for d in sdates}
    except Exception:
        sdate_dict = {d: parse(str(d)) for d in sdates}
    # The dates sorted based on their datetimes:
    sdates = (sorted(sdate_dict.items(), key=lambda x: x[1], reverse=False))
    # Retain only the first and last dates:
    sdates = [sdates[0][0], sdates[-1][0]] if len(sdates) > 1 else []

    # First, from the PLVA extract the Components making up 80-120% of the DTD PnL:
    selected = pl_by_components(pf_plva, dtd_pl, "The PLVA way", xclwrt, wsheet)
    print(selected)
    print('_' * 80)
    # Then, from the SENSI extract the Components making up 80-120% of the DTD PnL:
    pd.DataFrame().to_excel(xclwrt, sheet_name='FXO_SENSI', startrow=0, startcol=0)
    wsheet = xclwrt.book["FXO_SENSI"]
    selected = pl_by_components(pf_sensi, dtd_pl, "The SENSI way", xclwrt, wsheet)
    print(selected)
    print('_' * 80)

    # The 5 Components to consider & report about each time in the df_results table:
    the_comps = ['New deals', 'Fx Delta', 'Fx Gamma', 'Fx Vega', 'Fx smiles']
    print('Product Type - Component breakdown:\n')
    wsheet["F2"] = "Product Type - Component breakdown:"
    # Obsolete: making sure Delta & Gamma go hand in hand (the list already includes both):
    the_comps = set(the_comps)
    the_comps.add('Fx Gamma') if 'Fx Delta' in the_comps else the_comps.add('Fx Delta')
    the_comps = sorted(list(the_comps))
    # Evaluate the pivot table showing the breakdown by the Product Type/Component pairs of DTD PnL:
    pt_vs_delgam = pf_sensi.pivot_table(index='Product Type',
                                        columns='Component',
                                        values='DTD LCY',
                                        aggfunc=np.sum,
                                        margins=True,
                                        margins_name='Total').\
                                                sort_values('Total', ascending=(dtd_pl < 0))
    # Append the "Total" to the column list temporarily:
    tmp = the_comps.copy()
    tmp.append("Total")
    print(pt_vs_delgam[tmp])
    pt_vs_delgam[tmp].style.format('{:.2f}', na_rep="-").\
            to_excel(xclwrt, sheet_name='FXO_SENSI', startrow=2, startcol=5)
    del tmp
    print('_' * 80)

    # Set up the output table df_results with 5 rows (from the_comps list) and these 6 columns:
    cols = ['Component', 'Selected', 'PnL', 'Product_Type', 'Ccy/Instrument', 'Comment']
    df_results = pd.DataFrame(columns=cols)
    tmp_val = ['All', '', dtd_pl, '', '',
               f"Total {sign_it(dtd_pl, loc_ccy)}"]
    tmp_dic = {c: v for  c, v in zip(cols, tmp_val)}
    df_results = df_results.append(tmp_dic, ignore_index=True)
    # Fill in the 1st row "New deals":
    plva_comps = ['New deals']
    df_results = plva_bdown(pf_plva, dtd_pl, loc_ccy, plva_comps, df_results)

    print('=' * 80)
    print('Fx Delta' * 10)
    print('-' * 80)
    delta_pl = pf_sensi.loc[pf_sensi['Component'] == 'Fx Delta', 'DTD LCY'].sum()
    pd.DataFrame().to_excel(xclwrt, sheet_name='Fx Delta', startrow=0, startcol=0)
    wsheet = xclwrt.book["Fx Delta"]
    i_row = 1
    i_col = 1
    # Combine the previous 4 vars into a single tuple argument:
    tup4 = (xclwrt, wsheet, i_row, i_col)
    # Calling delta_bdown() the first time to select Currencies and pre-fill the 2nd row "Fx Delta":
    df_results, fxc_ccy, _ = delta_bdown(pf_sensi, 'PL Currency', 'Fx Delta', delta_pl, df_results,
                                         tup4)
    top_fxc_ccy = list(fxc_ccy.keys())
    # Get the commentary string with FX rates for the selected Currencies and fill in the 'Comment':
    delta_comment = get_rates(pf_sensi, fxc_ccy, 'Fx Delta', delta_pl, loc_ccy, sdates=sdates)
    df_results.loc[df_results['Component'] == 'Fx Delta', 'Comment'] = delta_comment
    print("\nProduct Type for Fx Delta aggregated across Currencies")
    i_col = 6
    tup4 = (xclwrt, wsheet, i_row, i_col)
    # Calling delta_bdown() the second time to select Product Types and fill in the "Fx Delta" row:
    df_results, top_ptype, i_col = delta_bdown(pf_sensi, 'Product Type', 'Fx Delta', delta_pl,
                                               df_results, tup4)
    top_ptype = list(top_ptype.keys())
    print(top_ptype)
    # Finally, complete the "Fx Delta" row:
    tup4 = (xclwrt, wsheet, i_row, i_col)
    outsourced(pf_sensi, 'Fx Delta', top_fxc_ccy, delta_pl, df_results, tup4)

    print('=' * 80)
    print('Fx Gamma' * 10)
    print('-' * 80)
    gamma_pl = pf_sensi.loc[pf_sensi['Component'] == 'Fx Gamma', 'DTD LCY'].sum()
    pd.DataFrame().to_excel(xclwrt, sheet_name='Fx Gamma', startrow=0, startcol=0)
    wsheet = xclwrt.book["Fx Gamma"]
    tup4 = (xclwrt, wsheet, 0, 0)
    # Calling gamma_bdown() to select Instruments and pre-fill the 3rd row "Fx Gamma":
    df_results, fxc_ccy, i_col = gamma_bdown(pf_sensi, ['PL Currency', 'Instrument'],
                                             'Product Type', 'Fx Gamma',
                                             gamma_pl, df_results, tup4)
    top_fxc_ccy = list(fxc_ccy.keys())
    # Get the commentary string with FX rates for the selected Instruments and fill in the 'Comment'
    gamma_comment = get_rates(pf_sensi, fxc_ccy, 'Fx Gamma', gamma_pl, loc_ccy, sdates=sdates)
    df_results.loc[df_results['Component'] == 'Fx Gamma', 'Comment'] = gamma_comment
    # Finally, complete the "Fx Gamma" row:
    tup4 = (xclwrt, wsheet, 0, i_col)
    outsourced(pf_sensi, 'Fx Gamma', top_fxc_ccy, gamma_pl, df_results, tup4)

    print('=' * 80)
    print('Fx smiles' * 10)
    print('-' * 80)
    smiles_pl = pf_sensi.loc[pf_sensi['Component'] == 'Fx smiles', 'DTD LCY'].sum()
    pd.DataFrame().to_excel(xclwrt, sheet_name='Fx smiles', startrow=0, startcol=0)
    wsheet = xclwrt.book["Fx smiles"]
    tup4 = (xclwrt, wsheet, 0, 0)
    # Calling gamma_bdown() to select Instruments and pre-fill the 4th row "Fx smiles":
    df_results, fxc_ccy, i_col = gamma_bdown(pf_sensi, ['PL Currency', 'Instrument'],
                                             'Product Type', 'Fx smiles',
                                             smiles_pl, df_results, tup4)
    top_fxc_ccy = list(fxc_ccy.keys())
    # Get the commentary string with FX rates for the selected Instruments and fill in the 'Comment'
    smiles_comment = get_rates(pf_sensi, fxc_ccy, 'Fx smiles', smiles_pl, loc_ccy, sdates=sdates)
    df_results.loc[df_results['Component'] == 'Fx smiles', 'Comment'] = smiles_comment
    # Finally, complete the "Fx smiles" row:
    tup4 = (xclwrt, wsheet, 0, i_col)
    outsourced(pf_sensi, 'Fx smiles', top_fxc_ccy, smiles_pl, df_results, tup4)

    print('=' * 80)
    print('Fx Vega' * 10)
    print('-' * 80)
    vega_pl = pf_sensi.loc[pf_sensi['Component'] == 'Fx Vega', 'DTD LCY'].sum()
    # Calling vega_bdown() to select Quotations and complete the 5th row "Fx Vega":
    df_results = vega_bdown(pf_sensi, 'Quotation', 'Fx Vega', vega_pl, loc_ccy, df_results, xclwrt)
    print('_' * 80)
    return df_results, selected


def fxo_demo(node_id='FX Options', ddate='20200203', entity='DBSSG', pub_holiday=None, files=None):
    """
    Summary:
    -----------
    Executes the workflow of P&L attribution for FX Options trading desk.
    First, it obtains the pre-cleaned and pre-filtered tables pf_sensi, plexp,
    and pf_plva from get_clean_portfolio(), which reads this information either
    from a SQL query or the SQL-generated files YYYYMMDD_sensi.entity.TXT,
    YYYYMMDD_IWORK_PLVA.TXT and YYYYMMDD_PLVA.entity.TXT. Then the actual
    DTD_PL $ amount is read from the Controller's Report/Commentary by
    get_dtd_pl(). Then an insight into the SENSI vs actual PnL is obtained
    from diag_sensi_vs_pl().
    First, a Component breakdown (and selection, but ignored at present)
    is obtained from the PLVA table by pl_by_components(). This step is
    then repeated using the SENSI table, from which the selection of
    sizable Components is taken for the commentary generation.
    Next, we evaluate pivot table pt_vs_delgam, which shows the $ breakdown of
    the SENSI table across the Product Types (rows) and selected Components
    (columns), the latter augmented by the 'Total'.
    If the selected Component list happens to have an item not found in the
    "Greeks" (typically 'New deals'), PLVA breakdown follows by PLVA_bdown().
    The for-loop then will deal with the remaining "Greeks" if they are found
    in the Component selection the_Comps, i.e., Fx Delta, Gamma, Vega via the
    Delta_bdown(), Gamma_bdown() and Vega_bdown(), respectively.
    These are supposed to calculate the values for the rows (five, including the
    Fx smiles) of the df_results table and update it.

    Parameters:
    -----------
    node_id (str):    the trading desk (should be 'FX Options' here)
    ddate (str):      the Business Date in the string format, e.g. '20190430'
    entity (str):     the trading DBS entity (typically 'DBSSG')
    pub_holiday (*):  if supplied (i.e. not None), it flags the previous day was a public holiday
    files (str):      the path to the SENSI, PLVA, _IWORK_PL files

    Returns:
    -----------
    df_results (DataFrame):  the table with 5 rows (for New deals, Fx Delta/Gamma/Vega/smiles) and
                             5 columns (Component, PL, Ccy/Instrument, Product Type, Commentary)
    Example: -------------------------------------------------------------------
    $ ./demo.py -d 20200108 -e DBSSG -n FX\ Options [-p 88] -f /home/jblogg/my_data/
    """

    path = pathlib.Path().cwd() / "meta_data"
    xclfile = f"{parse(ddate).strftime('%Y-%m-%d')}_FXO.xlsx"
    xclwrt = pd.ExcelWriter(path / xclfile, mode="w", engine='openpyxl')
    bold_font = Font(bold=True, size=12)

    curdate = ddate
    ddate = parse(curdate)
    # Get the clean SENSI, PL explain and PLVA tables for the given date, entity and node/desk:
    pf_sensi, plexp, pf_plva = get_clean_portfolio(curdate=curdate,
                                                   entity=entity,
                                                   node_id=node_id,
                                                   pub_holiday=pub_holiday,
                                                   files=files)

    # Extract the total DTD PnL for the date:
    comment_raw = files + "Controller's PL Commentary_CRT IRT FXO Q1 2020.ods"
    df_comment_raw = read_ods(comment_raw, 'FXO_til_19_Mar')
    dtd_pl = get_dtd_pl(df_comment_raw, curdate, node_id)

    # Test if it is a "good" SENSI day (i.e., in agreement with PL explain):
    _, _, goodday_sensi = diag_sensi_vs_pl(pf_sensi, plexp, dtd_pl, n_rows=8)

    df_results, selected = doit_doit(pf_sensi, pf_plva, dtd_pl, xclwrt)

    # final touch: indicate if the Components are selected based on the 80-120% of Total PnL
    df_results.loc[df_results['Component'].isin(selected), 'Selected'] = 'Y'
    print("\n", df_results)
    i_row = 3
    i_col = 1
    # highlight in yellow the cells with 'Y' value:
    highlight_y = lambda x: 'background-color: yellow' if x == 'Y' else ''
    df_results.style.applymap(highlight_y).format({"PnL": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='FXO_Commentary', index=False, startrow=i_row, startcol=i_col)
    wsheet = xclwrt.book["FXO_Commentary"]
    wsheet["B1"] = "SUMMARY"
    wsheet["B1"].font = bold_font
    wsheet["B3"] = "Selection Table:"
    i_row += df_results.shape[0] + 5
    wsheet["B" + str(i_row)] = "COMMENTARY:"
    wsheet["B" + str(i_row)].font = bold_font
    comments = []
    for i, cmt in enumerate(df_results.sort_values('PnL', ascending=(dtd_pl < 0)).\
                            loc[df_results['Selected'] == 'Y', 'Comment'], 1):
        wsheet["B" + str(i_row + i)] = cmt
        wsheet["B" + str(i_row + i)].font = bold_font
        comments.append(cmt)
    xclwrt.save()
    comm_dic = {"commentary": comments, "file_name": xclfile, "components": selected}
    return comm_dic 


def fxo_db(ddate, npth=None, tup5db=None):
    """
    Summary:
    -----------
    Same as fxo_demo() above, but relying on iWork DB to extract the SENSI and PLVA data.

    Parameters:
    -----------
    ddate (str):    the Business Date in the string format, e.g. '2019-04-30'
    npth (str):     the trading desk (should be 'FX Options' here)
    tup5db (tuple): the 5-tuple carrying the user, password, IP address, port and DB name details

    Returns:
    -----------
    df_results (DataFrame): the table with 5 rows (for New deals, Fx Delta/Gamma/Vega/smiles) and 6
                            cols: Component, Selected, PL, Ccy/Instrument, Product Type, Commentary
    """

    from Common_Flow.commentary_funtions import common_function
    path = pathlib.Path().cwd() / "meta_data"
    xclfile = f"{parse(ddate).strftime('%Y-%m-%d')}_FXO.xlsx"
    xclwrt = pd.ExcelWriter(path / xclfile, mode="w", engine='openpyxl')
    bold_font = Font(bold=True, size=12)

    if not npth:
        npth = 'GFM>DBS>DBS Singapore>DBSSG>Treasury>Sales and Trading>FX Desk>FX Options>'
    entity, node_id = np.array(npth.split('>'))[[3, -2]]
    mydb = PyMyDB(tup5db)
    mysql = TheQueries(mydb, ddate, entity, node_id)
    mysql.get_node_tree_path()
    dtd_pl = mysql.get_dtd_pl()
    pf_sensi = mysql.get_sensi()
    if pf_sensi.empty:
        xclwrt.save()
        return {"commentary": [], "file_name": xclfile, "components": []}
    pf_plva = mysql.get_plva()

    df_results, selected = doit_doit(pf_sensi, pf_plva, dtd_pl, xclwrt)

    # extract prominet Components from other desks for Default Logic:
    the_comps = ['New deals', 'Fx Delta', 'Fx Gamma', 'Fx Vega', 'Fx smiles']
    extras = set(selected).difference(set(the_comps))
    for comp in extras:
        sensi_nc = pf_sensi.query("Component == @comp")
        pl_comp = sensi_nc['DTD LCY'].sum()
        comment, xclwrt = common_function(sensi=sensi_nc, component=comp, pl_com=pl_comp,
                                          writer=xclwrt, date=ddate, nodepath=npth)
        tmp_dict = {'Component': comp, 'Selected': 'Y', 'PnL': pl_comp,
                    'Ccy/Instrument': '', 'Product_Type': '', 'Comment': comment}
        df_results = df_results.append(tmp_dict, ignore_index=True)

    # final touch: indicate if the Components are selected based on the 80-120% of Total PnL
    df_results.loc[df_results['Component'].isin(selected), 'Selected'] = 'Y'
    print("\n", df_results)
    i_row = 3
    i_col = 1
    # highlight in yellow the cells with 'Y' value:
    highlight_y = lambda x: 'background-color: yellow' if x == 'Y' else ''
    df_results.style.applymap(highlight_y).format({"PnL": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='FXO_Commentary', index=False, startrow=i_row, startcol=i_col)
    wsheet = xclwrt.book["FXO_Commentary"]
    wsheet["B1"] = "SUMMARY"
    wsheet["B1"].font = bold_font
    wsheet["B3"] = "Selection Table:"
    i_row += df_results.shape[0] + 5
    wsheet["B" + str(i_row)] = "COMMENTARY:"
    wsheet["B" + str(i_row)].font = bold_font
    comments = []
    for i, cmt in enumerate(df_results.sort_values('PnL', ascending=(dtd_pl < 0)).\
                            loc[df_results['Selected'] == 'Y', 'Comment'], 1):
        wsheet["B" + str(i_row + i)] = cmt
        wsheet["B" + str(i_row + i)].font = bold_font
        comments.append(cmt)
    xclwrt.save()
    comm_dic = {"commentary": comments, "file_name": xclfile, "components": selected}
    return comm_dic


def fxo_deflog(ext_sensi, comp='Fx Delta', xclwrt=None, tup5db=None):
    """
    Summary:
    -----------
    Similar to fx_demo() above, but receieving the SENSI table from outside and returninb the
    commentary for the component required by the "Default Logic".

    Parameters:
    -----------
    ext_sensi (DataFrame):  the SENSI table from another (non FXO) desk
    comp (str):             the FXO component of interest
    xclwrt (XlsWriter):     an Excel Writer object
    tup5db (tuple):         the 5-tuple carrying the DB details

    Returns:
    -----------
    df_results (DataFrame):  the table with 1 row (either  Fx Delta/Gamma/Vega/smiles) and 6 cols:
                             Component, Selected, PL, Ccy/Instrument, Product Type, Commentary
    xclwrt (XlsWriter):      the Excel Writer object updated
    """

    str_lst = ["Obtain the SENSI table for the trading desk of interest and the FX component."]
    if comp == 'Fx Delta':
        tmp_str = "Select the key Currencies making 80-120% of its PnL."
    if (comp == 'Fx Gamma') or (comp == 'Fx smiles'):
        tmp_str = "Select the key Instruments making 80-120% of its PnL."
    if comp == 'Fx Vega':
        tmp_str = "Select the key Quotations making 80-120% of its PnL."
        tmp_str += " For each of these Quotations select the Tenor range with 90-110% of its Pnl total."
    tmp_str += " For this subset select the key Product Types with 90-110% of the Pnl subtotal."
    str_lst.append(tmp_str)
    pd.DataFrame({'Workflow': str_lst}).\
            to_excel(xclwrt, sheet_name=f'{comp} FXO_Workflow', index=False, startrow=0, startcol=0)

    # "External" (i.e., FXO Desk) SENSI needed for the FX rates:
    entity = ext_sensi['Entity'].mode().iloc[0]
    # Extract the local currency for the current entity:
    loc_ccy = ext_sensi['LCY'].mode().iloc[0]
    ddate = ext_sensi['Business Date'].mode().iloc[0]
    mydb = PyMyDB(tup5db)
    mysql = TheQueries(mydb, ddate, entity, 'FX Options')
    mysql.get_node_tree_path()
    dtd_pl = mysql.get_dtd_pl()
    pf_sensi = mysql.get_sensi()

    # A dummy DataFrame
    pd.DataFrame().to_excel(xclwrt, sheet_name=f'{comp} breakdown', startrow=0, startcol=0)
    wsheet = xclwrt.book[f'{comp} breakdown']

    if 'Rate_move' not in pf_sensi.columns:
        pf_sensi = pf_sensi.assign(Rate_move=pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"])

    # For multi-Sensi Business Dates, the rate move is from the earliest to latest date.
    sdates = pf_sensi['Sensi Date'].unique().tolist()
    # Associate dates with their datetimes for sorting:
    sdate_dict = {d: parse(str(d)) for d in sdates}
    # The dates sorted based on their datetimes:
    sdates = (sorted(sdate_dict.items(), key=lambda x: x[1], reverse=False))
    # Retain only the first and last dates:
    sdates = [sdates[0][0], sdates[-1][0]] if len(sdates) > 1 else []
    
    cols = ['Component', 'Selected', 'PnL', 'Product_Type', 'Ccy/Instrument', 'Comment']
    df_results = pd.DataFrame(columns=cols)
    tmp_val = ['All', '', dtd_pl, '', '',
               f"{sign_it(dtd_pl, loc_ccy)}"]
    # tmp_dic = {c: v for  c, v in zip(cols, tmp_val)}
    tmp_dic = {c: v for  c, v in zip(cols, ['']*6)}
    df_results = df_results.append(tmp_dic, ignore_index=True)


    i_row = 1
    i_col = 1
    if comp == 'Fx Delta':
        delta_pl = ext_sensi.loc[ext_sensi['Component'] == 'Fx Delta', 'DTD LCY'].sum()
        # Combine the previous 4 vars into a single tuple argument:
        tup4 = (xclwrt, wsheet, i_row, i_col)
        df_results, fxc_ccy, _ = delta_bdown(ext_sensi, 'PL Currency', 'Fx Delta', delta_pl,
                                             df_results, tup4)
        top_fxc_ccy = list(fxc_ccy.keys())
        delta_comment = get_rates(pf_sensi, fxc_ccy, 'Fx Delta', delta_pl, loc_ccy, sdates=sdates)
        df_results.loc[df_results['Component'] == 'Fx Delta', 'Comment'] = delta_comment
        i_col = 6
        tup4 = (xclwrt, wsheet, i_row, i_col)
        df_results, top_ptype, i_col = delta_bdown(ext_sensi, 'Product Type', 'Fx Delta', delta_pl,
                                                   df_results, tup4)
        top_ptype = list(top_ptype.keys())
        tup4 = (xclwrt, wsheet, i_row, i_col)
        outsourced(ext_sensi, 'Fx Delta', top_fxc_ccy, delta_pl, df_results, tup4)

    elif comp == 'Fx Gamma':
        gamma_pl = ext_sensi.loc[ext_sensi['Component'] == 'Fx Gamma', 'DTD LCY'].sum()
        tup4 = (xclwrt, wsheet, 0, 0)
        df_results, fxc_ccy, i_col = gamma_bdown(ext_sensi, ['PL Currency', 'Instrument'],
                                                 'Product Type', 'Fx Gamma',
                                                 gamma_pl, df_results, tup4)
        top_fxc_ccy = list(fxc_ccy.keys())
        gamma_comment = get_rates(pf_sensi, fxc_ccy, 'Fx Gamma', gamma_pl, loc_ccy, sdates=sdates)
        df_results.loc[df_results['Component'] == 'Fx Gamma', 'Comment'] = gamma_comment
        tup4 = (xclwrt, wsheet, 0, i_col)
        outsourced(ext_sensi, 'Fx Gamma', top_fxc_ccy, gamma_pl, df_results, tup4)

    elif comp == 'Fx smiles':
        smiles_pl = ext_sensi.loc[ext_sensi['Component'] == 'Fx smiles', 'DTD LCY'].sum()
        tup4 = (xclwrt, wsheet, 0, 0)
        df_results, fxc_ccy, i_col = gamma_bdown(ext_sensi, ['PL Currency', 'Instrument'],
                                                 'Product Type', 'Fx smiles',
                                                 smiles_pl, df_results, tup4)
        top_fxc_ccy = list(fxc_ccy.keys())
        smiles_comment = get_rates(pf_sensi, fxc_ccy, 'Fx smiles', smiles_pl, loc_ccy, sdates=sdates)
        df_results.loc[df_results['Component'] == 'Fx smiles', 'Comment'] = smiles_comment
        # i_col += 5
        tup4 = (xclwrt, wsheet, 0, i_col)
        outsourced(ext_sensi, 'Fx smiles', top_fxc_ccy, smiles_pl, df_results, tup4)

    elif comp == 'Fx Vega':
        vega_pl = ext_sensi.loc[ext_sensi['Component'] == 'Fx Vega', 'DTD LCY'].sum()
        df_results = vega_bdown(ext_sensi, 'Quotation', 'Fx Vega', vega_pl, loc_ccy, df_results,
                                xclwrt)

    df_results.style.format({"PnL": '{:.2f}'}, na_rep="-").\
            to_excel(xclwrt, sheet_name='FXO_Commentary', index=False, startrow=1, startcol=1)
    wsheet = xclwrt.book['FXO_Commentary']
    i_row = df_results.shape[0] + 5
    i_col = 1
    bold_font = Font(bold=True, size=12)
    wsheet["B" + str(i_row)] = "SUMMARY:"
    wsheet["B" + str(i_row)].font = bold_font
    comment = ""
    for i, cmt in enumerate(df_results.iloc[1:, -1], 1):
        wsheet["B" + str(i_row + i)] = cmt
        wsheet["B" + str(i_row + i)].font = bold_font
        comment += cmt
    xclwrt.save()
    return comment, df_results


if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--entity', '-e', default='DBSSG',
            required=False, help="The trading DBS entity, typically 'DBSSG'")
    AP.add_argument('--date', '-d',
            required=False, help="date in this format '20190430'")
    AP.add_argument('--node_id', '-n',  default='FX Options',
            required=False, help="Trading Desk (e.g. 'Credit Trading' or 'Fx Options'")
    AP.add_argument('--pub_holiday', '-p', default=False,
            required=False, help="Day after a public holiday?")
    AP.add_argument('--files', '-f',
                    required=False, help="Path to the folder with the input file(s)")
    AP.add_argument('--component', '-c', choices=['Fx Delta', 'Fx Gamma', 'Fx smiles', 'Fx Vega'],
                    required=False, help="The component for Default Logic")
    ARGS = vars(AP.parse_args())
    if ARGS['date']:
        # cdic = fxo_demo(ARGS['node_id'], ARGS['date'], ARGS['entity'], ARGS['pub_holiday'], ARGS['files'])
        cdic = fxo_db(ARGS['date'])
        print(cdic)
    elif ARGS['component']:
        sensi = pd.read_csv(ARGS['files'])
        xclwrt = pd.ExcelWriter(f"FXO_{ARGS['component']}.xlsx", engine='openpyxl')
        print(fxo_deflog(sensi, ARGS['component'], xclwrt))
