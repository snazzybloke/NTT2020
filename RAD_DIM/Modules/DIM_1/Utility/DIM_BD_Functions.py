####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: Data Investigation Module - Bonds specific Functions
This is the third script for the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is to define half a dozen functions, called from the Trigger
and the subsequently imported scripts.
This is an R-to-Python translation of the script with the same name.
"""

import pathlib
import pandas as pd
import numpy as np
import yaml
import datetime as dt
from confglob import config, cPrice, cYield, cSpread

path = pathlib.Path().cwd()
b4rad_tup = path.parts[0: path.parts.index('tdap_rad') + 1]
conf_file = '/'.join(b4rad_tup) + "/config.yaml"
with open(conf_file, 'r') as f:
    env = yaml.load(f, Loader=yaml.FullLoader)
environment_type = env['environment_type']
env = env[environment_type]

# initialize the global variable (the table of bond rates without quotation):
blankQuotation = None


def AddQuotationBD(mergedData):
    """
    Summary:
    -----------
    Called from the RMS_Flags_Function script on the current day bond data
    (mergedData).
    It creates a new QUOTATION_FINAL column with initial values from QUOTATION
    (which may have string values like "PRICE" "YIELD" "SPREAD", "", nan etc).
    Then, based on string matches of RATELABEL column values, the values are
    either retained of replaced by cPrice, cYield, cSpread.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedData : DataFrame
        current day bond data

    Returns:
    -----------
    cp_mergedData : DataFrame
        a copy of the input table with QUOTATION_FINAL column added as
        the final bond quotation
    """

    # Python will modify the original DataFrame (unlike R), make a copy:
    cp_mergedData = mergedData.copy()
    if config['is_debug']:
        mergedData.to_csv("mergedData.csv", index=False)
    cp_mergedData['QUOTATION_FINAL'] = cp_mergedData['QUOTATION']
    for c in (cPrice, cYield, cSpread):
        cp_mergedData['QUOTATION_FINAL'].mask(cp_mergedData['RATELABEL'].str.contains(c),
                other=c, inplace=True)

    cp_mergedData['QUOTATION_FINAL'] = np.where(cp_mergedData['QUOTATION_FINAL'] == "P",
            cPrice, np.where(cp_mergedData['QUOTATION_FINAL'] == "Y",
                cYield, cp_mergedData['QUOTATION_FINAL']))
    return cp_mergedData


def SingleAskBidColumnBD(mergedData):
    """
    Summary:
    -----------
    Called from the Trigger and RMS_Flags_Function scripts on the current day
    bond data (mergedData).
    Creates a single column ASKNEW based on ASK,ASK1,ASK2 values for ASK prices
    and, similary, BIDNEW and MIDNEW for BID and MID, respectively.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedData : DataFrame
        current day bond data

    Returns:
    -----------
    appendInstruments : DataFrame
        current day bond data with ASKNEW, BIDNEW and MIDNEW columns
    """

    crit1 = "(INSTRUMENTTYPE == 'BD_CORP') & (QUOTATION_FINAL != '') & QUOTATION_FINAL.notna()"
    BD_CORP = mergedData.query(crit1, engine="python")

    tmp = np.where(BD_CORP['QUOTATION_FINAL'] == cPrice, BD_CORP['ASK'],\
            np.where(BD_CORP['QUOTATION_FINAL'] == cSpread, BD_CORP['ASK1'],\
                    np.where(BD_CORP['QUOTATION_FINAL'] == cYield, BD_CORP['ASK2'], BD_CORP['ASK'])
                    )
            )
    BD_CORP = BD_CORP.assign(ASKNEW=tmp)
    if BD_CORP['ASKNEW'].dtype == object:
        BD_CORP['ASKNEW'] = pd.to_numeric(BD_CORP['ASKNEW'], errors='coerce')

    tmp = np.where(BD_CORP['QUOTATION_FINAL'] == cPrice, BD_CORP['BID'],\
            np.where(BD_CORP['QUOTATION_FINAL'] == cSpread, BD_CORP['BID1'],\
                    np.where(BD_CORP['QUOTATION_FINAL'] == cYield, BD_CORP['BID2'], BD_CORP['BID'])
                    )
            )
    BD_CORP = BD_CORP.assign(BIDNEW=tmp)
    if BD_CORP['BIDNEW'].dtype == object:
        BD_CORP['BIDNEW'] = pd.to_numeric(BD_CORP['BIDNEW'], errors='coerce')

    tmp = np.where(BD_CORP['QUOTATION_FINAL'] == cPrice, BD_CORP['MID'],\
            np.where(BD_CORP['QUOTATION_FINAL'] == cSpread, BD_CORP['MID1'],\
                    np.where(BD_CORP['QUOTATION_FINAL'] == cYield, BD_CORP['MID2'], BD_CORP['MID'])
                    )
            )
    BD_CORP = BD_CORP.assign(MIDNEW=tmp)
    if BD_CORP['MIDNEW'].dtype == object:
        BD_CORP['MIDNEW'] = pd.to_numeric(BD_CORP['MIDNEW'], errors='coerce')

    crit2 = "(INSTRUMENTTYPE == 'BD_GOVT') & (QUOTATION_FINAL != '') & QUOTATION_FINAL.notna()"
    BD_GOVT = mergedData.query(crit2, engine="python")
    BD_GOVT = BD_GOVT.assign(ASKNEW=BD_GOVT['ASK'], BIDNEW=BD_GOVT['BID'], MIDNEW=BD_GOVT['MID'])

    OTHERS = mergedData.query("(INSTRUMENTTYPE != 'BD_GOVT') & (INSTRUMENTTYPE != 'BD_CORP')")
    tmp = np.where(OTHERS['QUOTATION_FINAL'] == cPrice, OTHERS['ASK'],\
            np.where(OTHERS['QUOTATION_FINAL'] == cSpread, OTHERS['ASK1'],\
                    np.where(OTHERS['QUOTATION_FINAL'] == cYield, OTHERS['ASK2'], OTHERS['ASK'])
                    )
            )
    OTHERS = OTHERS.assign(ASKNEW=tmp)
    if OTHERS['ASKNEW'].dtype == object:
        OTHERS['ASKNEW'] = pd.to_numeric(OTHERS['ASKNEW'], errors='coerce')

    tmp = np.where(OTHERS['QUOTATION_FINAL'] == cPrice, OTHERS['BID'],\
            np.where(OTHERS['QUOTATION_FINAL'] == cSpread, OTHERS['BID1'],\
                    np.where(OTHERS['QUOTATION_FINAL'] == cYield, OTHERS['BID2'], OTHERS['BID'])
                    )
            )
    OTHERS = OTHERS.assign(BIDNEW=tmp)
    if OTHERS['BIDNEW'].dtype == object:
        OTHERS['BIDNEW'] = pd.to_numeric(OTHERS['BIDNEW'], errors='coerce')

    tmp = np.where(OTHERS['QUOTATION_FINAL'] == cPrice, OTHERS['MID'],\
            np.where(OTHERS['QUOTATION_FINAL'] == cSpread, OTHERS['MID1'],\
                    np.where(OTHERS['QUOTATION_FINAL'] == cYield, OTHERS['MID2'], OTHERS['MID'])
                    )
            )
    OTHERS = OTHERS.assign(MIDNEW=tmp)
    if OTHERS['MIDNEW'].dtype == object:
        OTHERS['MIDNEW'] = pd.to_numeric(OTHERS['MIDNEW'], errors='coerce')

    appendInstruments = pd.concat([BD_CORP, BD_GOVT, OTHERS], ignore_index=True)
    global blankQuotation
    blankQuotation = mergedData.query("(QUOTATION_FINAL == '') | QUOTATION_FINAL.isna()", engine="python")
    return appendInstruments


def IdentifyMaturedRatesBD(currentDayData):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day bond data (currentDayData).
    It creates a new MATURED column with binary (0 or 1 if matured) values based
    on available/missing MATURITY values their relation with RECORDDATE values.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayData : DataFrame
        current day bond data

    Returns:
    -----------
    currentDayData : DataFrame
        the input table with MATURED column added
    """

    tmp = np.where(currentDayData['MATURITY'].isna(), 0,
            np.where(currentDayData['RECORDDATE'] == currentDayData['MATURITY'],
                1, 0)
            )
    currentDayData = currentDayData.assign(MATURED=tmp)
    return currentDayData


def IdentifyMaturingRatesBD(currentDayData):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day bond data (currentDayData).
    It creates a new MATURING column with binary (0 or 1 if maturing) values based
    on the difference between MATURITY and RECORDDATE values.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayData : DataFrame
        current day bond data

    Returns:
    -----------
    currentDayData : DataFrame
        the input table with MATURING column added
    """

    if currentDayData['RECORDDATE'].dtype == object:
        currentDayData['RECORDDATE'] = pd.to_datetime(currentDayData['RECORDDATE'])
    if currentDayData['MATURITY'].dtype == object:
        currentDayData['MATURITY'] = pd.to_datetime(currentDayData['MATURITY'])
    tmp = np.where(currentDayData['MATURITY'].isna(), 0,
            np.where(np.abs(currentDayData['RECORDDATE'] -
                            currentDayData['MATURITY']).dt.days == 1,
                    1, 0)
            )
    currentDayData = currentDayData.assign(MATURING=tmp)
    return currentDayData


def getMatureRatesBD(mergedDataBD):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day bond data (mergedDataBD)
    with MATURED and MATURING flags.
    It creates a new MATURING column with binary (0 or 1 if maturing) values based
    on the difference between MATURITY and RECORDDATE values.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedDataBD : DataFrame
        current day bond data

    Returns:
    -----------
    mergedDataBD : DataFrame
        the input table with either MATURED or MATURING values 1
    """

    matureBD = mergedDataBD.query("(MATURED == 1) | (MATURING == 1)")
    return matureBD


def removeMatureBD(mergedDataBD):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day bond data (mergedDataBD)
    with MATURED and MATURING flags.
    It simply filters the rows that are neither matured nor maturing.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedDataBD : DataFrame
        current day bond data

    Returns:
    -----------
    mergedDataBD : DataFrame
        the input table with both MATURED or MATURING values 0
    """

    mergedDataBD = mergedDataBD.query("(MATURED == 0) & (MATURING == 0)")
    mergedDataBD.drop(columns=['MATURED', 'MATURING'], inplace=True)
    return mergedDataBD
