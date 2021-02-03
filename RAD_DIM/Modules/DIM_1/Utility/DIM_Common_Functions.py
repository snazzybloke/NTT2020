####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: DIM Common Functions
This is the second script for the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is to define two dozen common functions, called from the Trigger
and the subsequently imported scripts.
This is an R-to-Python translation of the script with the same name.
"""

import pathlib
import pandas as pd
import numpy as np
import yaml
import datetime as dt
from dateutil.parser import parse
from confglob import acceptableNumDaysStaleCheck, staleValueNumDays, decimalPlace,\
        staleValueThreshold

path = pathlib.Path().cwd()
b4rad_tup = path.parts[0: path.parts.index('tdap_rad') + 1]
conf_file = '/'.join(b4rad_tup) + "/config.yaml"
with open(conf_file, 'r') as f:
    env = yaml.load(f, Loader=yaml.FullLoader)
environment_type = env['environment_type']
env = env[environment_type]
outputPath = env['outputPath'] 


def RemoveWeekends(data):
    """
    Summary:
    -----------
    Invoked from the Trigger script on the reliable historical clean data,
    typically prefiltered for the chosen ASSETTYPE (e.g. Bond, FX, IR).
    It simply identifies weekends in the RECORDDATE column and filters
    those out of the table.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    data : DataFrame
        reliable historical clean data

    Returns:
    -----------
    df : DataFrame
        the input table devoid of Saturdays and Sundays
    """

    df = data.assign(weekday=data['RECORDDATE'])
    df['weekday'] = pd.to_datetime(df['weekday'])
    df['weekday'] = df['weekday'].dt.day_name()
    return df.query("weekday not in ['Saturday', 'Sunday']").drop(columns='weekday')


def IdentifyNewRates(currentDayData, relHistorical):
    """
    Summary:
    -----------
    Invoked from the Trigger script on the current day unclean data
    (currentDayData) and reliable historical clean data (relHistorical).
    It simply identifies the rows in currentDayData with RELIABLEID values
    not seen in relHistorical and assigns a new column NEW_RATE values 1
    (or 0 if already seen in relHistorical).
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayData : DataFrame
        current day unclean data
    relHistorical : DataFrame
        reliable historical data

    Returns:
    -----------
    currentDayData : DataFrame
        updated current day unclean data
    """

    relHistRates = relHistorical['RELIABLEID'].unique().tolist()
    # bool_ser = currentDayData['RELIABLEID'].isin(relHistRates)
    # bool_ser = [0 if v else 1 for v in bool_ser]
    # currentDayData = currentDayData.assign(NEW_RATE=np.where(bool_ser,
    #     np.zeros_like(bool_ser, dtype=int), np.ones_like(bool_ser, dtype=int)))
    currentDayData['NEW_RATE'] = currentDayData.eval("RELIABLEID not in @relHistRates").astype(int)
    return currentDayData


def NewRatesSubset(currentDayData, currDayReliable):
    """
    Summary:
    -----------
    Invoked from the Trigger script on the current day unclean data
    (currentDayData) and current day golden rates data (currDayReliable).
    The rows with NEW_RATE values of 1 from the former and merged on
    RELIABLEID with the latter into a new table.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayData : DataFrame
        current day unclean data
    currDayReliable : DataFrame
        current day golden rates data
    
    Returns:
    -----------
    mergeReliable : DataFrame
        updated current day unclean data
    """

    newRates = currentDayData.query("NEW_RATE == 1")
    mergeReliable = pd.merge(newRates, currDayReliable, on='RELIABLEID', suffixes=('_new', '_cur'))
    # drop rows both for missing and empty string RATELABEL:
    mergeReliable.dropna(axis=0, subset=['RATELABEL'], inplace=True)
    bool_ser = mergeReliable['RATELABEL'] == ""
    mergeReliable.drop(bool_ser[bool_ser].index, axis=0, inplace=True)

    cs = ["RELIABLEID", "RECORDDATE_cur", "RATELABEL", "ASSETTYPE_new",
          "INSTRUMENTTYPE_new", "PRODUCTTYPE", "ASK_cur", "BID_cur", "ASK1_cur",
          "BID1_cur", "ASK2_cur", "BID2_cur", "LAST_cur", "CLOSING_cur",
          "HISTORICCLOSE_cur", "MID_cur", "MID1_cur", "MID2_cur", "CURRENCY",
          "TENOR", "MXCURRENCY", "MXGENTYPE", "ISINCODE", "MARKET", "FAMARKET",
          "RATETYPE_cur", "QUOTATION", "YIELDCURVELABEL", "MATURITY", "SMARKET",
          "NEW_RATE"]
    mergeReliable = mergeReliable[cs]
    tmp_dic = {c: c[:-4] for c in cs if "_cur" in c}
    mergeReliable.rename(tmp_dic, axis=1, inplace=True)
    return mergeReliable


def RatesForChecksSubset(currentDayData):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data
    (currentDayData).
    The rows with NEW_RATE values of 0 or missing are filtered and
    returned, while dropping the NEW_RATE column in the end.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayData : DataFrame
        current day unclean data
    
    Returns:
    -----------
    ratesForChecks : DataFrame
        current day unclean data filtered for NEW_RATE == 0 or missing values
        and finally NEW_RATE column dropped
    """

    # filter the NEW_RATE zero and missing (na) values:
    ratesForChecks = currentDayData.query("(NEW_RATE == 0) or (NEW_RATE != NEW_RATE)")
    ratesForChecks.drop(columns='NEW_RATE', inplace=True)
    return ratesForChecks


def GetProperties(ratesForChecks, reliableData):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data
    (ratesForChecks) and historical golden rates data (reliableData).
    It merges the two tables on RATELABEL values, retains old ASSETTYPE,
    INSTRUMENTTYPE and RATETYPE values from the former, takes on new values
    for RECORDDATE, MID, MID1, and MID2 from the latter, and a set of column
    values from either.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    ratesForChecks : DataFrame
        current day unclean data
    reliableData : DataFrame
        historical golden rates data
    
    Returns:
    -----------
    mergeReliable : DataFrame
        current day unclean data with new columns ("properties") added
        from reliableData
    """

    mergeReliable = pd.merge(ratesForChecks, reliableData, on='RELIABLEID', suffixes=('_cur', '_prev'))
    # drop rows both for missing and empty string RATELABEL:
    mergeReliable.dropna(axis=0, subset=['RATELABEL'], inplace=True)
    bool_ser = mergeReliable['RATELABEL'] == ""
    mergeReliable.drop(bool_ser[bool_ser].index, axis=0, inplace=True)

    cs = ["RELIABLEID", "RECORDDATE_cur", "RATELABEL", "ASSETTYPE_prev",
          "INSTRUMENTTYPE_prev", "PRODUCTTYPE", "ASK", "BID", "ASK1",
          "BID1", "ASK2", "BID2", "LAST", "CLOSING",
          "HISTORICCLOSE", "MID_cur", "MID1_cur", "MID2_cur", "CURRENCY",
          "TENOR", "MXCURRENCY", "MXGENTYPE", "ISINCODE", "MARKET", "FAMARKET",
          "RATETYPE_prev", "QUOTATION", "YIELDCURVELABEL", "MATURITY", "SMARKET"]
    mergeReliable = mergeReliable[cs]
    tmp_dic = {c: c[:-5] for c in cs if "_prev" in c}
    mergeReliable.rename(tmp_dic, axis=1, inplace=True)
    tmp_dic = {c: c[:-4] for c in cs if "_cur" in c}
    mergeReliable.rename(tmp_dic, axis=1, inplace=True)
    return mergeReliable


def FilterRawData(relHistorical, curDate):
    """
    Summary:
    -----------
    Called from the Trigger script on reliable historical clean data
    and (relHistorical) the current date of interest (curDate)
    It appears that the purpose is to filter the rows of the former which
    are in the range between curDate-acceptableNumDaysStaleCheck and curDate,
    split the rows based on their RELIABLEID/RECORDDATE values, and extract
    the latest staleValueNumDays rows from each group.
    NOTE: it seems the author confused relHistorical and relHistoricalForStale
    and assigned the latter TWICE based on the former with unitentional effects.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    relHistorical : DataFrame
        reliable historical clean data
    curDate : str (datetime also acceptable)
        current date of interest ("current day unclean data", says the R code)
    
    Returns:
    -----------
    relHistoricalForStale : DataFrame
        historical clean data with at least "staleValueNumDays" data points
        in the last "acceptableNumDaysStaleCheck" days
    """

    if type(curDate) == str:
        curDate = parse(curDate).date()
    previousYearDate = curDate - dt.timedelta(acceptableNumDaysStaleCheck)
    if relHistorical['RECORDDATE'].dtype == object:
        relHistorical['RECORDDATE'] = pd.to_datetime(relHistorical['RECORDDATE'])
    # NOTE: relHistoricalForStale assigned TWICE from relHistorical. What gives!?
    relHistoricalForStale = relHistorical.query("(RECORDDATE >= @previousYearDate) and (RECORDDATE < @curDate)")
    relHistoricalForStale = relHistorical.\
            sort_values(['RELIABLEID', 'RECORDDATE'], ascending=[True, False]).\
                    groupby('RELIABLEID').head(staleValueNumDays)
    return relHistoricalForStale


def RemoveExpiredFromHist(ratesForChecks, relHistoricalForStale):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data
    (ratesForChecks) and reliable historical clean data (relHistoricalForStale). 
    The purpose is to filter only the rows of the latter which share the
    common RELIABLEID values found in the former.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    ratesForChecks : DataFrame
        current day unclean data
    relHistoricalForStale : DataFrame
        reliable historical clean data
    
    Returns:
    -----------
    relHistoricalForStale : DataFrame
        reliable historical clean data with rows filtered only for
        RELIABLEID values which can be found in ratesForChecks
    """

    ratesActive = ratesForChecks['RELIABLEID'].unique().tolist()
    relHistoricalForStale = relHistoricalForStale.query("RELIABLEID in @ratesActive")
    return relHistoricalForStale


def helper1(x):
    """
    A simple helper function called by StaleCheck() below to help evaluate the
    Coefficients of Variation (CV) for 3 columns (ASK/BID/MID) for a frame-group
    x and return the values as a 3-column DataFrame.
    """

    tmp_dic = {}
    for abm in ['ASKNEW', 'BIDNEW', 'MIDNEW']:
        # NOTE: the original R func StaleCheck does not use the optional na.rm
        # flags, i.e.  sd(..., na.rm = T) / mean(..., na.rm = T)
        # but very likely it should, because this way it easily produces nan
        # tmp_dic.update({abm[:3] + "_CV_PCT": np.abs(x[abm].std() / x[abm].mean()) * 100})
        # to bring in line with the R-code:
        tmp_dic.update({abm[:3] + "_CV_PCT": np.abs(x[abm].std(skipna=False) / x[abm].mean(skipna=False)) * 100})
    return pd.DataFrame([tmp_dic])


def StaleCheck(relHistoricalForStale, ratesForChecks, staleValueThreshold):
    """
    Summary:
    -----------
    Called from the Trigger script on the reliable historical clean data
    (relHistoricalForStale), current day unclean data (ratesForChecks) and
    coefficient of variation stale value threshold (staleValueThreshold).
    The purpose is to generate binary (0 or 1) flags in the current day
    unclean data for the ASK, BID and MID columns, depending on whether
    their coefficent of variation has exceeded the threshold or not.
    
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    ratesForChecks : DataFrame
        current day unclean data
    relHistoricalForStale : DataFrame
        reliable historical clean data
    staleValueThreshold : number
        stale value threshold of cv
    
    Returns:
    -----------
    staleValueflags : DataFrame
        current day data with stale flags, an aggregated table (on RELIABLEID
        values) with 6 new columns ('..._CV_PCT' & 'STALE_...' for ASK/BID/MID)
    """

    tmp = relHistoricalForStale.sort_values(['RELIABLEID', 'RECORDDATE'],
            ascending=[True, False]).groupby('RELIABLEID')\
                    .filter(lambda x: len(x) >= staleValueNumDays)
    staleValueCheckData = tmp.groupby('RELIABLEID').head(staleValueNumDays)
    windowDataAgg = staleValueCheckData.append(ratesForChecks, ignore_index=True)
    tmp = windowDataAgg.groupby('RELIABLEID').filter(lambda x: len(x) >= staleValueNumDays + 1)
    windowDataAgg = tmp.groupby('RELIABLEID').apply(helper1).round(decimalPlace)
    try:
        staleValueflags = pd.merge(ratesForChecks,
                                   windowDataAgg.reset_index().drop(columns='level_1'),
                                   on = 'RELIABLEID', how='left')
        staleValueflags = staleValueflags.assign(STALE_ASK=
                np.where(staleValueflags['ASK_CV_PCT'] <= staleValueThreshold, 1, 0))
        staleValueflags = staleValueflags.assign(STALE_BID=
                np.where(staleValueflags['BID_CV_PCT'] <= staleValueThreshold, 1, 0))
        staleValueflags = staleValueflags.assign(STALE_MID=
                np.where(staleValueflags['MID_CV_PCT'] <= staleValueThreshold, 1, 0))
        # To bring it in agreement with the R code (a side effect of ifelse):
        staleValueflags['STALE_ASK'].where(staleValueflags['ASK_CV_PCT'].notna(),
                other=np.nan, inplace=True)
        staleValueflags['STALE_BID'].where(staleValueflags['BID_CV_PCT'].notna(),
                other=np.nan, inplace=True)
        staleValueflags['STALE_MID'].where(staleValueflags['MID_CV_PCT'].notna(),
                other=np.nan, inplace=True)
    except Exception:
        extras = pd.Index(['ASK_CV_PCT', 'BID_CV_PCT', 'MID_CV_PCT',
            'STALE_ASK', 'STALE_BID', 'STALE_MID'])
        staleValueflags = ratesForChecks.reindex(
                ratesForChecks.columns.union(extras), axis=1)
    return staleValueflags


def FlagCreation(ratesForChecks):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data
    (ratesForChecks).
    The purpose is to generate a binary (0 or 1) flag DIM_VIOLATION
    depending on any/all of the values of 3 "...CHECK" columns.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    ratesForChecks : DataFrame
        current day unclean data
    
    Returns:
    -----------
    ratesForChecks : DataFrame
        current day unclean data updated with the new DIM_VIOLATION column
    """

    ratesForChecks['DIM_VIOLATION'] = ratesForChecks\
            .eval("(ZERO_CHECK + NEGATIVE_CHECK + VALID_BIDASK_CHECK) >= 1").astype(int)
    return ratesForChecks


def staleDays(Price):
    """
    Summary:
    -----------
    This is an R-to-Python translation of the function with the same name.
    """

    day = 0
    for countDay in range(2, len(Price)+1):
        x = Price[:countDay]
        cv = np.abs(np.std(x, ddof=1) / np.mean(x) * 100)
        # if np.isnan(cv) | (cv > staleValueThreshold):  # problems when cv ~1.e-6
        # enforce R-code behaviour, good enough if cv is close to Threshold:
        if np.isnan(cv) | ~np.isclose(cv - staleValueThreshold, 0):
            return day
        else:
            day = countDay
    return day


def helper2(x):
    """
    A simple helper function called by Stale_Counter() below to help evaluate
    the number of stale days, by calling the function staleDays() above, for 3
    columns (ASK/BID/MID) of a frame-group x and returning the values as a
    3-column DataFrame.
    """

    tmp_dic = {}
    for abm in ['ASKNEW', 'BIDNEW', 'MIDNEW']:
        tmp_dic.update({"STALE_COUNTER_" + abm[:3]: staleDays(x[abm])})
    return pd.DataFrame([tmp_dic])


def Stale_Counter(ratesForChecks, relHistorical):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data
    (ratesForChecks) and reliable historical clean data (relHistorical).
    The purpose is count the number of days over which a rate has been stale.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    ratesForChecks : DataFrame
         current day's input rates
    relHistoricalForStale : DataFrame
        reliable historical clean data
    
    Returns:
    -----------
    ratesForChecks : DataFrame
         current day's rates with added column indicating stale counter
    """

    appendHistorical = pd.concat([ratesForChecks, relHistorical], ignore_index=True)
    appendHistorical['MATURITY'] = appendHistorical['MATURITY'].astype(str)
    staleTest = appendHistorical.sort_values(['RELIABLEID', 'RECORDDATE'], ascending=[True, False])\
                                .groupby('RELIABLEID').apply(helper2)
    try:
        ratesForChecks = pd.merge(ratesForChecks,
                staleTest.reset_index().drop(columns='level_1'), on = 'RELIABLEID', how='left')
    except Exception:
        extras = pd.Index(['STALE_COUNTER_ASK', 'STALE_COUNTER_BID', 'STALE_COUNTER_MID'])
        ratesForChecks = ratesForChecks.reindex(
                ratesForChecks.columns.union(extras), axis=1)
    return ratesForChecks


def SetFormat(currentDayRates, rmsThresholds):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day unclean data with flags
    (currentDayRates) and RMS thresholds for valid rates (rmsThresholds).
    The purpose is to set the format of the former as required for DIM Output.
    To this end it is first merged with a subset of columns of the latter.
    The 3 copies of it are then having columns rearranged/renamed (for
    ASK, BID and MID) and eventually are concatenated
    back into currentDayRates and returned as such.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayRates : DataFrame
        current day unclean data with flags
    rmsThresholds : DataFrame
        RMS thresholds for valid rates
    
    Returns:
    -----------
    currentDayRates : DataFrame
        current day unclean data with flags updated in required format
    """

    if currentDayRates['ASKNEW'].dtype == object:
        currentDayRates['ASKNEW'] = pd.to_numeric(currentDayRates['ASKNEW'], errors='coerce')
    if currentDayRates['BIDNEW'].dtype == object:
        currentDayRates['BIDNEW'] = pd.to_numeric(currentDayRates['BIDNEW'], errors='coerce')
    if currentDayRates['MIDNEW'].dtype == object:
        currentDayRates['MIDNEW'] = pd.to_numeric(currentDayRates['MIDNEW'], errors='coerce')
    currentDayRates.drop(currentDayRates.filter(like='Single').columns, axis=1, inplace=True)

    col = ['RATELABEL', 'AskDIMInRMS', 'BidDIMInRMS', 'MidDIMInRMS', 'ThreshASK', 'ThreshBID',
           'ThreshMID', 'Override', 'ABS TOLERANCE BID', 'ABS TOLERANCE ASK', 'ABS TOLERANCE MID',
           'REL TOLERANCE ASK', 'REL TOLERANCE BID', 'REL TOLERANCE MID']
    rmsThresholds = rmsThresholds[col]
    currentDayRates = pd.merge(currentDayRates, rmsThresholds, on='RATELABEL', how="left")
    tempAsk = currentDayRates.copy()
    tempBid = currentDayRates.copy()
    tempMid = currentDayRates.copy()

    coa = ["BIDNEW", "BID_CV_PCT", "STALE_BID", "STALE_COUNTER_BID", "MIDNEW", "MID_CV_PCT",
           "STALE_MID", "STALE_COUNTER_MID", "BidDIMInRMS", "MidDIMInRMS", "ThreshBID", "ThreshMID"]
    tempAsk.drop(coa, axis=1, inplace=True)
    tempAsk = tempAsk.assign(ASK_OR_BID="ASK")
    rn_dict = {"ASKNEW": "VALUE", "ASK_CV_PCT": "CV_PCT", "STALE_ASK": "STALE_CHECK",
               "STALE_COUNTER_ASK": "STALE_COUNTER", "AskDIMInRMS": "DIMChecksInRMS",
               "ThreshASK": "ThChecksInRMS"}
    tempAsk.rename(rn_dict, axis=1, inplace=True)

    # cob = [c.replace("BID", "ASK") for c in coa]  # ok, but no need to re-run
    cob = ['ASKNEW', 'ASK_CV_PCT', 'STALE_ASK', 'STALE_COUNTER_ASK', 'MIDNEW', 'MID_CV_PCT',
           'STALE_MID', 'STALE_COUNTER_MID', 'AskDIMInRMS', 'MidDIMInRMS', 'ThreshASK', 'ThreshMID']
    tempBid.drop(cob, axis=1, inplace=True)
    tempBid = tempBid.assign(ASK_OR_BID="BID")
    # rn_dict = {k.replace("ASK", "BID").replace("Ask", "Bid"): v for k, v in rn_dict.items()} # ok
    rn_dict = {'BIDNEW': 'VALUE', 'BID_CV_PCT': 'CV_PCT', 'STALE_BID': 'STALE_CHECK',
               'STALE_COUNTER_BID': 'STALE_COUNTER', 'BidDIMInRMS': 'DIMChecksInRMS',
               'ThreshBID': 'ThChecksInRMS'}
    tempBid.rename(rn_dict, axis=1, inplace=True)

    coc = ["ASKNEW", "ASK_CV_PCT", "STALE_ASK", "STALE_COUNTER_ASK", "BIDNEW", "BID_CV_PCT",
           "STALE_BID", "STALE_COUNTER_BID", "BidDIMInRMS", "AskDIMInRMS", "ThreshASK", "ThreshBID"]
    tempMid.drop(coc, axis=1, inplace=True)
    tempMid = tempMid.assign(ASK_OR_BID="MID")
    rn_dict = {'MIDNEW': 'VALUE', 'MID_CV_PCT': 'CV_PCT', 'STALE_MID': 'STALE_CHECK',
               'STALE_COUNTER_MID': 'STALE_COUNTER', 'MidDIMInRMS': 'DIMChecksInRMS',
               'ThreshMID': 'ThChecksInRMS'}
    tempMid.rename(rn_dict, axis=1, inplace=True)

    currentDayRates = pd.concat([tempAsk, tempBid, tempMid], ignore_index=True)
    currentDayRates['VALUE'] = currentDayRates['VALUE'].round(decimalPlace)
    currentDayRates['CV_PCT'] = currentDayRates['CV_PCT'].round(decimalPlace)
    return currentDayRates


def rmsdimFlags(DIM_OUTPUT, violation_data):
    """
    Summary:
    -----------
    Called from the Trigger script on the DIM output data (DIM_OUTPUT) and
    RMS flag data from daily violation reports (violation_data).
    (currentDayRates) and RMS thresholds for valid rates (rmsThresholds).
    The purpose is to incorporate the RMS flags in required output format.
    To this end the two tables are first merged on values of 3 columns.
    Then 3 "..._CHECK" column values are set to zero if (i) the column is
    not already there in the merged table, or (2) the values are missing.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    DIM_OUTPUT : DataFrame
        DIM output data
    violation_data : DataFrame
        RMS flag data from daily violation reports
    
    Returns:
    -----------
    DIM_OUTPUT : DataFrame
        DIM output data updated with RMS flags in the required format
    """

    DIM_OUTPUT = pd.merge(DIM_OUTPUT, violation_data, on=["RECORDDATE", "RELIABLEID", "ASK_OR_BID"],
                          how='left').sort_values("RELIABLEID", ignore_index=True)
    for c in ("ZERO_CHECK", "NEGATIVE_CHECK", "VALID_BIDASK_CHECK", "AR_CHECK", "RR_CHECK"):
        if (c not in DIM_OUTPUT.columns):
            DIM_OUTPUT[c] = 0
        else: # missing values because of outer merge
            DIM_OUTPUT[c].fillna(0, inplace=True)

    return DIM_OUTPUT


def TP_FP_Flag(DIM_OUTPUT):
    """
    Summary:
    -----------
    Called from the Trigger script on the DIM output data (DIM_OUTPUT).
    The purpose is to incorporate the RMS flags in required output format.
    To this end it looks for a change in value from unclean to clean (i.e.,
    VALUE, VIOLATION_VALUE and CLEAN_VALUE) to set the TP_FP flag as 
    "YES"/"NO"/"".
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    DIM_OUTPUT : DataFrame
        DIM output data
    
    Returns:
    -----------
    DIM_OUTPUT : DataFrame
        DIM output data updated with TP_FP flags in the required format
    """

    # NOTE: comparing floats with "==", unwise (e.g. if different float sizes/accuracies used):
    DIM_OUTPUT['MATCH'] = np.where(DIM_OUTPUT['VALUE'] == DIM_OUTPUT['VIOLATION_VALUE'],
            "YES", "NO")
    # To bring it in agreement with the R code (a side effect of ifelse):
    DIM_OUTPUT['MATCH'].where(DIM_OUTPUT['VIOLATION_VALUE'].notna(),
            other=np.nan, inplace=True)
    DIM_OUTPUT['TP_FP'] = np.where((DIM_OUTPUT['MATCH'] == "YES") &
                                   (DIM_OUTPUT['VALUE']==DIM_OUTPUT['CLEAN_VALUE']), "FP", "TP")
    DIM_OUTPUT['TP_FP'] = np.where(DIM_OUTPUT['MATCH'] == "NO", "", DIM_OUTPUT['TP_FP'])
    # To bring it in agreement with the R code (a side effect of ifelse):
    DIM_OUTPUT['TP_FP'].where(DIM_OUTPUT['MATCH'].notna(),
            other=np.nan, inplace=True)
    return DIM_OUTPUT


# 17 : Exporting DIM Output as CSV File
# params  - currentDayDIMOutput : current day DIM output
# params  - newRates : new rates in current day data
###
def Export_Output(currentDayDIMOutput, newRates):
    """
    Summary:
    -----------
    Called from the Trigger script on the current day DIM output data
    (currentDayDIMOutput) and new rates in current day data (newRates).
    It simply saves the two tables as CSV files.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayDIMOutput : DataFrame
        current day DIM output data
    newRates : DataFrame
        new rates in current day data
    
    Returns:
    -----------
    None : None
    """

    currentDayDIMOutput.to_csv(outputPath + "DIM_OUTPUT.csv", index=False)
    newRates.to_csv(outputPath + "NEW_RATES.csv", index=False)


def Export_Stale_Rates(currentDayDIMOutput):
    """
    Summary:
    -----------
    NOTE: never even called from anywhere!
    It simply saves Stale Rates with counter as a CSV File.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    currentDayDIMOutput : DataFrame
        current day DIM output data
    
    Returns:
    -----------
    None : None
    """

    if np.issubdtype(pd.to_datetime(currentDayDIMOutput['G.RECORDDATE']).dtype, np.datetime64):
        output_Date = str(currentDayDIMOutput['G.RECORDDATE'].dt.date.max())
    else:
        output_Date = str(pd.to_datetime(currentDayDIMOutput['G.RECORDDATE']).dt.date.max())
    crit = "(STALE_ASK == 1) | (STALE_BID == 1) | (STALE_MID == 1)"
    cols = ['G.RECORDDATE', 'S.RATELABEL', 'STALE_COUNTER_ASK', 'STALE_COUNTER_BID',
            'STALE_COUNTER_MID']
    staleRates = currentDayDIMOutput.query(crit)[cols]
    staleRates.to_csv(outputPath + "STALE_RATES_" + output_Date + ".csv", index=False)


def Export_Blank_Quotation(blankQuotation):
    """
    Summary:
    -----------
    Called from the Trigger script on the bond rates with no quotation
    (blankQuotation).
    It simply saves them as a CSV File.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    blankQuotation : DataFrame
        bond rates with no quotation
    
    Returns:
    -----------
    None : None
    """

    blankQuotation.to_csv(outputPath + "Blank_Quotation.csv", index=False)


def Export_Updated_Hist(relHistoricalBD,
                        relHistoricalFX,
                        relHistoricalEQ,
                        relHistoricalIR,
                        relHistoricalCM):
    """
    Summary:
    -----------
    NOTE: never even called from anywhere!
    It combines the historical clean data for BD, FX, EQ, IR and CM assets
    and saves them as a CSV File.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    relHistoricalBD : DataFrame
        historical clean data for BD
    relHistoricalFX : DataFrame
        historical clean data for FX
    relHistoricalEQ : DataFrame
        historical clean data for EQ
    relHistoricalIR : DataFrame
        historical clean data for IR
    relHistoricalCM : DataFrame
        historical clean data for CM
    
    Returns:
    -----------
    None : None
    """

    updatedHistRelData = pd.concat([relHistoricalBD,
                                    relHistoricalFX,
                                    relHistoricalEQ,
                                    relHistoricalIR,
                                    relHistoricalCM], ignore_index=True)
    updatedHistRelData.to_csv(outputPath + "updatedHistRelData_.csv", index=False)


def Export_Mature(matured):
    """
    Summary:
    -----------
    Called from the Trigger script on the matured bond rates (matured).
    It simply saves them as a CSV File.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    matured : DataFrame
       matured bond rates
    
    Returns:
    -----------
    None : None
    """

    # NOTE: output_Date not even used. Commented out:
    # output_Date <- as.character(max(matured$RECORDDATE))
    matured.to_csv(outputPath +  "matureRates.csv", index=False)


def find_help(abm, CheckCriteria):
    """
    Summary:
    -----------
    A helper function called by findASK(), findBid() and findMid() below.
    It combines the bodies of the 3 almost identical R-functions with the same
    names into a single body, but called with the new abm parameter as either
    "ASK"/"BID"/"MID".

    Parameters:
    -----------
    abm : str
        either "ASK", "BID" or "MID"
    CheckCriteria : Series/DataFrame (single column)
        values either missing or strings like "ASK,BID"
    
    Returns:
    -----------
    flagToReturn : boolean
    """

    abm2Present = [abm + "2" in str(c) for c in CheckCriteria]
    abm1Present = [abm + "1" in str(c) for c in CheckCriteria]
    abmPresent = [(abm + "2" not in str(c)) & (abm + "1" not in str(c)) & (abm in str(c)) for c in CheckCriteria]
    flagToReturn = [True if c1 | c2 | c3 else False for c1, c2, c3 in zip(abm2Present, abm1Present, abmPresent)]
    return flagToReturn


def findASK(CheckCriteria):
    """
    Summary:
    -----------
    Called by createRMSPassDIM() below.
    This is an R-to-Python translation of a function with the same name, whose
    main body has been moved to find_help() above, since it is shared with the
    functions findBid() and findMid().

    Parameters:
    -----------
    CheckCriteria : Series/DataFrame (single column)
        values either missing or strings like "ASK,BID"
    
    Returns:
    -----------
    the result of find_help() : boolean
    """

    return find_help("ASK", CheckCriteria)


def findBid(CheckCriteria):
    """
    Summary:
    -----------
    Called by createRMSPassDIM() below.
    This is an R-to-Python translation of a function with the same name, whose
    main body has been moved to find_help() above, since it is shared with the
    functions findASK() and findMid().

    Parameters:
    -----------
    CheckCriteria : Series/DataFrame (single column)
        values either missing or strings like "ASK,BID"
    
    Returns:
    -----------
    the result of find_help() : boolean
    """

    return find_help("BID", CheckCriteria)


def findMid(CheckCriteria):
    """
    Summary:
    -----------
    Called by createRMSPassDIM() below.
    This is an R-to-Python translation of a function with the same name, whose
    main body has been moved to find_help() above, since it is shared with the
    functions findASK() and findBid().

    Parameters:
    -----------
    CheckCriteria : Series/DataFrame (single column)
        values either missing or strings like "ASK,BID"
    
    Returns:
    -----------
    the result of find_help() : boolean
    """

    return find_help("MID", CheckCriteria)


def createRMSPassDIM(dataFile, check1, check2, check3):
    """
    Summary:
    -----------
    Called from the DIM_LoadData script on the collated RMS thresholds
    (dataFile) with relevant columns (check1, check2, check3).
    New binary value (1 or 0) columns, AskDIMInRMS, BidDIMInRMS and
    BidDIMInRMS created, values assigned based on the combination
    of booleans returned by findASK()/Bid()/Mid() for check1/2/3.

    Parameters:
    -----------
    dataFile : DataFrame
        typically rmsThresholds
    check1, check2, check3 : str (x3)
        column names of dataFile, typically
        "ZERO.CHECK.ATTRIBUTE", "NEG.CHECK.ATTRIBUTE", "VALID.BID.ASK.RULE"
    
    Returns:
    -----------
    dataFile : DataFrame
        updated rmsThresholds
    """

    dataFile['AskDIMInRMS'] = [1 if c1 | c2 | c3 else 0 for c1, c2, c3 in
            zip(findASK(dataFile[check1]), findASK(dataFile[check2]), findASK(dataFile[check3]))]
    dataFile['BidDIMInRMS'] = [1 if c1 | c2 | c3 else 0 for c1, c2, c3 in
            zip(findBid(dataFile[check1]), findBid(dataFile[check2]), findBid(dataFile[check3]))]
    dataFile['MidDIMInRMS'] = [1 if c1 | c2 | c3 else 0 for c1, c2, c3 in
            zip(findMid(dataFile[check1]), findMid(dataFile[check2]), findMid(dataFile[check3]))]
    return dataFile


def findThASKBID(dataFile, CheckCriteria):
    """
    Summary:
    -----------
    Called from the DIM_LoadData script on the collated RMS thresholds
    (dataFile) with the list of relevant columns (CheckCriteria).
    New binary value (1 or 0) columns, ThreshAsk, ThreshBID and
    ThreshMID created, values assigned based on the availability
    (missing or not) of values for ABS/REL TOLERANCES ASK/BID/MID,
    respectively.

    Parameters:
    -----------
    dataFile : DataFrame
        typically rmsThresholds
    CheckCriteria : list
        Collated RMS thresholds with relevant columns,
        usually along the lines (with each element optional):
        ["ABS.TOLERANCE.BID", "ABS.TOLERANCE.ASK", "ABS.TOLERANCE.MID",
         "REL.TOLERANCE.BID", "REL.TOLERANCE.ASK", "REL.TOLERANCE.MID",
         "ABS.TOLERANCE.BID1", "ABS.TOLERANCE.ASK1", "ABS.TOLERANCE.MID1",
         "REL.TOLERANCE.BID1", "REL.TOLERANCE.ASK1", "REL.TOLERANCE.MID1",
         "ABS.TOLERANCE.BID2", "ABS.TOLERANCE.ASK2", "ABS.TOLERANCE.MID2",
         "REL.TOLERANCE.BID2", "REL.TOLERANCE ASK2", "REL.TOLERANCE.MID2"]
    
    Returns:
    -----------
    dataFile : DataFrame
        updated rmsThresholds
    """

    # coa = [c for c in CheckCriteria if "ASK" in c]
    # cob = [c for c in CheckCriteria if "BID" in c]
    # com = [c for c in CheckCriteria if "MID" in c]
    dataFile['ThreshASK'] = dataFile[CheckCriteria].filter(like='ASK', axis=1).notna().any(axis=1)\
                            .astype(int)
    dataFile['ThreshBID'] = dataFile[CheckCriteria].filter(like='BID', axis=1).notna().any(axis=1)\
                            .astype(int)
    dataFile['ThreshMID'] = dataFile[CheckCriteria].filter(like='MID', axis=1).notna().any(axis=1)\
                            .astype(int)
    return dataFile
