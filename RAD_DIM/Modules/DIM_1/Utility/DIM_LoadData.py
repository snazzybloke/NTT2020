####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: Data Investigation Module IR Functions
This is the one of the last scripts in the DIM Module imported by the Trigger.
The purpose of this script is to carry out data related tasks by
defining and executing three functions, called from here.
It also saves about half a dozen relevant working tables in
a Report_date subdirectory inside the input directory.
This is an R-to-Python translation of the script with the same name.
Note that the R code does not define any functions but rather repeats
over and again the same lines of code on different tables/column.
"""

import pandas as pd
from confglob import config
from Modules.DIM_1.DIM_Trigger import *

curDate_s = str(curDate.date())


def col_check(df, reqd_cols, str2pt):
    """
    Summary:
    -----------
    Called from the function fcol_check() (if data read from a file) or
    directly from code lines below.
    Performs a simple check that the data table (df) comes with required
    columns (reqd_cols).
    This function does not appear per se in the R code, but its body is
    effectively being executed multiple times.

    Parameters:
    -----------
    df : DataFrame
        any data table of interest
    reqd_cols : list
        list of column names (i.e., strings) that must be in the df
    str2pt : str
        a string to print if some columns cannot be found in the df

    Returns:
    -----------
    None : None
    """

    exist_cols = df.columns
    # Boolean list (True if column found, False otherwise):
    not_exist_cols = [c in exist_cols for c in reqd_cols]

    # Check if all required columns exist in input data
    if sum(not_exist_cols) != len(not_exist_cols):
        # list of indices of not found columns:
        tmp = [i for i in range(len(not_exist_cols)) if not not_exist_cols[i]]
        print(f"Expected column(s) {np.array(reqd_cols)[tmp]} unavailable from " + str2pt + " data")
        exit()


def fcol_check(filename, reqd_cols, str2pt):
    """
    Summary:
    -----------
    Called from the code lines below.
    Performs a simple check that the data table file (filename) comes with
    required columns (reqd_cols) by calling col_check() above.
    This function does not appear per se in the R code, but its body is
    effectively being executed multiple times.

    Parameters:
    -----------
    filename : str
        any data table file of interest
    reqd_cols : list
        list of column names (i.e., strings) that must appear in the file
    str2pt : str
        a string to print if some columns do not appear in the file

    Returns:
    -----------
    None : None
    """

    # Load thresholds only 4 rows sample data to check data consistencies for thresholds
    file_path = config['inputPath'] + filename
    df = pd.read_csv(file_path, nrows = 4)
    col_check(df, reqd_cols, str2pt)
    del df


#--------------------------------------#
# 1. Check if required columns exist   #
#--------------------------------------#

# Load current day violation report to obtain RMS flags
violation = pd.read_csv(config['inputPath'] + violationReport)

dropcol = ["VCGCOMMENT","USERCOMMENT"]

cols = violation.columns

violation_duplicate = violation[violation.duplicated(subset=set(cols).difference(set(dropcol)))]

if not violation_duplicate.empty:
    violation_duplicate_all = violation.query("RATELABEL in @violation_duplicate.RATELABEL")
    violation_duplicate_all.to_csv(config['outputPath'] + "violation_duplicate_check_comment_" +
            curDate_s + ".csv", index=False)
    print("Violation report comment duplicate error!")
    print("Execution halted")
    # NOTE: ARGS, if exists, should be  vars(argparse.ArgumentParser()).parse_args()
    if "ARGS" in locals() or x in globals():
        if ARGS["AUTO_RUN_FLAG"]:
            violation = violation[~violation.duplicated(subset=set(cols).difference(set(dropcol)))]
        else:
            exit()

print("There is no violation report comment duplicate!")

# i>
# Previous day reliable data required columns

prevRelReqCols = [prevRelRecordDate, prevRelRateId, prevRelRateLabel, prevRelAssetType,
                  prevRelInstrumentType, prevRelProductType, prevRelAsk, prevRelBid,
                  prevRelAsk1, prevRelBid1, prevRelAsk2, prevRelBid2, prevRelMid, prevRelMid1,
                  prevRelMid2, prevRelMidnew, prevRelCurrency, prevRelTenor, prevRelMXCurrency,
                  prevRelMXGentype, prevRelIsinCode, prevRelMarket, prevRelFAMarket,
                  prevRelRateType, prevRelSMarket, prevRelQuotation, prevRelCurveLabel,
                  prevRelMaturity]

col_check(prevDayReliable, prevRelReqCols, "previous day reliable")

# ii>
# Current Day Data

curRelReqCols = [curRelRecordDate, curRelRateId, curRelAssetType, curRelInstrumentType, curRelBid,
                 curRelAsk, curRelBid1, curRelAsk1, curRelBid2, curRelAsk2, curRelMid, curRelMid1,
                 curRelMid2, S_RATETYPE, curRelClose, curRelLast, curRelHistoric]

fcol_check(currentDayDataFile, curRelReqCols, "current day reliable")

# iii>
# Historical Reliable Data

histRelReqCols = [histRelRecordDate, histRelRateId, histRelRateLabel, histRelAssetType,
                  histRelInstrumentType, histRelProductType, histRelAsk, histRelBid, histRelAsk1,
                  histRelBid1, histRelAsk2, histRelBid2, histRelMid, histRelMid1, histRelMid2,
                  histRelCurrency, histRelTenor, histRelMXCurrency, histRelMXGentype,
                  histRelIsinCode, histRelMarket, histRelFAMarket, histRelRateType, histRelSMarket,
                  histRelQuotation, histRelCurveLabel, histRelMaturity, histRelQuotationFinal,
                  histRelAskNew, histRelBidNew, histRelMidNew, histRelLast, histRelClose,
                  histRelHistoric, histRelASKSingle, histRelBIDSingle, histRelAdjustedASK,
                  histRelAdjustedBID]

col_check(relHistorical, histRelReqCols, "historical reliable")

# iv> Thresholds files

#Load the required columns
threshReqCols = [threshRateLabel, threshAssetType, threshInstrumentType]

threshReqColsAB = [threshAbsBID, threshAbsASK, threshAbsMID,
                   threshRelBID, threshRelASK, threshRelMID]

threshReqColsA1B1 = [threshAbsBID1, threshAbsASK1, threshAbsMID1,
                     threshRelBID1, threshRelASK1, threshRelMID1]

threshReqColsA2B2 = [threshAbsBID2, threshAbsASK2, threshAbsMID2,
                     threshRelBID2, threshRelASK2, threshRelMID2]

threshReqColsDIM = [zero_check_criteria, negative_check_criteria, valid_bid_ask_criteria]

threshReqColsBDCorp = threshReqCols + threshReqColsDIM +\
                       threshReqColsAB + threshReqColsA1B1 + threshReqColsA2B2
threshReqColsBDFuture = threshReqCols + threshReqColsDIM + threshReqColsAB + threshReqColsA1B1
threshReqColsBDGovt = threshReqCols + threshReqColsDIM + threshReqColsAB + threshReqColsA1B1
threshReqColsEQSpot = threshReqCols + threshReqColsDIM + threshReqColsAB + threshReqColsA1B1
threshReqColsFXFuture = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXSmile = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXSpot = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXSwap = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXVol = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsMRS = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsMRSVD = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXSwapImpliedRel = threshReqCols + threshReqColsDIM + threshReqColsAB
threshReqColsFXSwapImpliedDer = threshReqCols + threshReqColsDIM + threshReqColsAB

fcol_check(threshFileBDCORP, threshReqColsBDCorp, "BD Corp thresholds")
fcol_check(threshFileBDFUTURES, threshReqColsBDFuture, "BD Futures thresholds")
fcol_check(threshFileBDGOVT, threshReqColsBDGovt, "BD Govt thresholds")
fcol_check(threshFileEQSPOT, threshReqColsEQSpot, "EQ Spot thresholds")
fcol_check(threshFileFXFUTURES, threshReqColsFXFuture, "FX Futures thresholds")
fcol_check(threshFileFXSMILES, threshReqColsFXSmile, "FX smile thresholds")
fcol_check(threshFileFXSPOT, threshReqColsFXSpot, "FX Spot thresholds")
fcol_check(threshFileFXSWAP, threshReqColsFXSwap, "FX Swap thresholds")
fcol_check(threshFileFXVOL, threshReqColsFXVol, "FX Vol thresholds")
fcol_check(threshFileMRS, threshReqColsMRS, "MRS thresholds")
fcol_check(threshFileMRSVD, threshReqColsMRSVD, "MRS VD thresholds")
tmpf = "FX_SWAP_IMPLIED_Rel_" + curDate_s + ".csv"
fcol_check(tmpf, threshReqColsFXSwapImpliedRel, "FX Swap Implied Rel thresholds")
tmpf = "FX_SWAP_IMPLIED_Der_" + curDate_s + ".csv"
fcol_check(tmpf, threshReqColsFXSwapImpliedDer, "FX Swap Implied Der thresholds")

#--------------------------------------#
# 2. Load the unclean current day data #
#--------------------------------------#
currentDayData = pd.read_csv(config['inputPath'] + currentDayDataFile)

# Filtering and renaming columns in current day unclean data
currentDayDataR = currentDayData.query("RATETYPE == 'R'")
currentDayDataD = currentDayData.query("(RATETYPE == 'D') & (INSTRUMENTTYPE in @derivedinstrument)")
currentDayData = pd.concat([currentDayDataR, currentDayDataD], ignore_index=True)
del currentDayDataR, currentDayDataD
currentDayData.rename({'RATEID': 'RELIABLEID'}, axis=1, inplace=True)

# Changing column classes
currentDayData['RECORDDATE'] = pd.to_datetime(currentDayData['RECORDDATE'],
        format="%d-%b-%y %H.%M.%S.%f %p", errors="coerce").dt.normalize()
colNames = ["RELIABLEID", "ASK", "BID", "ASK1", "BID1", "ASK2", "BID2", "MID", "MID1", "MID2",
            "CLOSING", "LAST", "HISTORICCLOSE"]
for c in colNames:
    currentDayData[c] = pd.to_numeric(currentDayData[c], errors='coerce')
    currentDayData[c] = currentDayData[c].round(decimalPlace)

# Load the Previous day golden rates data for all assets
# Changing column classes of previous day reliable file
prevDayReliable = prevDayReliable[prevRelReqCols]
prevDayReliable.RECORDDATE = pd.to_datetime(prevDayReliable['RECORDDATE'], errors="coerce")
prevDayReliable.MATURITY = pd.to_datetime(prevDayReliable['MATURITY'], format = "%d/%m/%Y",
        errors="coerce")
colNames = ["RELIABLEID", "G.ASK", "G.BID", "G.ASK1", "G.BID1", "G.ASK2", "G.BID2",
            "MID", "MID1", "MID2"]
for c in colNames:
    prevDayReliable[c] = pd.to_numeric(prevDayReliable[c], errors='coerce')

# Changing reliable historical clean data
relHistorical = relHistorical[histRelReqCols]
relHistorical['RECORDDATE'] = pd.to_datetime(relHistorical['RECORDDATE'], errors="coerce")
colNames = ["RELIABLEID", "ASK", "BID", "ASK1", "BID1", "ASK2", "BID2", "ASKNEW", "BIDNEW", "MID",
            "MID1", "MID2", "MIDNEW", "ASKSingle", "BIDSingle", "AdjustedASK", "AdjustedBID",
            "CLOSING", "LAST", "HISTORICCLOSE"]
for c in colNames:
    relHistorical[c] = pd.to_numeric(relHistorical[c], errors='coerce')

# Load the stock event data
# Changing column classes
stockEventData['ADJUST_BY_VALUE'] = pd.to_numeric(stockEventData['ADJUST_BY_VALUE'], errors='coerce')

currDayReliable = pd.read_csv(config['inputPath'] + currDayReliableFile)
# Changing column classes for current day reliable clean data
currDayReliable['RECORDDATE'] = pd.to_datetime(currDayReliable['RECORDDATE'],
        format="%d-%b-%y %H.%M.%S.%f %p", errors="coerce").dt.normalize()
currDayReliable.rename({"RATEID": "RELIABLEID"}, axis=1, inplace=True)
currDayReliable['MATURITY'] = pd.to_datetime(currDayReliable['MATURITY'], format = "%d/%m/%Y",
        errors="coerce")
currDayReliableR = currDayReliable.query("RATETYPE == 'R'")
currDayReliableD = currDayReliable.query("(RATETYPE == 'D') & (INSTRUMENTTYPE in @derivedinstrument)")
currDayReliable = pd.concat([currDayReliableR, currDayReliableD], ignore_index=True)
colNames = ["RELIABLEID", "ASK", "BID", "ASK1", "BID1", "ASK2", "BID2",
            "LAST", "CLOSING", "HISTORICCLOSE"]
for c in colNames:
    currDayReliable[c] = pd.to_numeric(currDayReliable[c], errors='coerce')


def loadRMSThresholdFiles(staticDate):
    """
    Summary:
    -----------
    Called from the code line below.
    It reads about twenty CSV files for the given date (staticDate)
    using the appropriate subsets of columns and concatenates those into
    a single rmsThresholds table.
    This is a R-to-Python translation of the function with the same name
    found in Utility/common.R, provided here for convenience.

    Parameters:
    -----------
    staticDate : str
        date in the "%Y-%m-%d" format

    Returns:
    -----------
    rmsThresholds : DataFrame
        RMS thresholds for valid rates
    """

    thresholdBDCORP_Rel = pd.read_csv(config['inputPath'] +  "BD_CORP_Rel_" + staticDate + ".csv",
            usecols=threshReqColsBDCorp, dtype=str)
    thresholdBDCORP_Der = pd.read_csv(config['inputPath'] +  "BD_CORP_Der_" + staticDate + ".csv")
    thresholdBDFUTU = pd.read_csv(config['inputPath'] +  "BD_FUTURES_Rel_" + staticDate + ".csv",
            usecols=threshReqColsBDFuture, dtype=str)
    thresholdBDGOVT_Rel = pd.read_csv(config['inputPath'] +  "BD_GOVT_Rel_" + staticDate + ".csv",
            usecols=threshReqColsBDGovt, dtype=str)
    thresholdBDGOVT_Der = pd.read_csv(config['inputPath'] +  "BD_GOVT_Der_" + staticDate + ".csv")
    BD_SDS_Der = pd.read_csv(config['inputPath'] +  "BD_SDS_Der_" + staticDate + ".csv")
    BD_SDS_Rel = pd.read_csv(config['inputPath'] +  "BD_SDS_Rel_" + staticDate + ".csv")
    thresholdEQSPOT = pd.read_csv(config['inputPath'] +  "EQ_SPOT_Rel_" + staticDate + ".csv",
            usecols=threshReqColsEQSpot, dtype=str)
    thresholdFXFUTU = pd.read_csv(config['inputPath'] +  "FX_FUTURES_Rel_" + staticDate + ".csv",
            usecols=threshReqColsFXFuture, dtype=str)
    thresholdFXSMILE = pd.read_csv(config['inputPath'] +  "FX_SMILE_Rel_" + staticDate + ".csv",
            usecols=threshReqColsFXSmile, dtype=str)
    thresholdFXSPOT = pd.read_csv(config['inputPath'] +  "FX_SPOT_Rel_" + staticDate + ".csv",
            usecols=threshReqColsFXSpot, dtype=str)
    thresholdFXSWAP = pd.read_csv(config['inputPath'] +  "FX_SWAP_Rel_" + staticDate + ".csv",
            usecols=threshReqColsFXSwap, dtype=str)
    thresholdFXVOL = pd.read_csv(config['inputPath'] +  "FX_VOL_Rel_" + staticDate + ".csv",
            usecols=threshReqColsFXVol, dtype=str)
    MRS_Der = pd.read_csv(config['inputPath'] +  "MRS_Der_" + staticDate + ".csv")
    MRS_Rel = pd.read_csv(config['inputPath'] +  "MRS_Rel_" + staticDate + ".csv",
            usecols=threshReqColsMRS, dtype=str)
    FX_SWAP_IMPLIED_Rel = pd.read_csv(config['inputPath'] +  "FX_SWAP_IMPLIED_Rel_" + staticDate + ".csv",
            usecols=threshReqColsMRSVD, dtype=str)
    FX_SWAP_IMPLIED_Der = pd.read_csv(config['inputPath'] +  "FX_SWAP_IMPLIED_Der_" + staticDate + ".csv",
            usecols=threshReqColsMRSVD, dtype=str)
    MRS_VD_Der = pd.read_csv(config['inputPath'] +  "MRS_VD_Der_" + staticDate + ".csv",
            usecols=threshReqColsMRSVD, dtype=str)
    MRS_VD_REL = pd.read_csv(config['inputPath'] +  "MRS_VD_Rel_" + staticDate + ".csv")

    #RBIND ALL THE DIFFERENT THRESHOLD FILES
    rmsThresholds = pd.concat([thresholdBDGOVT_Rel, thresholdBDGOVT_Der, thresholdBDCORP_Rel,
                                thresholdBDCORP_Der, thresholdBDFUTU, thresholdEQSPOT,
                                thresholdFXFUTU, thresholdFXSMILE, thresholdFXSPOT, thresholdFXSWAP,
                                thresholdFXVOL, MRS_Der, MRS_Rel, FX_SWAP_IMPLIED_Rel,
                                FX_SWAP_IMPLIED_Der, MRS_VD_Der, MRS_VD_REL, BD_SDS_Der,
                                BD_SDS_Rel], ignore_index=True)
    return rmsThresholds


rmsThresholds = loadRMSThresholdFiles(curDate_s)

# setting name format right (NOTE: may have to c.replace(".", " ") each column name c of interest)
for c in threshReqColsAB + threshReqColsA1B1 + threshReqColsA2B2:
    rmsThresholds[c] = pd.to_numeric(rmsThresholds[c], errors='coerce')

rmsThresholds = createRMSPassDIM(rmsThresholds, threshReqColsDIM[0], threshReqColsDIM[1], threshReqColsDIM[2])

rmsThresholds = findThASKBID(rmsThresholds, threshReqColsAB + threshReqColsA1B1 + threshReqColsA2B2)

rmsThresholds['Override'] = rmsThresholds['INSTRUMENTTYPE'].isin(instrumentsAskBidMid).astype(int)

#--------------------------#
#4.Store Daily input files #
#--------------------------#

# Create directory for storing input data
import pathlib

dpath = pathlib.Path().cwd() / config['inputPath'] / ("Report_" + curDate_s)
dpath.mkdir()

currDayReliable.to_csv(dpath / ("daily_goldenrates_" + curDate_s + ".csv"), index=False)

# Write current day unclean data
currentDayData.to_csv(dpath / ("daily-uncleansed-rate_" + curDate_s + ".csv"), index=False)

# Current data stock event data
stockEventData.to_csv(dpath / "stockEventData.csv", index=False)

# Previous day clean reliable data
prevDayReliable.to_csv(dpath / "previousDayReliable.csv", index=False)

# Reliable historical clean data
relHistorical.to_csv(dpath / "historical_golden_rates.csv", index=False)

# Daily thresholds file - saved daily for easy reference
rmsThresholds.to_csv(dpath / "rmsThresholds.csv", index=False)
