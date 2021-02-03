####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description:  DIM - Input Parameters (Data Structure & File names)
This is the first script for the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is to set (i) the file names for 2-3 dozen input/output files
based on the curDate (and staticDate, which is derived from it), and
(ii) a large number of variables that refer to column names in data tables.
This is a verbatim R-to-Python translation of the script with the same name.
"""

import confglob as cg
import datetime as dt
from dateutil.parser import parse
#--------------------------------------------------#
# 1 :    Working directory and data path           #
#--------------------------------------------------#
# Input and Output Data path directories

# Name of Current day unclean reliable data for all assets
currentDayDataFile = "daily-uncleansed-rate_" + cg.curDate + ".csv"

# Current Day Reliable Cleaned / Golden Rates Data
currDayReliableFile = "daily_goldenrates_" + cg.curDate + ".csv"

# Current day cleansing violation report
violationReport = "daily-cleansing-violation-report_" + cg.curDate + ".csv"

#-------------------------------------------------#
# Declaring date for static files                 #
#-------------------------------------------------#

curDate = parse(cg.curDate)
threshold_switchdate = parse(cg.threshold_switchdate)

if curDate < threshold_switchdate:
    n = cg.daydatemapping.loc[cg.daydatemapping["Day"] == cg.staticDay, 'n']
    # change from wday(curDate)> n to wday(curDate)>= n, then tuesday will use this week data
    if curDate.weekday() >= n:
        staticDate = curDate - abs(curDate.weekday() - n)
    else:
        staticDate = curDate - abs(abs(curDate.weekday() - n) - 6)
    staticDate = dt.date(staticDate)
else:
    staticDate = curDate

# update the global var:
cg.staticDate = staticDate

strstDate  = staticDate.strftime('%Y-%m-%d')
print(f"Threshold file date: {strstDate}")

#--------------------------------------------------#
# 2 :    Threshold file names parameterized        #
#--------------------------------------------------#

# BD CORP
threshFileBDCORP = "BD_CORP_Rel_" + strstDate + ".csv"
# BD FUTURES
threshFileBDFUTURES = "BD_FUTURES_Rel_" + strstDate + ".csv"
# BD GOVT
threshFileBDGOVT = "BD_GOVT_Rel_" + strstDate + ".csv"
# # BD CORP Dyn
# threshFileBDCORPDyn = "BD_CORP_" + dynDate + ".csv"
# # # BD GOVT Dyn
# threshFileBDGOVTDyn = "BD_GOVT_" + dynDate + ".csv"
# EQ SPOT
threshFileEQSPOT = "EQ_SPOT_Rel_" + strstDate + ".csv"
# FX FUTURES
threshFileFXFUTURES = "FX_FUTURES_Rel_" + strstDate + ".csv"
# FX SMILES
threshFileFXSMILES = "FX_SMILE_Rel_" + strstDate + ".csv"
# FX SPOT
threshFileFXSPOT = "FX_SPOT_Rel_" + strstDate + ".csv"
# FX SWAP
threshFileFXSWAP = "FX_SWAP_Rel_" + strstDate + ".csv"
# FX VOL
threshFileFXVOL = "FX_VOL_Rel_" + strstDate + ".csv"
# MRS
threshFileMRS = "MRS_Rel_" + strstDate + ".csv"
# MRS VD
threshFileMRSVD = "MRS_VD_Der_" + strstDate + ".csv"
#--------------------------------------------------#
# 3 : Expected Column Names (Required)             #
#--------------------------------------------------#
# i> Current Day Reliable Data
# RECORDDATE
curRelRecordDate = "RECORDDATE"

# Rate Id
curRelRateId = "RATEID"

# Rate type
# S.RATETYPE = "RATETYPE"
S_RATETYPE = "RATETYPE"

# ASSETTYPE
curRelAssetType = "ASSETTYPE"

# INSTRUMENTTYPE
curRelInstrumentType = "INSTRUMENTTYPE"

# Ask Price
curRelAsk = "ASK"

# Bid Price
curRelBid = "BID"

# Ask1
curRelAsk1 = "ASK1"

# Bid1
curRelBid1 = "BID1"

# Ask2
curRelAsk2 = "ASK2"

# Bid2
curRelBid2 = "BID2"

# Mid
curRelMid = "MID"

# Mid1
curRelMid1 = "MID1"

# Mid2
curRelMid2 = "MID2"

# Closing
curRelClose = "CLOSING"

# Last
curRelLast = "LAST"

# Historic Close
curRelHistoric = "HISTORICCLOSE"


# ii> Previous Day Reliable Data
# RECORDDATE
prevRelRecordDate = "RECORDDATE"

# Rate Id
prevRelRateId = "RELIABLEID"

# Ratelabel
prevRelRateLabel = "RATELABEL"

# Asset Type
prevRelAssetType = "ASSETTYPE"

# Instrument type
prevRelInstrumentType = "INSTRUMENTTYPE"

# Product type
prevRelProductType = "PRODUCTTYPE"

# prevrency
prevRelCurrency = "CURRENCY"

# Tenor
prevRelTenor = "TENOR"

# MX Currency
prevRelMXCurrency = "MXCURRENCY"

# Mxgentype
prevRelMXGentype = "MXGENTYPE"

# ISINCODE
prevRelIsinCode = "ISINCODE"

# Market
prevRelMarket = "MARKET"

# FA Market
prevRelFAMarket = "FAMARKET"

# Rate Type
prevRelRateType = "RATETYPE"

# S Market
prevRelSMarket = "SMARKET"

# Quotation
prevRelQuotation = "QUOTATION"

# Curve Label
prevRelCurveLabel = "YIELDCURVELABEL"

# Maturity
prevRelMaturity = "MATURITY"

# Ask Price
prevRelAsk = "G.ASK"

# Bid Price
prevRelBid = "G.BID"

# Ask1
prevRelAsk1 = "G.ASK1"

# Bid1
prevRelBid1 = "G.BID1"

# Ask2
prevRelAsk2 = "G.ASK2"

# Bid2
prevRelBid2 = "G.BID2"

# Mid
prevRelMid = "MID"

# Mid1
prevRelMid1 = "MID1"

# Mid2
prevRelMid2 = "MID2"

# Midnew
prevRelMidnew = "MIDNEW"


# iii> Historical Reliable Data
# Date of record
histRelRecordDate = "RECORDDATE"

# Rate Id
histRelRateId = "RELIABLEID"

# Ratelabel
histRelRateLabel = "RATELABEL"

# Asset Type
histRelAssetType = "ASSETTYPE"

# Instrument type
histRelInstrumentType = "INSTRUMENTTYPE"

# Product type
histRelProductType = "PRODUCTTYPE"

# Currency
histRelCurrency = "CURRENCY"

# Tenor
histRelTenor = "TENOR"

# MX Currency
histRelMXCurrency = "MXCURRENCY"

# Mxgentype
histRelMXGentype = "MXGENTYPE"

# ISINCODE
histRelIsinCode = "ISINCODE"

# Market
histRelMarket = "MARKET"

# FA Market
histRelFAMarket = "FAMARKET"

# Rate Type
histRelRateType = "RATETYPE"

# S Market
histRelSMarket = "SMARKET"

# Quotation
histRelQuotation = "QUOTATION"

# Curve Label
histRelCurveLabel = "YIELDCURVELABEL"

# Maturity
histRelMaturity = "MATURITY"

# Ask Price
histRelAsk = "ASK"

# Bid Price
histRelBid = "BID"

# Ask1
histRelAsk1 = "ASK1"

# Bid1
histRelBid1 = "BID1"

# Ask2
histRelAsk2 = "ASK2"

# Bid2
histRelBid2 = "BID2"

# Bid1
histRelMid = "MID"

# Bid1
histRelMid1 = "MID1"

# Bid1
histRelMid2 = "MID2"

# QUOTATION_FINAL
histRelQuotationFinal = "QUOTATION_FINAL"

# ASKNEW
histRelAskNew = "ASKNEW"

# BIDNEW
histRelBidNew = "BIDNEW"

# MIDNEW
histRelMidNew = "MIDNEW"

# LAST
histRelLast = "LAST"

# CLOSING
histRelClose = "CLOSING"

# HISTORIC CLOSE
histRelHistoric = "HISTORICCLOSE"

# ASK Single
histRelASKSingle = "ASKSingle"

# BID Single
histRelBIDSingle = "BIDSingle"

# Adjusted ASK
histRelAdjustedASK = "AdjustedASK"

# Adjusted BID
histRelAdjustedBID = "AdjustedBID"


# iv> Thresholds Files
#Ratelabel
threshRateLabel = "RATELABEL"

#Assettype
threshAssetType = "ASSETTYPE"

#Instrumenttype
threshInstrumentType = "INSTRUMENTTYPE"
##
#BIDthreshold
threshAbsBID = "ABS TOLERANCE BID"

#ASKthreshold
threshAbsASK = "ABS TOLERANCE ASK"

#MIDthreshold
threshAbsMID = "ABS TOLERANCE MID"

#BIDthreshold
threshRelBID = "REL TOLERANCE BID"

#ASKthreshold
threshRelASK = "REL TOLERANCE ASK"

#MIDthreshold
threshRelMID = "REL TOLERANCE MID"

##1
#BIDthreshold
threshAbsBID1 = "ABS TOLERANCE BID1"

#ASKthreshold
threshAbsASK1 = "ABS TOLERANCE ASK1"

#MIDthreshold
threshAbsMID1 = "ABS TOLERANCE MID1"

#BIDthreshold
threshRelBID1 = "REL TOLERANCE BID1"

#ASKthreshold
threshRelASK1 = "REL TOLERANCE ASK1"

#MIDthreshold
threshRelMID1 = "REL TOLERANCE MID1"


##2
#BIDthreshold
threshAbsBID2 = "ABS TOLERANCE BID2"

#ASKthreshold
threshAbsASK2 = "ABS TOLERANCE ASK2"

#MIDthreshold
threshAbsMID2 = "ABS TOLERANCE MID2"

#BIDthreshold
threshRelBID2 = "REL TOLERANCE BID2"

#ASKthreshold
threshRelASK2 = "REL TOLERANCE ASK2"

#MIDthreshold
threshRelMID2 = "REL TOLERANCE MID2"

# DIM Check Columns

# Zero Check attribute
zero_check_criteria = "ZERO CHECK ATTRIBUTE"

# Negative Check Columns
negative_check_criteria = "NEG CHECK ATTRIBUTE"

# Valid Bid Ask
valid_bid_ask_criteria = "VALID BID/ASK RULE"

# Dynamic file cols
# 
# #ASK
# threshDynASK = "THRESHOLD_ASK"
# 
# #BID
# threshDynBID = "THRESHOLD_BID"
