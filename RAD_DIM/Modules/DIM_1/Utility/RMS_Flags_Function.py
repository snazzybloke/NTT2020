####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: Extracting RMS flags from violation report
This is the final script in the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is described by the following steps:
 1. Create the ASK_OR_BID column in violations table, with values based on
    the violations message VIOLATIONMSG values
 2. Extracting unclean values for ASK/BID/MID from violation report,
    based on RULENAME values (".*ABS/REL.*" check) to ar_rr_check
 3. Extracting unclean values from violation report, based on RULENAME values
    (".*ZERO.*" check) to zero_check and assign ASK/BID/MID zero to those rows.
 4. Extracting unclean values for ASK/BID/MID from violation report,
    based on RULENAME values (".*NEGATIVE.*" check) to negative_check
    (to indicate if Negative check passed or failed)
 5. Extracting unclean values for ASK/BID from violation report,
    based on RULENAME values (".*VALID.*" check) to valid_check
    (to indicate if valid bid ask check passed or failed)
Eventually, the check tables are concatenated and ASK/BID/MID
renames to VIOLATION_VALUE.

This is an R-to-Python translation of the script with the same name.
"""

import numpy as np
import pandas as pd
import re
from confglob import config
from Modules.DIM_1.DIM_Trigger import violation, curDate, currDayReliable, AddQuotationBD,\
        SingleAskBidColumnBD, SingleAskBidColumnFX, SingleAskBidColumnEQ, SingleAskBidColumnIR,\
        SingleAskBidColumnCM, blankQuotation

###
# 1. Creating single ASK/BID/MID columns to extract the type of value
###

if not violation.empty:
    violation['RECORDDATE'] = pd.to_datetime(violation['UPDATEDDATE'],
            format="%d-%b-%y %H.%M.%S.%f %p", errors="coerce")
    violation['ASK_OR_BID'] = np.where(violation['VIOLATIONMSG'].str.contains(".*BID.*"), "BID", np.nan)
    for c in (".*ASK.*", ".*BID1.*", ".*ASK1.*", ".*MID.*", ".*MID1.*"):
        violation['ASK_OR_BID'].mask(violation['VIOLATIONMSG'].str.contains(c),
                other=c[2: -2], inplace=True)
    violation.dropna(axis=0, subset=['ASK_OR_BID'], inplace=True)

###
# 2. Extracting unclean values from violation message
###
  
    if (not violation['RULENAME'].str.contains(".*ABSOLUTE.*").empty) or\
            (not violation['RULENAME'].str.contains(".*RELATIVE.*").empty):
        ar_rr_check = violation.\
                query("RULENAME in ['ABSOLUTE TOLERANCE CHECK', 'RELATIVE TOLERANCE CHECK']")

        # if ASK_OR_BID == ASK, find the ASK value between '=' and ';' (or just ""):
        ar_rr_check = ar_rr_check.assign(ASK=np.where(ar_rr_check['ASK_OR_BID']=="ASK",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""), ""))
        ar_rr_check['ASK'] = ar_rr_check['ASK'].mask((ar_rr_check['ASK_OR_BID'] == "ASK") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)
        ar_rr_check['ASK'] = ar_rr_check['ASK'].mask(ar_rr_check['ASK_OR_BID']=="ASK1",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""))
        ar_rr_check['ASK'] = ar_rr_check['ASK'].mask((ar_rr_check['ASK_OR_BID'] == "ASK1") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)

        ar_rr_check = ar_rr_check.assign(BID=np.where(ar_rr_check['ASK_OR_BID']=="BID",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""), ""))
        ar_rr_check['BID'] = ar_rr_check['BID'].mask((ar_rr_check['ASK_OR_BID'] == "BID") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)
        ar_rr_check['BID'] = ar_rr_check['BID'].mask(ar_rr_check['ASK_OR_BID']=="BID1",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""))
        ar_rr_check['BID'] = ar_rr_check['BID'].mask((ar_rr_check['ASK_OR_BID'] == "BID1") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)

        ar_rr_check = ar_rr_check.assign(MID=np.where(ar_rr_check['ASK_OR_BID']=="MID",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""), ""))
        ar_rr_check['MID'] = ar_rr_check['MID'].mask((ar_rr_check['ASK_OR_BID'] == "MID") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)
        ar_rr_check['MID'] = ar_rr_check['MID'].mask(ar_rr_check['ASK_OR_BID']=="MID1",
                ar_rr_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: x.find(";")-1] if ("=" in x) & (";" in x) else ""))
        ar_rr_check['MID'] = ar_rr_check['MID'].mask((ar_rr_check['ASK_OR_BID'] == "MID1") &\
                ar_rr_check['VIOLATIONMSG'].str.contains("divide by zero", flags=re.I), other=0)
    
        ar_rr_check['ASK'] = pd.to_numeric(ar_rr_check['ASK'], errors='coerce')
        ar_rr_check['BID'] = pd.to_numeric(ar_rr_check['BID'], errors='coerce')
        ar_rr_check['MID'] = pd.to_numeric(ar_rr_check['MID'], errors='coerce')
    else:
        ar_rr_check = pd.DataFrame()

###
# 4. Creating single column to indicate Zero check passed or failed
###
    if not violation['RULENAME'].str.contains(".*ZERO.*").empty:
        zero_check = violation[violation['RULENAME'].str.contains(".*ZERO.*")]
        zero_check = zero_check.assign(ASK=0)
        zero_check = zero_check.assign(BID=0)
        zero_check = zero_check.assign(MID=0)
    else:
        zero_check = pd.DataFrame()

###
# 5. Creating single column to indicate Negative check passed or failed
###
    if not violation['RULENAME'].str.contains(".*NEGATIVE.*").empty:
        negative_check = violation[violation['RULENAME'].str.contains(".*NEGATIVE.*")]

        negative_check = negative_check.assign(ASK=np.where(negative_check['ASK_OR_BID']=="ASK",
                negative_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: -2] if "=" in x else ""), ""))

        negative_check = negative_check.assign(BID=np.where(negative_check['ASK_OR_BID']=="BID",
                negative_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: -2] if "=" in x else ""), ""))

        negative_check = negative_check.assign(MID=np.where(negative_check['ASK_OR_BID']=="MID",
                negative_check['VIOLATIONMSG']\
                .apply(lambda x: x[x.find("=")+1: -2] if "=" in x else ""), ""))

    else:
        negative_check = pd.DataFrame()

###
# 6. Creating single column to indicate valid bid ask check passed or failed
###
  
    if not violation['RULENAME'].str.contains(".*VALID.*").empty:
        valid_bidask_check = violation[violation['RULENAME'].str.contains(".*VALID.*")]
    
        def check(msg):
            ask_index = msg.find("ASK")
            bid_index = msg.find("BID")
            if ask_index < bid_index:
                return 1
            else:
                return 0
        
        valid_bidask_check = valid_bidask_check.assign(askearly=valid_bidask_check['VIOLATIONMSG']
                .apply(check))
        valid_bidask_check = valid_bidask_check.assign(ASK=
            np.where((valid_bidask_check['askearly'] == 1) & (valid_bidask_check['ASK_OR_BID'] == "ASK"),
                valid_bidask_check['VIOLATIONMSG']\
                    .apply(lambda x: x[x.find("ASK=")+4: x.find(",")]),
            np.where((valid_bidask_check['askearly'] == 0) & (valid_bidask_check['ASK_OR_BID'] == "ASK"),
                valid_bidask_check['VIOLATIONMSG']\
                     .apply(lambda x: x[x.find("ASK=")+4: x.find("]")]),
            np.where((valid_bidask_check['askearly'] == 1) & (valid_bidask_check['ASK_OR_BID'] == "ASK1"),
                valid_bidask_check['VIOLATIONMSG']\
                    .apply(lambda x: x[x.find("ASK1=")+5: x.find(",")]),
            np.where((valid_bidask_check['askearly'] == 0) & (valid_bidask_check['ASK_OR_BID'] == "ASK1"),
                valid_bidask_check['VIOLATIONMSG']\
                     .apply(lambda x: x[x.find("ASK1=")+4: x.find("]")]), np.nan)))))

        valid_bidask_check = valid_bidask_check.assign(BID=
            np.where((valid_bidask_check['askearly'] == 0) & (valid_bidask_check['ASK_OR_BID'] == "ASK"),
                valid_bidask_check['VIOLATIONMSG']\
                    .apply(lambda x: x[x.find("BID=")+4: x.find(",")]),
            np.where((valid_bidask_check['askearly'] == 1) & (valid_bidask_check['ASK_OR_BID'] == "ASK"),
                valid_bidask_check['VIOLATIONMSG']\
                     .apply(lambda x: x[x.find("BID=")+4: x.find("]")]),
            np.where((valid_bidask_check['askearly'] == 0) & (valid_bidask_check['ASK_OR_BID'] == "ASK1"),
                valid_bidask_check['VIOLATIONMSG']\
                    .apply(lambda x: x[x.find("BID1=")+5: x.find(",")]),
            np.where((valid_bidask_check['askearly'] == 1) & (valid_bidask_check['ASK_OR_BID'] == "ASK1"),
                valid_bidask_check['VIOLATIONMSG']\
                     .apply(lambda x: x[x.find("BID1=")+4: x.find("]")]), np.nan)))))

        
        valid_bidask_check['ASK'] = pd.to_numeric(valid_bidask_check['ASK'], errors='coerce')
        valid_bidask_check['BID'] = pd.to_numeric(valid_bidask_check['BID'], errors='coerce')
        valid_bidask_check.drop(columns='askearly', inplace=True)
        
        valid_bidask_check_ask = valid_bidask_check.drop(columns='BID')
        valid_bidask_check_ask.rename({'ASK': 'VIOLATION_VALUE'}, axis=1, inplace=True)
        
        valid_bidask_check_bid = valid_bidask_check.drop(columns='ASK')
        valid_bidask_check_bid['ASK_OR_BID'] = np.where(valid_bidask_check_bid['ASK_OR_BID'] == "ASK", "BID", "BID1")
        valid_bidask_check_bid.rename({'BID': 'VIOLATION_VALUE'}, axis=1, inplace=True)
        
        valid_bidask_check = pd.concat([valid_bidask_check_ask, valid_bidask_check_bid], ignore_index=True)
        del  valid_bidask_check_ask, valid_bidask_check_bid
        
    else:
        valid_bidask_check = pd.DataFrame()

####################################################################################################
    violation_values = pd.concat([ar_rr_check, zero_check, negative_check], ignore_index=True)
    del ar_rr_check, zero_check, negative_check

    violation_values_ask = violation_values.query("(ASK_OR_BID == 'ASK') | (ASK_OR_BID == 'ASK1')")
    # violation_values_ask = violation_values_ask.iloc[:, :-2]
    violation_values_ask.drop(columns=['BID', 'MID'], inplace=True)
    violation_values_ask.rename({'ASK': 'VIOLATION_VALUE'}, axis=1, inplace=True)

    violation_values_bid = violation_values.query("(ASK_OR_BID == 'BID') | (ASK_OR_BID == 'BID1')")
    violation_values_bid.drop(columns=['MID', 'ASK'], inplace=True)
    violation_values_bid.rename({'BID': 'VIOLATION_VALUE'}, axis=1, inplace=True)

    violation_values_mid = violation_values.query("(ASK_OR_BID == 'MID') | (ASK_OR_BID == 'MID1')")
    violation_values_mid.drop(columns=['ASK', 'BID'], inplace=True)
    violation_values_mid.rename({'MID': 'VIOLATION_VALUE'}, axis=1, inplace=True)

    violation_values = pd.concat([violation_values_ask, violation_values_bid, violation_values_mid,
                                  valid_bidask_check], ignore_index=True)
    del violation_values_ask, violation_values_bid, violation_values_mid, valid_bidask_check

    violation_values = violation_values.assign(VIOLATION=1)

    violation_values = violation_values[['RECORDDATE', 'RATEID', 'RATELABEL', 'LABEL', 'RULENAME',
                                         'ASK_OR_BID', 'VIOLATION', 'VIOLATION_VALUE',
                                         'VIOLATIONMSG', 'VCGCOMMENT', 'USERCOMMENT', 'LABEL2',
                                         'LABEL1']]

    violation_values['RATEID'] = pd.to_numeric(violation_values['RATEID'], errors='coerce')

    violation_values['ASK_OR_BID'] = np.where(violation_values['ASK_OR_BID'] == "ASK1", "ASK",
            np.where(violation_values['ASK_OR_BID'] == "BID1", "BID",
            np.where(violation_values['ASK_OR_BID'] == "MID1", "MID", violation_values['ASK_OR_BID'])))
    
    violation_values['FILEDATE'] = pd.to_datetime(curDate, errors="coerce")
    
    violation_values = violation_values[['FILEDATE', 'RECORDDATE', 'RATEID', 'RATELABEL', 'LABEL',
                        'RULENAME', 'ASK_OR_BID', 'VIOLATION_VALUE', 'USERCOMMENT', 'VCGCOMMENT']]
###################################################################################################
  
violation_data = violation_values\
        .query("RULENAME not in ['ABSOLUTE TOLERANCE CHECK', 'RELATIVE TOLERANCE CHECK']")
violation_data = violation_data[['FILEDATE', 'RATELABEL', 'RATEID', 'ASK_OR_BID', 'RULENAME']]\
        .sort_values(['RATELABEL', 'RULENAME'], ignore_index=True)
violation_data.drop_duplicates(inplace=True, ignore_index=True)
  
violation_data['ZERO_CHECK'] = violation_data["RULENAME"].str.contains("ZERO").astype(int)
violation_data['NEGATIVE_CHECK'] = violation_data["RULENAME"].str.contains("NEGATIVE").astype(int)
violation_data['VALID_BIDASK_CHECK'] = violation_data["RULENAME"].str.contains("VALID").astype(int)

violation_data = violation_data.groupby(['FILEDATE', 'RATEID', 'ASK_OR_BID'], as_index=False)\
        [['ZERO_CHECK', 'NEGATIVE_CHECK', 'VALID_BIDASK_CHECK']].sum()

violation_data.rename({"FILEDATE": "RECORDDATE", "RATEID": "RELIABLEID"}, axis=1, inplace=True)

violation_data['RELIABLEID'] = pd.to_numeric(violation_data['RELIABLEID'], errors='coerce')

######## TP_FP Flags

violated_values = violation_values[['FILEDATE', 'RATEID', 'RATELABEL',
                                     'ASK_OR_BID', 'VIOLATION_VALUE']]
violated_values.drop_duplicates(inplace=True, ignore_index=True)

BD = currDayReliable.query("ASSETTYPE == 'BD'")
BD = AddQuotationBD(BD)
BD = SingleAskBidColumnBD(BD)

FX = currDayReliable.query("ASSETTYPE == 'FX'")
FX = SingleAskBidColumnFX(FX)

EQ = currDayReliable.query("ASSETTYPE == 'EQ'")
EQ = SingleAskBidColumnEQ(EQ)

IR = currDayReliable.query("ASSETTYPE == 'IR'")
IR = SingleAskBidColumnIR(IR)

CM = currDayReliable.query("ASSETTYPE == 'CM'")
CM = SingleAskBidColumnCM(CM)

cleanValue = pd.concat([BD, blankQuotation, FX, EQ, IR, CM], ignore_index=True)
del BD, blankQuotation, FX, EQ, IR, CM

cleanValue = cleanValue[['RECORDDATE', 'RELIABLEID', 'RATELABEL', 'ASKNEW', 'BIDNEW', 'MIDNEW']]

askClean = cleanValue[['RECORDDATE', 'RELIABLEID', 'RATELABEL', 'ASKNEW']]
askClean = askClean.assign(ASK_OR_BID="ASK")
askClean.rename({"ASKNEW": "CLEAN_VALUE"}, axis=1, inplace=True)

bidClean = cleanValue[['RECORDDATE', 'RELIABLEID', 'RATELABEL', 'BIDNEW']]
bidClean = bidClean.assign(ASK_OR_BID="BID")
bidClean.rename({"BIDNEW": "CLEAN_VALUE"}, axis=1, inplace=True)

midClean = cleanValue[['RECORDDATE', 'RELIABLEID', 'RATELABEL', 'MIDNEW']]
midClean = midClean.assign(ASK_OR_BID="MID")
midClean.rename({"MIDNEW": "CLEAN_VALUE"}, axis=1, inplace=True)

cleanValue = pd.concat([askClean, bidClean, midClean], ignore_index=True)
del askClean, bidClean, midClean

violation_clean_value = pd.merge(
    violated_values, cleanValue.rename({'RECORDDATE': 'FILEDATE', 'RELIABLEID': 'RATEID'}, axis=1),
        on = ['FILEDATE', 'RATEID', 'ASK_OR_BID'], how="left")  # or "outer"? (too many rows, though)
#        right_on = ["RECORDDATE", "RELIABLEID", "ASK_OR_BID"]

violation_clean_value = violation_clean_value.assign(RATELABEL=violation_clean_value['RATELABEL_x'])
violation_clean_value.drop(columns=["RATELABEL_x", "RATELABEL_y"], inplace=True)

if config['is_debug']:
    violation_clean_value.to_csv("Module1_DIM_violation_clean_value.csv", index=False)

violation_clean_value['VIOLATION_VALUE'] = pd.to_numeric(violation_clean_value['VIOLATION_VALUE'],
        errors="coerce").round(5)
violation_clean_value['CLEAN_VALUE'] = pd.to_numeric(violation_clean_value['CLEAN_VALUE'],
        errors="coerce").round(5)

violation_clean_value.drop_duplicates(inplace=True, ignore_index=True)

df_agg = violation_clean_value.groupby(['FILEDATE', 'RATEID', 'ASK_OR_BID'])\
       .size().reset_index(name='NROW')
violation_clean_value = pd.merge(df_agg, violation_clean_value, on=['FILEDATE', 'RATEID', 'ASK_OR_BID'], how='outer')

violation_clean_value = violation_clean_value.assign(Remove=np.where((violation_clean_value['NROW'] > 1) & violation_clean_value['VIOLATION_VALUE'].isna(), 1, 0))

violation_clean_value = violation_clean_value.query("Remove == 0").drop(columns=['NROW', 'Remove'])

#### Rolling up check level data to ask and bid level as per DIM format
violation_data_all = violation_values[['FILEDATE', 'RATELABEL', 'RATEID', 'ASK_OR_BID',
    'RULENAME', 'LABEL']].sort_values(['RATEID', 'RULENAME'], ignore_index=True)
violation_data_all.drop_duplicates(inplace=True, ignore_index=True)

violation_data_all['ZERO_CHECK'] = np.where(violation_data_all['RULENAME']\
        .str.contains("ZERO"), 1, 0)
violation_data_all['NEGATIVE_CHECK'] = np.where(violation_data_all['RULENAME'].str.contains("NEGATIVE"), 1, 0)
violation_data_all['VALID_BIDASK_CHECK'] = np.where(violation_data_all['RULENAME'].str.contains("VALID"), 1, 0)
violation_data_all['AR_CHECK'] = np.where(violation_data_all['RULENAME'].str.contains("ABSOLUTE"), 1, 0)
violation_data_all['RR_CHECK'] = np.where(violation_data_all['RULENAME'].str.contains("RELATIVE"), 1, 0)

violation_data_all = violation_data_all\
        .groupby(['FILEDATE', 'RATEID', 'ASK_OR_BID', 'LABEL'], as_index=False)\
                [['ZERO_CHECK', 'NEGATIVE_CHECK', 'VALID_BIDASK_CHECK', 'AR_CHECK', 'RR_CHECK']]\
                .sum()
violation_data_all.rename({"FILEDATE": "RECORDDATE", "RATEID": "RELIABLEID"}, axis=1, inplace=True)

violation_data_all['RELIABLEID'] = pd.to_numeric(violation_data_all['RELIABLEID'], errors="coerce")

violation_data_all = pd.merge(violation_data_all,
        violation_clean_value.rename({'FILEDATE': 'RECORDDATE', 'RATEID': 'RELIABLEID'}, axis=1),
        on = ["RECORDDATE", "RELIABLEID", "ASK_OR_BID"], how="outer") # or left?

if config['is_debug']:
    violation_data_all.to_csv("Module1_DIM_violation_data_all.csv", index=False)

### Extract filter criteria for comments

violationComments = violation_values.copy()

violationComments = violationComments.assign(Final_Comment=np.where(
    violationComments['VCGCOMMENT'].isna() | (violationComments['VCGCOMMENT'] == ""),
    violationComments['USERCOMMENT'], violationComments['VCGCOMMENT']))

violationComments = violationComments.assign(Chlngng_Quote_Provider=np.where(
    violationComments['Final_Comment'].str.contains("challenging quote provider", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Chlngng_Source_Provider=np.where(
    violationComments['Final_Comment'].str.contains("challenging source provider", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Creat=np.where(
    violationComments['Final_Comment'].str.contains("creat", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Current_rates=np.where(
    violationComments['Final_Comment'].str.contains("current rates", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Due_To_Market=np.where(
    violationComments['Final_Comment'].str.contains("due to market", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Inline_With_Mkt=np.where(
    violationComments['Final_Comment'].str.contains("inline with market", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Matur=np.where(
    violationComments['Final_Comment'].str.contains("matur", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(No_Closing=np.where(
    violationComments['Final_Comment'].str.contains("no closing", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(No_Source=np.where(
    violationComments['Final_Comment'].str.contains("no source", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Only_Source_Updating=np.where(
    violationComments['Final_Comment'].str.contains("only source updating", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Roll=np.where(
    violationComments['Final_Comment'].str.contains("roll", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Unable_To_Find_Good=np.where(
    violationComments['Final_Comment'].str.contains("unable to find good", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Verified_Agnst_Source=np.where(
    violationComments['Final_Comment'].str.contains("verified against source", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Inline_Nghbrng_Tenor=np.where(
    violationComments['Final_Comment'].str.contains("in line with neighbouring tenors",
        flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Broker_Inline=np.where(
    violationComments['Final_Comment'].str.contains("brokers are quoting in line", flags=re.I, na=False),
    True, False))

violationComments = violationComments.assign(Good_Others=np.where(
    violationComments['Final_Comment'].str.contains("Good - Others", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Bad_Others=np.where(
    violationComments['Final_Comment'].str.contains("Bad - Others", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(OIS_Proj=np.where(
    violationComments['Final_Comment'].str.contains("OIS Proj", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Negative_Rates=np.where(
    violationComments['Final_Comment'].str.contains("Negative Rates", flags=re.I, na=False), True, False))

violationComments = violationComments.assign(Good_ReCal_RateGrp=np.where(
    violationComments['Final_Comment'].str.contains("Good - .* Recalculation of rate group",
        flags=re.I, na=False), True, False))


containCols = ["Chlngng_Quote_Provider", "Chlngng_Source_Provider", "Creat", "Current_rates",
               "Due_To_Market", "Inline_With_Mkt", "Matur", "No_Closing", "No_Source",
               "Only_Source_Updating", "Roll", "Unable_To_Find_Good", "Verified_Agnst_Source",
               "Inline_Nghbrng_Tenor", "Broker_Inline", "Good_Others", "Bad_Others", "OIS_Proj",
               "Negative_Rates", "Good_ReCal_RateGrp"]

violationComments = violationComments.assign(Contains_Count=violationComments[containCols].sum(axis=1))

violationComments = violationComments.assign(Filter_Criteria=violationComments[containCols]
        .apply(lambda row: row[row==True].index[0] if (row==True).any() else "_None_", axis=1))

violationComments['Filter_Criteria'] = violationComments['Filter_Criteria']\
        .where(violationComments['Contains_Count'] <= 1, "Multiple")

violationComments.drop(columns=['RECORDDATE', 'LABEL', 'RULENAME'], inplace=True)

violationComments['VIOLATION_VALUE'] = pd.to_numeric(violationComments['VIOLATION_VALUE'],
        errors="coerce").round(5)

## Removing duplicate rows

df_agg = violationComments.groupby(['FILEDATE', 'RATEID', 'ASK_OR_BID'])\
       .size().reset_index(name='Rows')
violationComments = pd.merge(df_agg, violationComments, on=['FILEDATE', 'RATEID', 'ASK_OR_BID'], how='outer')

violationComments = violationComments.assign(Remove=np.where(
    (violationComments['Rows'] > 1) & violationComments['VIOLATION_VALUE'].isna(), 1, 0))
violationComments = violationComments.query("Remove == 0").drop(columns=['Rows', 'Remove'])
violationComments.drop_duplicates(inplace=True, ignore_index=True)

df_agg = violationComments.groupby(['FILEDATE', 'RATEID', 'ASK_OR_BID'])\
        .size().reset_index(name='Rows')
violationComments = pd.merge(df_agg, violationComments, on=['FILEDATE', 'RATEID', 'ASK_OR_BID'], how='outer')

violationComments = violationComments.assign(Remove=np.where(
    (violationComments['Rows'] > 1) & (violationComments['Final_Comment'] == ""), 1, 0))

violationComments = violationComments.query("Remove == 0")\
        .drop(columns=['Rows', 'Remove', 'USERCOMMENT', 'VCGCOMMENT'])
violationComments.drop(columns='VIOLATION_VALUE', inplace=True)
violationComments.drop_duplicates(inplace=True, ignore_index=True)

#### Merge with violation values data

violation_data_all = pd.merge(violation_data_all,
        violationComments.rename({'FILEDATE': 'RECORDDATE', 'RATEID': 'RELIABLEID'}, axis=1),
        on = ["RECORDDATE", "RELIABLEID", "ASK_OR_BID"], how="outer") # or left?

violation_data_all = violation_data_all.assign(RATELABEL=violation_data_all['RATELABEL_x'])
violation_data_all.drop(columns=["RATELABEL_x", "RATELABEL_y"], inplace=True)


if config['is_debug']:
    violation_data_all.to_csv("Module1_DIM_violation_data_all_2.csv", index=False)
