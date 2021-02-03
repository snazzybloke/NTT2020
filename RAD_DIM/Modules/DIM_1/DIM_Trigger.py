####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

"""
This is the Trigger script for the Data Investigation Module (DIM), which
carries out the step-by-step execution of the associated tasks.
The purpose of this module is to identify rates that show data issues
through multiple checks and relevant business rules.
This is a verbatim R to Python translation of the script with the same name.
Note that the corresponding directory needs to be renamed DIM_1
(i.e., not the 1_DIM as in R), which is an acceptable name to Python.
This script is supposed to be invoked from the Master script in the
Reliable_Framework directory. All the importing takes place relative
to that directory. 


  Input Files        :                                                                    
                       1. currentDayDataFile  : Current day unclean reliable              
                       2. relHistFile         : Reliable historical clean                 
                       3. previousDayDataFile : Previous day clean reliable               
                       4. stockEventDataFile  : Stock event data for reliable             
                       5. thresholds          : RMS thresholds for valid rates            
  Output Files       :                                                                    
                       1. newRates            : New Rates in a day                        
                       2. matureBD            : Matured or maturing rates                 
                       3. blankQuotation      : Bond rates with no quotation              
                       4. Updated_Hist        : Historical data updated after stock event 
                       5. DIM_OUTPUT          : Output of DIM framework with flags        
  Steps              :                                                                    
                      1 . Sourcing DIM input parameters                                   
                      2 . Loading all DIM Functions                                       
                      3 . Fetch Files from Maria DB                                       
                      4 . Load unclean data from R server                                 
                      5 . Filter for new rates                                            
                      6 . Prepare bonds data for running DIM checks                       
                      7 . Perform DIM checks for bonds                                    
                      8 . Prepare forex data for running DIM checks                       
                      9 . Perform DIM checks on forex                                     
                      10. Prepare equities data for running DIM checks                    
                      11. Perform DIM checks on equities                                  
                      12. Prepare interest rates data for running DIM checks              
                      13. Perform DIM checks on interest rates                            
                      14. Prepare commodities data for running DIM checks                 
                      15. Perform DIM checks on commodities                               
                      16. Export outputs

Functions:
* addRate(df, rate_mapping_df, xRateIDField, yRateIDField, how = 'left')
* reassignRateLabel(df)
"""

from confglob import *
import pandas as pd

#------------------------------------------------#
# 1 : LOADING INPUT PARAMETERS                   #
#------------------------------------------------#
print("1 : LOADING INPUT PARAMETERS  ")

# 1.1 Source Input parameters for DIM Data
# source("./Modules/1_DIM/Utility/DIM_InputParameters.R")
# print(f"curDD: {curDate}")
# print(f"ip.threshRelMID2: {ip.threshRelMID2}")

from Modules.DIM_1.Utility.DIM_InputParameters import *

#------------------------------------------------#
# 2 : Loading DIM Functions                      #
#------------------------------------------------#
print("2 : Loading DIM Functions   ")

# 2.1 Source common functions used used in DIM
# source("./Modules/1_DIM/Utility/DIM_Common_Functions.R")
# import  Modules.DIM_1.Utility.DIM_Common_Functions as cf
from Modules.DIM_1.Utility.DIM_Common_Functions import *

# 2.2  Source Bonds specific functions in DIM
# source("./Modules/1_DIM/Utility/DIM_BD_Functions.R")
from Modules.DIM_1.Utility.DIM_BD_Functions import AddQuotationBD, SingleAskBidColumnBD,\
        IdentifyMaturedRatesBD, IdentifyMaturingRatesBD, getMatureRatesBD, removeMatureBD,\
        blankQuotation

# 2.3 Source Forex specific functions in DIM
# source("./Modules/1_DIM/Utility/DIM_FX_Functions.R")
from Modules.DIM_1.Utility.DIM_FX_Functions import SingleAskBidColumnFX

# 2.4 Source Equities specific functions in DIM
# source("./Modules/1_DIM/Utility/DIM_EQ_Functions.R")
from Modules.DIM_1.Utility.DIM_EQ_Functions import SingleAskBidColumnEQ

# 2.5 Source Interest Rates specific functions in DIM
# source("./Modules/1_DIM/Utility/DIM_IR_Functions.R")
from Modules.DIM_1.Utility.DIM_IR_Functions import SingleAskBidColumnIR

# 2.6 Source Commodities specific functions in DIM
# source("./Modules/1_DIM/Utility/DIM_CM_Functions.R")
from Modules.DIM_1.Utility.DIM_CM_Functions import SingleAskBidColumnCM

#------------------------------------------------#
# 3 : Fetch Files from DB                        #
#------------------------------------------------#
print("3 : Fetch Files from DB")

# 3.1 Connect to MariaDB
# source("./DB_Scripts/Config/ConnectMDB.R")
from DB_Scripts.Config.ConnectMDB import *

# 3.2 Fetch historical reliable clean data
# source("./DB_Scripts/Data_Read_Write/FetchHistRelCleanData.R")
from DB_Scripts.Data_Read_Write.FetchHistRelCleanData import get_fetchTable_historical_golden_rates

relHistorical = get_fetchTable_historical_golden_rates("../../")
relHistorical['RECORDDATE'] = pd.to_datetime(relHistorical['RECORDDATE'], errors="coerce")
dmax = relHistorical['RECORDDATE'].dt.date.max()
prevDayReliable = relHistorical.query("RECORDDATE == @dmax")
prevDayReliable.rename({c: "G." + c for c in ("ASK", "BID", "ASK1", "BID1", "ASK2", "BID2")},
        axis=1, inplace=True)

# 3.3 Fetch stock event data
# source("./DB_Scripts/Data_Read_Write/FetchStockEventData.R")
from DB_Scripts.Data_Read_Write.FetchStockEventData import *

#------------------------------------------------#
# 4 : LOAD DATA AND FILTER ASSET TYPES           #
#------------------------------------------------#
print("4 : LOAD DATA AND FILTER ASSET TYPES")

# 4.1 Load input data to R server
# source("./Modules/1_DIM/Utility/DIM_LoadData.R")
from Modules.DIM_1.Utility.DIM_LoadData import *

relHistorical = relHistorical.query("RECORDDATE < @curDate")

# 4.3 Filtering current day data for required asset types
currentDayData = currentDayData.query("ASSETTYPE in @reqAssetTypes")
# NOTE: logFile only in R: ./Utility/common.R:logFile <- function(df, name2Log)
# logFile(currentDayData, 'DIM_4_currentDayData.csv')

#------------------------------------------------#
# 5 : FILTER FOR NEW RATES                       #
#------------------------------------------------#
print("5 : FILTER FOR NEW RATES ")

## 5.1 - Creating Flag for NEW RATES
currentDayData = IdentifyNewRates(currentDayData, relHistorical)

## 5.2 - Creating Subset for NEW RATES if it reliableid not in historical.
newRates = NewRatesSubset(currentDayData, currDayReliable)

## 5.3 - Creating subset for the rates to be merged: The new rates will be removed.
ratesForChecks = RatesForChecksSubset(currentDayData)

# logFile(currentDayData, 'DIM_5_currentDayData.csv')
# logFile(newRates, 'DIM_5_newRates.csv')
# logFile(ratesForChecks, 'DIM_5_ratesForChecks.csv')

################################################################################
##                                 DIM FOR BD                                 ##
################################################################################

#------------------------------------------------#
# 6 : BD data preparation for DIM Checks         #
#------------------------------------------------#
print("6 : BD data preparation for DIM Checks ")

## 6.1 - Get BD rates for current day
currentDayDataBD = ratesForChecks.query("ASSETTYPE == @cBond")

## 6.2 - Get historical data for BD
relHistoricalBD = relHistorical.query("ASSETTYPE == @cBond")

## 6.3 - Removing weekends data in historical data
relHistoricalBD = RemoveWeekends(relHistoricalBD)

## 6.4 - Get BD rates from previous day's cleaned reliable data
prevDayReliableBD = prevDayReliable.query("ASSETTYPE == @cBond")

## 6.5 - Getting properties for current day from reliable data
mergedDataBD = GetProperties(currentDayDataBD, prevDayReliableBD)

## 6.6 - Adding Quotation column to merge values in single ask and bid column
mergedDataBD = AddQuotationBD(mergedDataBD)

## 6.7 - Adding ASKNEW and BIDNEW columns for merged values from different columns in current day
mergedDataBD = SingleAskBidColumnBD(mergedDataBD)
from Modules.DIM_1.Utility.DIM_BD_Functions import blankQuotation
curDayBlankQuotation = blankQuotation.copy()

## 6.8 - Creating Flag for MATURED RATES
mergedDataBD = IdentifyMaturedRatesBD(mergedDataBD)

## 6.9 - Creating Flag for MATURING RATES
mergedDataBD = IdentifyMaturingRatesBD(mergedDataBD)

## 6.10 - Subset mature and maturing rates 
matureBD = getMatureRatesBD(mergedDataBD)

## 6.11 - Remove mature and maturing rates 
mergedDataBD = removeMatureBD(mergedDataBD)

## 6.12 - Subsetting for latest 14 days of data in historical data
relHistoricalForStaleBD = FilterRawData(relHistoricalBD, curDate)

## 6.13 - Getting historical data only for rates which are present on current day
relHistoricalForStaleBD = RemoveExpiredFromHist(mergedDataBD, relHistoricalForStaleBD)

## 6.14 - STALE VALUE CHECK
mergedDataBD = StaleCheck(relHistoricalForStaleBD, mergedDataBD, staleValueThreshold)

## 6.15 - CREATING COUNTER FOR STALE VALUE
mergedDataBD = Stale_Counter(mergedDataBD, relHistoricalBD)

################################################################################
##                                 DIM FOR FX                                 ##
################################################################################

#------------------------------------------------#
# 7 : FX data preparation for DIM Checks         #
#------------------------------------------------#
print("7 : FX data preparation for DIM Checks")

## 7.1 - Get FX rates for current day
currentDayDataFX = ratesForChecks.query("ASSETTYPE == @cForex")

## 7.2 - Get historical data for FX
relHistoricalFX = relHistorical.query("ASSETTYPE == @cForex")

## 7.3 - Removing weekends data in historical data
relHistoricalFX = RemoveWeekends(relHistoricalFX)

## 7.4 - Get FX rates from previous day's cleaned reliable data
prevDayReliableFX = prevDayReliable.query("ASSETTYPE == @cForex")

## 7.5 - Getting properties for current day from reliable data
mergedDataFX = GetProperties(currentDayDataFX, prevDayReliableFX)

## 7.6 - Adding ASKNEW and BIDNEW columns for merged values from different columns in current day
mergedDataFX = SingleAskBidColumnFX(mergedDataFX)

## 7.7 - Subsetting for latest 14 days of data in historical data
relHistoricalForStaleFX = FilterRawData(relHistoricalFX, curDate)

## 7.8 - Getting historical data only for rates which are present on current day
relHistoricalForStaleFX = RemoveExpiredFromHist(mergedDataFX, relHistoricalForStaleFX)

## 7.9 - STALE VALUE CHECK
mergedDataFX = StaleCheck(relHistoricalForStaleFX, mergedDataFX, staleValueThreshold)

## 7.10 - CREATING COUNTER FOR STALE VALUE
mergedDataFX = Stale_Counter(mergedDataFX, relHistoricalFX)

################################################################################
##                                 DIM FOR EQ                                 ##
################################################################################

#------------------------------------------------#
# 8 : EQ data preparation for DIM Checks         #
#------------------------------------------------#
print("8 : EQ data preparation for DIM Checks")

## 8.1 - Get EQ rates for current day
currentDayDataEQ = ratesForChecks.query("ASSETTYPE == @cEquity")

## 8.2 - Get historical data for EQ
relHistoricalEQ = relHistorical.query("ASSETTYPE == @cEquity")

## 8.3 - Removing weekends data in historical data
relHistoricalEQ = RemoveWeekends(relHistoricalEQ)

## 8.4 - Get EQ rates from previous day's cleaned reliable data
prevDayReliableEQ = prevDayReliable.query("ASSETTYPE == @cEquity")

## 8.5 - Getting properties for current day from reliable data
mergedDataEQ = GetProperties(currentDayDataEQ, prevDayReliableEQ)

## 8.6 - Creating ASKNEW and BIDNEW columns for merged values from different columns in current day
mergedDataEQ = SingleAskBidColumnEQ(mergedDataEQ)

## 8.7 - Subsetting for latest 14 days of data in historical data
relHistoricalForStaleEQ = FilterRawData(relHistoricalEQ, curDate)

## 8.8 - Getting historical data only for rates which are present on current day
relHistoricalForStaleEQ = RemoveExpiredFromHist(mergedDataEQ, relHistoricalForStaleEQ)

## 8.9 - STALE VALUE CHECK
mergedDataEQ = StaleCheck(relHistoricalForStaleEQ, mergedDataEQ, staleValueThreshold)

## 8.10 - CREATING COUNTER FOR STALE VALUE
mergedDataEQ = Stale_Counter(mergedDataEQ, relHistoricalEQ)

################################################################################
##                                 DIM FOR IR                                 ##
################################################################################
#------------------------------------------------#
# 9 : IR data preparation for DIM Checks        #
#------------------------------------------------#
print("9 : IR data preparation for DIM Checks")

## 9.1 - Get IR rates for current day
currentDayDataIR = ratesForChecks.query("ASSETTYPE == @cInterestRate")

## 9.2 - Get historical data for IR
relHistoricalIR = relHistorical.query("ASSETTYPE == @cInterestRate")

## 9.3 - Removing weekends data in historical data
relHistoricalIR = RemoveWeekends(relHistoricalIR)

## 9.4 - Get IR rates from previous day's cleaned reliable data
prevDayReliableIR = prevDayReliable.query("ASSETTYPE == @cInterestRate")

## 9.5 - Getting properties for current day from reliable data
mergedDataIR = GetProperties(currentDayDataIR, prevDayReliableIR)

## 9.6 - Adding ASKNEW and BIDNEW columns for merged values from different columns in current day
mergedDataIR = SingleAskBidColumnIR(mergedDataIR)

## 9.7 - Subsetting for latest 14 days of data in historical data
relHistoricalForStaleIR = FilterRawData(relHistoricalIR, curDate)

# logFile(mergedDataIR, "DIM_9_mergedDataIR.csv")
# logFile(relHistoricalForStaleIR, "DIM_9_relHistoricalForStaleIR.csv")

## 9.8 - Getting historical data only for rates which are present on current day
relHistoricalForStaleIR = RemoveExpiredFromHist(mergedDataIR, relHistoricalForStaleIR)

## 9.9 - STALE VALUE CHECK
mergedDataIR = StaleCheck(relHistoricalForStaleIR, mergedDataIR, staleValueThreshold)

## 9.10 - CREATING COUNTER FOR STALE VALUE
mergedDataIR = Stale_Counter(mergedDataIR, relHistoricalIR)

################################################################################
##                                 DIM FOR CM                                 ##
################################################################################
#-------- ---------------------------------------#
# 10 : CM data preparation for DIM Checks        #
#--------- --------------------------------------#
print("10 : CM data preparation for DIM Checks")

## 10.1 - Get CM rates for current day
currentDayDataCM = ratesForChecks.query("ASSETTYPE == @cCommodity")

## 10.2 - Get historical data for CM
relHistoricalCM = relHistorical.query("ASSETTYPE == @cCommodity")

## 10.3 - Removing weekends data in historical data
relHistoricalCM = RemoveWeekends(relHistoricalCM)

## 10.4 - Get CM rates from previous day's cleaned reliable data
prevDayReliableCM = prevDayReliable.query("ASSETTYPE == @cCommodity")

## 10.5 - Getting properties for current day from reliable data
mergedDataCM = GetProperties(currentDayDataCM, prevDayReliableCM)

## 10.6 - Adding ASKNEW and BIDNEW columns for merged values from different columns in current day
mergedDataCM = SingleAskBidColumnCM(mergedDataCM)

## 10.7 - Subsetting for latest 14 days of data in historical data
relHistoricalForStaleCM = FilterRawData(relHistoricalCM, curDate)

## 10.8 - Getting historical data only for rates which are present on current day
relHistoricalForStaleCM = RemoveExpiredFromHist(mergedDataCM, relHistoricalForStaleCM)

## 10.9 - STALE VALUE CHECK
mergedDataCM = StaleCheck(relHistoricalForStaleCM, mergedDataCM, staleValueThreshold)

## 10.10 - CREATING COUNTER FOR STALE VALUE
mergedDataCM = Stale_Counter(mergedDataCM, relHistoricalCM)

#------------------------------------------------#
# 11 : Export Outputs                            #
#------------------------------------------------#
def addRate(df, rate_mapping_df, xRateIDField, yRateIDField, how = 'left'):
    df = pd.merge(df, rate_mapping_df.drop_duplicates().rename({yRateIDField: xRateIDField}, axis=1),
            on = xRateIDField, how=how)
    return df

print("11 : Export Outputs  ")
print("11.1 Merge all asset classes")

mergedData = pd.concat([mergedDataBD, mergedDataFX, mergedDataEQ, mergedDataIR, mergedDataCM],
        ignore_index=True)

# logFile(mergedData, "DIM_11_mergedData.csv")
# Normalize the ratelabel
mergedData.drop(columns='RATELABEL', inplace=True)
# source("./DB_Scripts/Data_Read_Write/FetchMasterRate.R")
from DB_Scripts.Data_Read_Write.FetchMasterRate import fetchMasterRate
rateMapping = fetchMasterRate("../../")
rateMapping = rateMapping[['RATEID', 'RATELABEL']]
rateMapping['RATEID'] = pd.to_numeric(rateMapping['RATEID'], errors="coerce")
mergedData['RELIABLEID'] = pd.to_numeric(mergedData['RELIABLEID'], errors="coerce")
mergedData = addRate(mergedData, rateMapping, 'RELIABLEID', 'RATEID')

# logFile(mergedData, "DIM_11_mergedData_addRate.csv")
# logFile(rmsThresholds, "DIM_11_rmsThresholds.csv")

print("11.2 Setting the format on ask, bid and mid level")
DIM_OUTPUT = SetFormat(mergedData, rmsThresholds)
DIM_OUTPUT['RELIABLEID'] = DIM_OUTPUT['RELIABLEID'].astype(str)
# logFile(DIM_OUTPUT, "DIM_11_DIM_OUTPUT.csv")

print("11.3 Set RMS flags")
# source("./Modules/1_DIM/Utility/RMS_Flags_Function.R")
from Modules.DIM_1.Utility.RMS_Flags_Function import *
violation_data_all['RELIABLEID'] = violation_data_all['RELIABLEID'].astype(str)
DIM_OUTPUT = rmsdimFlags(DIM_OUTPUT, violation_data_all)

def reassignRateLabel(df):
    df = df.rename({'RATELABEL_x': 'RATELABEL'}, axis=1).drop(columns='RATELABEL_y')
    return df

DIM_OUTPUT = reassignRateLabel(DIM_OUTPUT)

# logFile(violation_data_all, "Module1_11.3_violation_data_all.csv")
# logFile(DIM_OUTPUT, "Module1_11.3_DIM_OUTPUT.csv")

print("11.4 Create TP and FP Flag")
DIM_OUTPUT = TP_FP_Flag(DIM_OUTPUT)
# logFile(DIM_OUTPUT, "Module1_11.4_DIM_OUTPUT.csv")

# print("11.5 CREATING FLAGS FOR ANOMALIES")
DIM_OUTPUT = FlagCreation(DIM_OUTPUT)

# logFile(DIM_OUTPUT, "Module1_11.5_DIM_OUTPUT.csv")

print("11.6 Exporting the DIM Output and New Rates file")
Export_Output(DIM_OUTPUT, newRates)

print("11.7 Exporting the Rates for bonds with blank quotation")
Export_Blank_Quotation(curDayBlankQuotation)

print("11.8 Export Mature Rates")
Export_Mature(matureBD)

print("11.9 Disconnect Maria DB")
# source("./DB_Scripts/Config/DisconnectMDB.R")
