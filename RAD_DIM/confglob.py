#############################################
# print(curDate)
# curDate = "2019-04-17"
curDate = "2020-09-01"
# print(curDate)
#############################################
import pandas as pd
daydatemapping = pd.DataFrame(columns=['Day', 'n'])
daydatemapping['Day'] = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daydatemapping['n'] = range(0,7)
#############################################
import datetime as dt
from dateutil.parser import parse
threshold_switchdate = "2018-09-28"
#Before "2018-09-28", it is weekly updated, after that, it is daily updated
if parse(curDate) < parse(threshold_switchdate):
  staticDay = "Tuesday"
else:
  staticDay = curDate
############################################
acceptableNumDaysStaleCheck = 90
staleValueNumDays = 2
staleValueThreshold = 0
decimalPlace = 5
############################################
cPrice = "PRICE"
cYield = "YIELD"
cSpread = "SPREAD"
cBond = "BD"
cForex = "FX"
cEquity = "EQ"
cInterestRate = "IR"
cCommodity = "CM"

instrumentsAskBidMid = ["FX_SMILE"]

config = {'is_debug': True, 'outputPath': "./", "inputPath": "../../input/"}
reqAssetTypes = ["BD","FX","EQ","IR", "CM"]
derivedinstrument = ['MRS_VD', 'FX_SWAP_IMPLIED', 'MRS', 'BD_GOVT', 'BD_CORP', 'BD_SDS', 'BD_FUTURES']
