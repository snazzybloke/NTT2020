####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: Data Investigation Module Commodity specific functions.
This is the seventh script for the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is to define a single function, called from the Trigger
and the RMS_Flags_Function scripts.
This is an R-to-Python translation of the script with the same name.
"""

def SingleAskBidColumnCM(mergedData):
    """
    Summary:
    -----------
    Called from the Trigger and RMS_Flags_Function scripts on the current day
    interest rate data (mergedData).
    Creates columns ASKNEW, BIDNEW and MIDNEW initialised with the ASK, BID
    and MID values, respectively.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedData : DataFrame
        current day commodity data

    Returns:
    -----------
    mergedData : DataFrame
        current day commodity data with ASKNEW, BIDNEW and MIDNEW columns
    """

    mergedData = mergedData.assign(ASKNEW=mergedData['ASK'])
    mergedData = mergedData.assign(BIDNEW=mergedData['BID'])
    mergedData = mergedData.assign(MIDNEW=mergedData['MID'])
    return mergedData
