####################################################
# Author: Ante Bilic                               #
# Since: Oct 22, 2020                              #
# Copyright: The RAD Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

""" Description: Data Investigation Module - Equities specific Functions
This is the fifth script for the Data Investigation Module (DIM) imported by the Trigger.
The purpose of this script is to define a single function, called from the Trigger
and the RMS_Flags_Function scripts.
This is an R-to-Python translation of the script with the same name.
"""

import numpy as np

def SingleAskBidColumnEQ(mergedData):
    """
    Summary:
    -----------
    Called from the Trigger and RMS_Flags_Function scripts on the current day
    equities data (mergedData).
    Creates columns ASKNEW, BIDNEW and MIDNEW initialised with the ASK, BID
    and MID values, respectively.
    This is an R-to-Python translation of the function with the same name.

    Parameters:
    -----------
    mergedData : DataFrame
        current day equities data

    Returns:
    -----------
    cp_mergedData : DataFrame
        a copy of current day equities data with ASKNEW, BIDNEW and MIDNEW columns
    """

    cp_mergedData = mergedData.copy()
    tmp = np.where(~(cp_mergedData['ASK'].isna() & cp_mergedData['BID'].isna()),
            cp_mergedData['ASK'], cp_mergedData['LAST'])
    cp_mergedData = cp_mergedData.assign(ASKSingle=tmp)
    tmp = np.where(~(cp_mergedData['ASK'].isna() & cp_mergedData['BID'].isna()),
            cp_mergedData['BID'], cp_mergedData['LAST'])
    cp_mergedData = cp_mergedData.assign(BIDSingle=tmp)
    cp_mergedData['MIDSingle'] = cp_mergedData['MID'].where(cp_mergedData['MID'].notna(),
            cp_mergedData['LAST'])
    tmp = np.where(~(cp_mergedData['ASKSingle'].isna() & cp_mergedData['BIDSingle'].isna()),
            cp_mergedData['ASKSingle'], cp_mergedData['CLOSING'])
    cp_mergedData = cp_mergedData.assign(ASKSingle=tmp)
    tmp = np.where(~(cp_mergedData['ASKSingle'].isna() & cp_mergedData['BIDSingle'].isna()),
            cp_mergedData['BIDSingle'], cp_mergedData['CLOSING'])
    cp_mergedData = cp_mergedData.assign(BIDSingle=tmp)
    cp_mergedData['MIDSingle'] = cp_mergedData['MIDSingle']\
            .where(cp_mergedData['MIDSingle'].notna(), cp_mergedData['CLOSING'])
    tmp = np.where(~(cp_mergedData['ASKSingle'].isna() & cp_mergedData['BIDSingle'].isna()),
            cp_mergedData['ASKSingle'], cp_mergedData['HISTORICCLOSE'])
    cp_mergedData = cp_mergedData.assign(ASKSingle=tmp)
    tmp = np.where(~(cp_mergedData['ASKSingle'].isna() & cp_mergedData['BIDSingle'].isna()),
            cp_mergedData['BIDSingle'], cp_mergedData['HISTORICCLOSE'])
    cp_mergedData = cp_mergedData.assign(BIDSingle=tmp)
    cp_mergedData['MIDSingle'] = cp_mergedData['MIDSingle']\
            .where(cp_mergedData['MIDSingle'].notna(), cp_mergedData['HISTORICCLOSE'])
    cp_mergedData['ASKNEW'] = cp_mergedData['ASKSingle']
    cp_mergedData['BIDNEW'] = cp_mergedData['BIDSingle']
    cp_mergedData['MIDNEW'] = cp_mergedData['MIDSingle']
    return cp_mergedData
