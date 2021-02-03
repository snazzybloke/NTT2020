"""Summary:
   -------
   Collection of functions called by both FX Options and CREDIT TRADING.
"""

####################################################
# Author: Ante Bilic                               #
# Since: Mar 27, 2020                              #
# Copyright: The PLC Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

import re
import pathlib
from calendar import month_abbr
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta, SA
from dateutil.parser import parse
from config.configs import getLogger
from config import configs
env = configs.env
logger = getLogger("common_futils_lcy.py")
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def get_clean_portfolio(curdate='20200108',
                        entity='DBSSG',
                        node_id='FX Options',
                        pub_holiday=False,
                        files=''):
    """
    Summary:
    --------
    Invoked for and 'CREDIT TRADING' and 'FX Options' node_id.
    Cleans and the *SENSI*TXT files and extracts from it only the rows relevant
    to the relevant trade desk (i.e., node_id).
        Sensi Date             Node Component ... Sensi Risk (OCY) DTD SGD Rate_move  Industry
    0  09-Jan-2020  DBSSG CRD ABSSP  IR Delta ...              0.0     0.0   0.00703       NaN
    1  09-Jan-2020  DBSSG CRD ABSSP  IR Delta ...              0.0     0.0   0.04205       NaN
    2  09-Jan-2020  DBSSG CRD ABSSP  IR Delta ...              0.0     0.0   0.03657       NaN
    3  09-Jan-2020  DBSSG CRD ABSSP  IR Delta ...              0.0     0.0   0.03089       NaN
    4  09-Jan-2020  DBSSG CRD ABSSP  IR Delta ...              0.0     0.0  -0.31999       NaN
    ...
    First, the Structure file (DBS portfolio structure in a given month) needs
    to be combined with another file nodeid_info (Node IDs with thresholds.csv)
    which has a column of our concern, 'NODEID'. Using the function argument
    node_id, we must filter the rows in nodeid_info where the col 'NODEID'
    matches the value of node_id. Then from these rows we extract the the
    level, node_level_id, which designates where in the trading desk hierarchy
    the desired node_id sits. Next, we combine the extracted information with
    the Structure file to filter only the relevant rows to subdesk_portfolio:
    (i) node_id value is used again, now to match the values of the column
    identified above (and saved in node_level_id) in the trading desk hierarchy
    and (ii) its 'Classification' column value is not equal to 'Close'. Finally,
    the subdesk_portfolio 'Portfolio' column values are extracted to node_list.
    Now that we have the list of relevant portfolios in node_list, the
    *SENSI*.TXT file(s), whose name matches the desired date(s) and trading
    entity, is read and only the rows whose 'Node' col values are in this
    node_list are filtered to pf_sensi.
    Another column 'Industry' is added via merge with issuer_mapping
    (read from a ReportsQuery-FA...csv file).
    The YYYYMMDD_IWORK_PL.TXT file, which provides the PnL Explained info is
    read, rows with 'PL Type' col matching either 'BTL' or 'PL'
    and 'Portfolio' col matching the trading entity (typically 'DBSSG')
    are filtered int the PL DataFrame.
    The YYYYMMDD_PLVA.DBSSG.TXT file is an alternative source of data for
    FX Options, which should be compared against the corresponding data in the
    *SENSI*.TXT file. On a "good" SENSI day the are in good agreement.
    -----------
    :param curdate: str        the business date as a string (e.g., '20190412')
    :param entity:  str        the trading entity (e.g., 'DBSSG', 'DBSHK', 'DBSSYD', etc)
    :param node_id: str        the trading desk (e.g., 'FX Options', 'CREDIT TRADING')
    :param pub_holiday: bool   the indicator if the date was preceded by a public holiday
    :param files: str          the path to the folder with the input files
    --------
    :return:
    pf_sensi: DataFrame        the relevant/filtered rows (and columns) from the *SENSI*TXT file
    plexp: DataFrame           the relevant/filtered rows (and columns) from the *IWORK_PL.TXT file
    pf_plva: DataFrame         the relevant/filtered rows (and columns) from the *PLVA.DBS.TXT file
    """
    sensi_dates = [parse(curdate)]
    pcwd = pathlib.Path(files)
    struct_file = list(pcwd.glob(str(sensi_dates[0].year) + '-*' +
                                 str(sensi_dates[0].month) + '*Structure.xls'))[0]
    structure = pd.read_excel(struct_file, sheet_name=0, header=None)
    # find the row where the column names are given (starts with 'Portfolio'):
    i_column_row = (structure.iloc[:, 0] == 'Portfolio').idxmax()
    structure.rename(columns=structure.loc[i_column_row], inplace=True)
    structure.drop(structure.head(i_column_row+1).index, inplace=True)
    nodeid_info = pd.read_csv(pcwd / 'Node IDs with thresholds.csv')
    node_level_id = nodeid_info.loc[nodeid_info['NODEID'] == node_id, 'NODE_LEVELID'].iloc[0]
    substructure = structure.query("(`{0}` == @node_id) & (Entity == @entity) &\
                                    (Classification != 'Close')".format(node_level_id))
    node_list = substructure['Portfolio'].unique().tolist()
    if (node_id == 'FX Options') and ('DBSSG FX OP EXE' in node_list):
        node_list.remove('DBSSG FX OP EXE')

    # if curdate follows a public holiday, prepend the date:
    if pub_holiday:
        sensi_dates.insert(0, sensi_dates[0] + relativedelta(days=-1))
    # if curdate is Monday, prepend the Saturday
    if sensi_dates[0].weekday() == 0:  # curdate is Monday, prepend the Saturday
        sensi_dates.insert(0, sensi_dates[0] + relativedelta(weekday=SA(-1)))
    # start with curdate and work backwards:
    ddate = sensi_dates.pop(-1)
    iwork_filepath = pcwd / (ddate.strftime('%Y%m%d') + '_SENSI.' + entity + '.TXT.gz')
    iwork_plvapath = pcwd / (ddate.strftime('%Y%m%d') + '_PLVA.' + entity + '.TXT.gz')
    sensi = pd.read_csv(iwork_filepath, sep=';', header=1, skiprows=-1, low_memory=False)
    sensi.drop(sensi.tail(1).index, inplace=True)
    plva = pd.read_csv(iwork_plvapath, sep=';', header=1, low_memory=False).head(-1)
    # looping back in dates, prepend their data:
    for isd in sensi_dates[::-1]:
        iwork_filepath = pcwd / (isd.strftime('%Y%m%d') + '_sensi.' + entity + '.TXT.gz')
        tmp = pd.read_csv(iwork_filepath, sep=';', header=1, skiprows=-1, low_memory=False)
        tmp.drop(tmp.tail(1).index, inplace=True)
        sensi = tmp.append(sensi, ignore_index=True)
        iwork_plvapath = pcwd / (isd.strftime('%Y%m%d') + '_PLVA.' + entity + '.TXT.gz')
        tmp = pd.read_csv(iwork_plvapath, sep=';', header=1, skiprows=-1, low_memory=False).head(-1)
        plva = tmp.append(plva, ignore_index=True)
    # sensi.drop_duplicates(keep='first')
    iwork_pl = pcwd / (ddate.strftime('%Y%m%d') + '_IWORK_PL' + '.TXT.gz')
    plexp = pd.read_csv(iwork_pl, sep=';', header=1, low_memory=False)
    tmp = plexp.query("`PL Type` in ('BTL', 'PL')")
    plexp = tmp.fillna({'Portfolio': 'XXXX'})
    tmp = plexp.query("Portfolio.str.contains(@entity)", engine='python')
    plexp = tmp.groupby('Portfolio')[['DTD LCY']].sum().reset_index()
    del tmp

    pf_plva = plva.query("Node in @node_list")
    pf_sensi = sensi.query("Node in @node_list")
    pf_sensi = pf_sensi.drop_duplicates(subset=[
        'Sensi Date', 'Node', 'Entity', 'Component', 'Product Type',
        'PL Currency', 'Val Group', 'Instrument', 'LCY', 'Raw Component',
        'Product Group', 'Issuer Curve', 'Issuer Name', 'Seniority',
        'Underlying', 'Yield Curve', 'Tenor', 'Underlying Maturity',
        'Quotation', 'Multiplier', 'Type', 'PL Explain', 'PL Sensi',
        'Strike', 'Mkt (T)', 'Mkt (T-1)', 'Sensi Risk (OCY)', 'DTD ORIG',
        'DTD LCY', 'DTD SGD'], keep='first')
    pf_sensi = pf_sensi.assign(Rate_move=pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"])
    pf_sensi = pf_sensi.drop(columns=['Business Date', 'Entity', 'Val Group',
                                      'Product Group', 'Issuer Curve',
                                      'Issuer Name', 'Seniority', 'Underlying',
                                      'Underlying Maturity',
                                      'Multiplier', 'Type', 'PL Explain',
                                      'PL Sensi', 'Strike'])
    if node_id == 'CREDIT TRADING':
        issuer_mapping = pcwd / ('ReportsQuery-FA Explain Credit Breakdown by Issuer_' +\
                         curdate[-2:] + ' ' + month_abbr[parse(curdate).month] + '.csv')
        df_issuer_mapping = pd.read_csv(issuer_mapping)
        df_issuer_mapping = df_issuer_mapping.head(-2)
        df3 = pd.merge(pf_sensi, df_issuer_mapping[['Instrument', 'Industry']],
                       left_on=['Instrument'], right_on=['Instrument'], how='left')
        pf_sensi = df3.drop_duplicates(keep='first')
    return pf_sensi, plexp, pf_plva


def get_dtd_pl(df_comment_raw, ddate, node_id):
    """
    Summary:
    --------
    Invoked only for 'CREDIT TRADING' and 'FX Options' node_id.
    Simply filters out the rows with matching DAY and NODEID
    and returns the 1st value (a float) of DTD_PL from the surviving row(s).

    Parameters:
    -----------
    df_comment_raw (DatFrame):  read from 'ReportsQuery-PL Commentary...csv'
    ddate (str):                date of ineterest (e.g. '20190412')
    node_id (str):              trading desk (e.g., 'CREDIT TRADING')

    Returns:
    --------
    dtd_pl (number):            the value from 'DTD_PL' col from the first row
                                whose 'DAY' & 'NODEID' col values
                                match ddate & node_id, respectively
    """
    df_comment_raw['DAY'] = pd.to_datetime(df_comment_raw['DAY']).dt.date
    fdate = pd.to_datetime(ddate).date()
    df_comment_raw_clean = df_comment_raw.query("(DAY == @fdate) & (NODEID == @node_id)")
    idxcol = df_comment_raw_clean.columns.get_loc('DTD_PL')
    dtd_pl = df_comment_raw_clean.iloc[0, idxcol]
    return dtd_pl


def pivtab_best(pfx_sensi, the_idx='Node', the_col='Raw Component', n_best=1, dtd_pl=None):
    """
    Summary:
    --------
    Invoked for 'CREDIT TRADING' and 'FX Options' node_id.
    Calculates the pivot table of a filtered form of *SENSI*.TXT file (i.e, the
    pf_sensi dataframe and its filtered descendants, pfc_sensi, pfccy_sensi etc)
                     Bench Roll      ...  Time Effect  Yield curves (zc)     Total
    Portfolio                        ...
    DBSSG CRD AFSH            0      ...         0.00       0.000000  6.464594e+05
    DBSSG CRD SGD             0      ...         0.00       0.000000  2.454460e+05
    DBSSG CRD INTG            0      ...         0.00       0.000000  2.342747e+05
    DBSSG CRD DESK            0      ...         0.00       0.000000  1.646836e+05
    DBSSG CRD TRDAS           0      ...         0.00       0.000000  1.572851e+05
    ....

    Parameters:
    -----------
    pfx_sensi (DatFrame):     the (pre-filtered) portfolio dataframe
    the_idx (str):            use its values for rows to break down the DTD SGD/LCY
    the_col (str):            use its values for columns to break down the
                              DTD SGD/LCY amount and select the value with the top
                              n_best contribution(s) for each portfolio
    n_best (int):             how many top contributions for each portfolio
    dtd_pl (int):             the Day-to-Day Profit/Loss extracted from
                              ReportsQuery-PL Commentary...csv

    Returns:
    --------
    pfx_comp (DataFrame):    for each row (i.e., portfolio/node) the n_best
                             Raw Component (or PL Currency/Yield Curve/Tenor)
                             with the most prominent contributions to DGD SGD/LCY
    """
    if the_col == 'Raw Component':
        n_best = 1  # enforcing 1 here because 1 is the only sensible choice in this case

    pivot_sensi = pfx_sensi.pivot_table(index=the_idx,
                                        columns=the_col,
                                        values='DTD LCY',
                                        aggfunc=np.sum,
                                        fill_value=0,
                                        margins=True,
                                        margins_name='Total').head(-1)
    pivot_sensi.index.name = the_idx
    pivot_sensi.columns.name = None
    # sorting dependent om P/L sign:
    if dtd_pl < 0:
        pivot_sensi = pivot_sensi.query("Total < 0").sort_values('Total', ascending=True)
        comp_idx = np.argsort(pivot_sensi.values[:, :-1], axis=1)[:, :n_best]
    else:
        pivot_sensi = pivot_sensi.query("Total > 0").sort_values('Total', ascending=False)
        # flip needed because argsort-ing from low to high:
        comp_idx = np.fliplr(np.argsort(pivot_sensi.values[:, :-1], axis=1)[:, -n_best:])

    pfx_comp = pd.DataFrame(columns=[the_idx, the_col])
    pfx_comp[the_idx] = pivot_sensi.index

    if the_col == 'Raw Component':
        for i in range(pivot_sensi.shape[0]):
            if pivot_sensi.columns[comp_idx[i, 0]] == 'Credit Delta':
                comp_idx[i, 0] = pivot_sensi.columns.get_loc('Credit Delta Opening')
            elif pivot_sensi.columns[comp_idx[i, 0]] == 'IR Delta':
                comp_idx[i, 0] = pivot_sensi.columns.get_loc('IR Delta Opening')
            elif pivot_sensi.columns[comp_idx[i, 0]] == 'IR Delta Derived':
                comp_idx[i, 0] = pivot_sensi.columns.get_loc('IR Delta Opening')
            elif pivot_sensi.columns[comp_idx[i, 0]] == 'IR Basis Delta':
                comp_idx[i, 0] = pivot_sensi.columns.get_loc('IR Basis Delta Opening')
    pfx_comp[the_col] = pivot_sensi.columns[comp_idx[:, 0]]
    for j in range(1, n_best):
        new_col = the_col + '_' + str(j+1)
        pfx_comp = pfx_comp.assign(**{new_col: pivot_sensi.columns[comp_idx[:, j]]})
    return pfx_comp


def get_top_contribution(pivtab, p_sensi, first_col='Node', sec_col='Raw Component'):
    """
    Summary:
    -----------
    Simply merges the pivot table with the pre-filtered SENSI on 'Node' and
    the_col columns, effectively generating the updated p_sensi, filtering
    only the top contribution from the_col values, for the next merge step of
    pivot-merge-filter (Raw_Component->PL Currency->Yield Curve->Tenor)

    Parameters:
    -----------
    pivtab (DataFrame):   the result of the pivtab_best() function above
    p_sensi (DataFrame):  the pre-distilled SENSI (pf, pfc, pfcc, pfccy _sensi)
    the_col (string):     the col on which to merge (in addition to 'Node')

    Returns:
    -----------
    ... (DataFrame):      the product of the merge of the two input DataFrames
    """
    # Match (a single or two word column name)_digit(1-9):
    pattern = '^([A-Za-z]+\s*[A-Za-z]+)_([1-9]+)'
    matched = re.match(pattern, sec_col)
    if matched:
        pf_col = matched.groups()[0]
    else:
        pf_col = sec_col
    return pd.merge(pivtab.reset_index(), p_sensi, left_on=[first_col, sec_col],
                    right_on=[first_col, pf_col], how='inner')


def diag_sensi_vs_pl(pf_sensi, plexp, dtd_pl, n_rows=8):
    """
    Summary:
    -----------
    Invoked only for 'CREDIT TRADING' and 'FX Options' node_id.
    Prints a diagonal (if SENSI ~= PL) matrix of DTD SGD contributions for
    each portfolio (rows) from SENSI vs DTD LCY contribution from PL.

    Parameters:
    -----------
    pf_sensi (DataFrame):    the table from the pre-distilled SENSI
    plexp (DataFrame)        the table from the pre-distilled IWORK_PL.TXT
    dtd_pl (number):         the DTD_PL obtained in get_dtd_pl() from the
                             ReportsQueryPL...csv file. Only the SIGN matters.
    nrows (int):             how many rows of the matrix to print

    Returns:
    -----------
    sensi_dtd_pl (number):   the DTD_PL amount calculated from the SENSI table
    goodday_sensi (bool):    flag if SENSI approach explains P/L well on the day
    """
    sensi_dtd_pl = pf_sensi['DTD LCY'].sum()
    # For FX Options compare the following two:
    print(f"PL total vs SENSI: {dtd_pl:.2f}, {sensi_dtd_pl:.2f}\n")
    df2 = pd.merge(pf_sensi, plexp, left_on=['Node'], right_on=['Portfolio'], how='left')
    df2.drop(columns='Portfolio', inplace=True)
    df2.rename({'DTD LCY_x': 'DTD LCY', 'DTD LCY_y': 'DTD LCY_exp'}, axis=1, inplace=True)
    # keep only the same sign(DTD LCY_exp) == sign(dtd_pl) rows, i.e. Portfolios/Nodes
    pf_sensi = df2.query("`DTD LCY_exp` * @dtd_pl > 0")
    # For CREDIT TRADING compare the following two:
    print(f"vs SENSI with the right sign: {dtd_pl:.2f}, {pf_sensi['DTD LCY'].sum():.2f}\n")
    pfpf = pf_sensi.pivot_table(index='Node',
                                columns='DTD LCY_exp',
                                values='DTD LCY',
                                aggfunc=np.sum,
                                fill_value=0,
                                margins_name='Total',
                                margins=True)
    # put it in the diagonal-matrix form:
    pfpf.sort_values('Total', ascending=(dtd_pl < 0), inplace=True)
    if dtd_pl > 0:
        new_cols = np.fliplr(pfpf.columns.values.reshape(1, -1))[0]
    else:
        new_cols = np.roll(pfpf.columns.values.reshape(1, -1), 1)[0]
    print('=' * 80)
    print(pfpf[new_cols].drop('Total', axis=0).iloc[:n_rows, :n_rows+1])
    print('-' * 80)
    print("\n")
    if abs((dtd_pl - sensi_dtd_pl) / dtd_pl) > 0.25:
        goodday_sensi = False
    else:
        goodday_sensi = True
    return pf_sensi, sensi_dtd_pl, goodday_sensi


def old_format(num):
    """
    Summary:
    -----------
    Invoked only for 'CREDIT TRADING' and 'FX Options' node_id.
    Simply evaluates the order of magnitude of paramater num
    (i.e., Kilo, Mega, Giga,...) for printing in a nicer format

    Parameters:
    -----------
    num (float):       a number

    Returns:
    -----------
    final_num (str):   reformated num as a string
    """
    magnitude = 0
    if (abs(num) >= 0) & (abs(num) < 1000000):
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        final_num = f"{num:.0f}{['', 'K', 'M', 'G', 'T', 'P'][magnitude]}"
    else:
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        final_num = f"{num:.2f}{['', 'K', 'M', 'G', 'T', 'P'][magnitude]}"
    return final_num


def helper1(num, fnum, magnitude, i):
    """
    Summary:
    -----------
    Simply reducing the "cognitive complexity" by 6 points of human_format() below.

    Parameters:
    -----------
    num (float):       a number
    fnum (str):        reformatted num as a string
    magnitude (int):   order of magntude (1, 2, etc..)
    i (int):           if supplied, print this many digits after the decimal point

    Returns:
    -----------
    fnum (str):   reformatted num as a string
    """
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    fnum += f"{num:.{i}f}" if i else f"{num:.0f}"
    fnum += f"{['', 'K', 'M', 'G', 'T', 'P'][magnitude]}"
    return fnum


def human_format(num, i=None, round2k=True):
    """
    Summary:
    -----------
    Invoked only for 'CREDIT TRADING' and 'FX Options' node_id.
    Simply evaluates the order of magnitude of paramater num
    (i.e., Kilo, Mega, Giga,...) for printing in a nicer format

    Parameters:
    -----------
    num (float):       a number
    i (int):           if supplied, print this many digits after the decimal point
    round2k (bool):    a flag whether to round smaller than 1k or not

    Returns:
    -----------
    final_num (str):   reformatted num as a string
    """

    if num > 0:
        final_num = '+'
    elif num < 0:
        final_num = '-'

    num = abs(num)
    if num < 1e5:
        if num < 1e3 and round2k:
            num /= 1000.0
            final_num += f"{num:.{i}f}K" if i else f"{num:.1f}K"
        else:
            magnitude = 0
            final_num = helper1(num, final_num, magnitude, i)
    else:
        num /= 1e6
        final_num += f"{num:.{i}f}M" if i else f"{num:.1f}M"
    return final_num


def sign_it(num, ccy='SGD'):
    """
    Summary:
    -----------
    Simply produces "Gain/Loss" for +/- sign in the humanum string.

    Parameters:
    -----------
    num (float):       a number, $ PnL
    currency (str):    e.g., "S$" for SGD

    Returns:
    -----------
    fnum (str):     reformatted num as a string
    """

    humanum = human_format(num)
    if humanum[0] == '+':
        fnum = 'PL Gain of ' + ccy + ' ' + humanum[1:]
    elif humanum[0] == '-':
        fnum = 'PL Loss of ' + ccy + ' ' + humanum[1:]
    return fnum


def get_idxminmax0(pf, the_col, xrac='frac', min_xrac=0.9, max_xrac=1.1):
    """
    Summary:
    -----------
    To select Tenors, Strikes etc in the desired range of fractions
    (for tenors 90-110%) to their total $ amount.

    Parameters:
    -----------
    pf (DataFrame):    the pre-distilled SENSI, for FX Vega tenors usually
                       filtered to contain a single Quotation, e.g. CNH-USD
    the_col (str):     the column from the DataFrame where the selection is
                       taking place (usually 'Tenor')
    xrac (str):        either 'frac' if one is interested in the partial
                       contributions to the total $ amount, or 'vrac' if one
                       is interested in the partial variance contributions
    min_xrac (float):  the lower bound of the desired range to select
    max_xrac (float):  the upper bound of the desired range to select

    Returns:
    -----------
    pf.iloc[:, ] (Series): the selected row range (iMin, iMax], column the_col
    """
    ixrac = pf.columns.get_loc(xrac)
    icol = pf.columns.get_loc(the_col)
    n = pf.shape[0]
    pf[xrac] = pf[xrac].astype('float')
    imax = pf[xrac].idxmax()
    for im in range(0, n):   # shift away from iMax
        for jw in range(im, n):   # window size around iMax
            sumxrac = pf.iloc[imax - im: imax - im + jw + 1, ixrac].sum()
            if min_xrac < sumxrac < max_xrac:
                # The desired range identified! Break the 1st loop...
                break
        if min_xrac < sumxrac < max_xrac:
            # ... and the 2nd loop:
            break

    imin = imax - im
    imax = imax - im + jw
    return pf.iloc[imin: imax+1, icol]


def get_idxminmax(pf_, the_col, xrac='frac', min_xrac=0.9, max_xrac=1.1):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' Node_id.
    Currently called only from get_Vega_tenors_rates() to select tenors in the
    desired range of fractions (for tenors 90-110%) to their total $ amount.

    Parameters:
    -----------
    pf_ (DataFrame):   a pre-distilled SENSI table, for FX Vega tenors usually
                       filtered so as to contain a single Quotation, e.g. CNH-USD
    the_col (str):     the column from the pf_ DataFrame where the selection is
                       taking place (usually 'Tenor')
    xrac (str):        either 'frac' if one is interested in the partial
                       contributions to the total $ amount, or 'vrac' if one
                       is interested in the partial variance contributions
    min_xrac (float):  the lower bound of the desired fraction/percentage range to select
    max_xrac (float):  the upper bound of the desired fraction/percentage range to select
    vrac_min (float):  fraction/percentage threshold below which to truncate the partial variances

    Returns:
    -----------
    pf_.iloc[:, ] (Series): the selected row range [i_min, i_max) from column the_col
    """
    i_xrac = pf_.columns.get_loc(xrac)
    i_col = pf_.columns.get_loc(the_col)
    pf_[xrac] = pf_[xrac].astype('float')
    i_max = pf_[xrac].idxmax()

    sumxrac = pf_.iloc[i_max, i_xrac]
    lower = i_max
    upper = i_max
    i_up = 1
    i_down = 1
    while True:
        if min_xrac < sumxrac < max_xrac:
            break
        if i_max + i_up < pf_.shape[0]:
            tmp_up = pf_.iloc[i_max + i_up, i_xrac]
        else:
            tmp_up = -np.infty
        if i_max - i_down >= 0:
            tmp_down = pf_.iloc[i_max - i_down, i_xrac]
        else:
            tmp_down = -np.infty
        if abs(sumxrac + tmp_up - 1) < abs(sumxrac + tmp_down - 1):
            sumxrac += tmp_up
            upper = i_max + i_up
            i_up += 1
            continue
        if abs(sumxrac + tmp_up - 1) > abs(sumxrac + tmp_down - 1):
            sumxrac += tmp_down
            lower = i_max - i_down
            i_down += 1
            continue

    return pf_.iloc[lower: upper+1, i_col]


def get_idxminmax2(pf_, the_col, xrac='frac', min_xrac=0.9, max_xrac=1.1, xrac_min=None):
    """
    Summary:
    -----------
    Invoked only for 'FX Options'  and IRD Node_id.
    A variation of get_idxminmax() from FXO, which always make selection so
    as to stay closer to a total of 100%. This function, in contrast, aims for to get
    closer to the desired range (min_xrac, max_xrac) of fractions (for tenors 90-110%).

    Parameters:
    -----------
    pf_ (DataFrame):   a pre-distilled SENSI table, for FX Vega tenors usually
                       filtered so as to contain a single Quotation, e.g. CNH-USD
    the_col (str):     the column from the pf_ DataFrame where the selection is
                       taking place (usually 'Tenor')
    xrac (str):        either 'frac' if one is interested in the partial
                       contributions to the total $ amount, or 'vrac' if one
                       is interested in the partial variance contributions
    min_xrac (float):  the lower bound of the desired fraction/percentage range to select
    max_xrac (float):  the upper bound of the desired fraction/percentage range to select
    xrac_min (float):  fraction/percentage threshold below which to discard the rows

    Returns:
    -----------
    pf_.iloc[:, ] (Series): the selected row range [i_min, i_max) from column the_col
    """
    if xrac_min:
        # Series with True or False values:
        tmp_ser = (pf_[xrac].abs() < xrac_min)
        # drop indices where True, i.e. tmp_ser.index[tmp_ser]
        pf_.drop(tmp_ser.index[tmp_ser].tolist(), axis=0, inplace=True)
        # re-index the DataFrame after dropping rows:
        pf_.reset_index(inplace=True, drop=True)
    i_xrac = pf_.columns.get_loc(xrac)
    i_col = pf_.columns.get_loc(the_col)
    i_max = pf_[xrac].idxmax()
    sumxrac = pf_.iloc[i_max, i_xrac]
    lower = i_max
    upper = i_max
    i_up = 1
    i_down = 1
    while True:
        if min_xrac < sumxrac < max_xrac:
            break
        balance = 1.0 - sumxrac
        if i_max + i_up < pf_.shape[0]:
            tmp_up = pf_.iloc[i_max + i_up, i_xrac]
        else:
            tmp_up = -np.infty
        if i_max - i_down >= 0:
            tmp_down = pf_.iloc[i_max - i_down, i_xrac]
        else:
            tmp_down = -np.infty
        move = sorted([t for t in [tmp_up, tmp_down] if t > -np.infty], reverse=(balance > 0))[0]
        if  move == tmp_up:
            sumxrac += tmp_up
            upper = i_max + i_up
            i_up += 1
            continue
        if  move == tmp_down:
            sumxrac += tmp_down
            lower = i_max - i_down
            i_down += 1
            continue
    return pf_.iloc[lower: upper+1, i_col]


def get_selection(pf_, tot_amount, idx, dollar='DTD LCY', th=None, top3=True):
    """
    Summary:
    -----------
    Invoked only for 'FX Options' and 'IRD' Node_id. From the pre-filtered pf_ table
    it selects up to 3 top rows with most prominant $ contributions.
    The process comprises these steps:
    0) Assign the fractional $ contribution 'frac' and partial variance 'vrac'
    1) Sort the rows on abs(frac) (not discriminating on the sign of frac)
    2) Drop the rows if BOTH their frac < 5% AND vrac < 5% (th[4])
    3) If all |frac| < 0.25 (th1), select no rows
    4) If (all frac > 0) & (any frac > 0.8 (th0)), select this row;
       else select top rows in 0.8 (th3) < Sum_i frac_i < 1.2 (th4)
    5) If you ended up with more than 3 rows, take simply the top 3 rows.
       Else, if you ended up with zero rows, take simply the top 3 rows
       (if top3 == True).

    Parameters:
    -----------
    pf_ (DataFrame):    a pre-filtered SENSI table from which to select the rows
    tot_amount (float): the Total $ amount (the sum of the, Dollar column)
    idx (str):          the column from the table from which to select the items
    dollar (str):       the column from the table with the $ values
    th (list):          5 thresholds (floats): [0] if all frac > 0, pick the
                        one higher than this (0.5); [1] if all frac smaller than
                        this, don't select any; [2] & [3] the 80-120% range;
                        [4] the mininal frac & vrac to consider (usually 5%)
    top3 (bool):        the flag whether to enfore the selection of top 3

    Returns:
    -----------
    top_bit (list):     the series-turned-tolist() with the selected items
    """
    pf_ = pf_.assign(frac=round(pf_[dollar].div(pf_[dollar].sum()), 3))
    pf_ = pf_.assign(absfrac=pf_['frac'].abs())
    pf_.sort_values('absfrac', inplace=True, ignore_index=True, ascending=False)
    pf_ = pf_.assign(cumfrac=round(pf_['frac'].cumsum(), 3))
    n_size = pf_.shape[0]
    mu = tot_amount / n_size
    pf_ = pf_.assign(vrac=round((pf_[dollar] - mu)**2 / (n_size * pf_[dollar].var(ddof=0)), 4))
    pf_ = pf_.query("(vrac > @th[4]) | (absfrac > @th[4])")
    pf_.reset_index(inplace=True, drop=True)
    del pf_['absfrac']
    i_idx = pf_.columns.get_loc(idx)
    if ((th[2] < pf_['cumfrac'].iloc[:6]) & (pf_['cumfrac'].iloc[:6] < th[3])).any():
        # looking for a multiplet of items in the 80-120% range
        i_top = pf_.query("@th[2] < cumfrac < @th[3]").index[0]
        top_bit = pf_.iloc[:i_top+1, i_idx]
    elif ((th[2] < pf_['frac'].iloc[:3]) & (pf_['frac'].iloc[:3] < th[3])).any():
        # looking for an item in the DTD 80-120% range, but restricted to within the top 3 only
        # the reason for this is to prevent selection of such an item from the middle of the list
        i_top = pf_.query("@th[2] < frac < @th[3]").index[0]
        top_bit = pf_.iloc[i_top: i_top+1, i_idx]
    else:
        top_bit = pd.Series(dtype='category')

    if (top_bit.shape[0] > 3) or (top_bit.empty and top3):
        # If the item list is still too long, try to truncate it to 3 items.
        # First, if the top item is above the range, allow negative fracs:
        if pf_['frac'].iloc[0] > th[3]:
            # if the 1st item already above the range, allow negative fracs:
            top_bit = pf_.iloc[:3, i_idx]
        # Second, if the top item(s) has a negative frac:
        elif pf_['frac'].iloc[0] < 0:
            # count the negative items in the top3:
            cnt_neg = pf_.head(3).query("frac < 0").shape[0]
            # abs(rolling frac sum, window=3, - 1):
            pf_ = pf_.assign(absum3_1=abs(pf_['frac'].rolling(3).sum() - 1))
            # find its min, but limited to top (3 + cnt_neg) only:
            i_top = pf_['absum3_1'].iloc[:3 + cnt_neg].idxmin()
            top_bit = pf_.iloc[i_top-2: i_top+1, i_idx]
            pf_.drop('absum3_1', axis=1, inplace=True)
        # Finally, if all failed, drop the negative fracs:
        else:
            pf_ = pf_.query("frac > 0")
            top_bit = pf_.iloc[:3, i_idx]
    return top_bit.tolist()


def get_selection2(pf_, tot_amount, idx, dollar='DTD LCY', th=None, top3=True):
    """
    Summary:
    -----------
    Similar to the get_selection() above, but a lot simpler.
    It stacks the negative fraction/percentage items (descending order)
    on top of the positive ones (also in descending order).
    Then it uses get_idxminmax2(), usually employed to find tenor range.

    Parameters:
    -----------
    pf_ (DataFrame):    a pre-filtered SENSI table from which to select the rows
    tot_amount (float): the Total $ amount (the sum of the, Dollar column)
    idx (str):          the column from the table from which to select the items
    dollar (str):       the column from the table with the $ values
    th (list):          5 thresholds (floats): [0] if all frac > 0, pick the
                        one higher than this (0.5); [1] if all frac smaller than
                        this, don't select any; [2] & [3] the 80-120% range;
                        [4] the mininal frac & vrac to consider (usually 5%)
    top3 (bool):        the flag whether to enfore the selection of top 3

    Returns:
    -----------
    top_bit (list):     the series-turned-tolist() with the selected items
    """
    pf_ = pf_.assign(frac=round(pf_[dollar].div(pf_[dollar].sum()), 3))
    pf_.sort_values('frac', inplace=True, ignore_index=True, ascending=False)
    # get the number of rows with a negative fraction/percentage:
    i_roll = pf_.query("frac < 0").shape[0]
    pf_rolled = pd.DataFrame(columns=pf_.columns, data=np.roll(pf_.values, i_roll, axis=0))
    pf_rolled['frac'] = pd.to_numeric(pf_rolled['frac'])

    select = get_idxminmax2(pf_rolled, idx, xrac='frac', min_xrac=th[2], max_xrac=th[3])
    # if more than 3 got selected, sort them by the absolute fraction/percentage and pick top 3:
    if top3 and (len(select) > 3):
        tmp = select.tolist()
        pf_ = pf_.assign(absfrac=pf_['frac'].abs())
        my3 = pf_[pf_[idx].isin(tmp)].sort_values('absfrac', ignore_index=True, ascending=False)
        return my3[idx].tolist()[:3]
    # otherwise keep all of the selected:
    else:
        return select.tolist()


def lst_unpack(str_start='', d_list=None, end='.'):
    """
    Summary:
    --------
    A simple utility to unpack d_list into the combined output string in an Excel cell

    Parameters:
    -----------
    str_start (str):   the wording before d_list
    d_list (list):     the list to unpack
    end (str):         the end character

    Returns:
    -----------
    final_str (str):    the combined output
    """
    final_str = (str_start + " {}," * (len(d_list) - 1) + " {}" + end).format(*d_list)
    return final_str


def highlight(c='yellow', lst=None):
    """
    Summary:
    --------
    A simple utility to use for highlighting fields of a DataFrame via applymap().
    This was initially done using lambda:
    highlight_y = lambda x: 'background-color: yellow' if x in lst else ''
    but the if-else adds to Cognitive Complexity in Sonar, especially when
    nested inside a block or a loop, so it's better to take it out.
    This is done via a helper function do_it(x) which will receive Series
    elements via applymap()

    Parameters:
    -----------
    c (str):           the string describing the color (yellow, orange, green)
    lst (list):        contains the elements which we wish to highlight by c

    Returns:
    -----------
     (str):           the color or an empty string for each table element x
    """
    def do_it(x):
        # The issue: how to use applymap with multiple arguments?
        # One could define a Class and pass the extra arguments to the Class constructor. Easier:
        # stackoverflow.com/questions/52794791/call-multiple-argument-function-in-applymap-in-python
        return f'background-color: {c}'  if x in lst else ''
    return do_it


def highlight_ser(x, c='yellow', lst=None):
    """
    Summary:
    --------
    A simple utility to use for highlighting either row/column of a DataFrame via apply().
    This was initially done using lambda:
    highlight_ = lambda x: ['background-color: yellow'] * x.shape[0] if x.name in lst\
                            else [''] * x.shape[0]
    but the if-else adds to Cognitive Complexity in Sonar, especially when
    nested inside a block or a loop, so it's better to take it out.

    Parameters:
    -----------
    x (Series)         a Series (row/column) of a DataFrame
    c (str):           the string describing the color (yellow, orange, green)
    lst (list):        contains the row/column names which we wish to highlight by c

    Returns:
    -----------
    [(str)]:           the color (or an empty string) broadcast into a list of the same len as x
    """

    return [f'background-color: {c}'] * x.shape[0] if x.name in lst else [''] * x.shape[0]
