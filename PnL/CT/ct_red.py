# coding: utf-8

####################################################
# Author: Ante Bilic                               #
# Since: Mar 27, 2020                              #
# Copyright: The PLC Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

"""Summary:
   -------
   Collection of functions for CREDIT TRADING P/L attribution using SENSI method.
"""

import re
from calendar import month_abbr
from itertools import islice
import numpy as np
import pandas as pd
from pandas_ods_reader import read_ods
from dateutil.parser import parse
from common_futils import get_clean_portfolio, get_dtd_pl, pivtab_best,\
    get_top_contribution, diag_sensi_vs_pl, old_format

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class PassDaFrame:
    """
    Needed simply to unify the 3 lambdas in trim_irdeltas() into a single function,
    since they take a single argument x, but their DataFrames differ and
    need to be passed effectively as another argument.
    """
    def __init__(self, frame):
        """
        :param frame: DataFrame    - a SENSI table (pfc_sensi, pfccy_sensi, or pfccy_2_sensi)
        """
        self.frame = frame

    def get_dtd(self):
        """
        Returns the the 'DTD' column for the given SENSI table
        :return: float   - the DTD column
        """
        return self.frame['DTD SGD']

    def tab_pl_contrib(self, x_node):
        """
        Evaluates the Total of the 'DTD' column for the given SENSI table, pre-filtered
        for the node/portfolio from irdeltas.
        :param x_node: str    - a portfolio from the 'Node' column
        :return: float        - the sum of the DTD column
        """
        return self.frame.query("Node == @x_node")['DTD SGD'].sum()


def get_tenors_rates(pf_any, dtd_pl):
    """
    Summary:
    Invoked only for 'CREDIT TRADING' Node_id.
    Handles 3 separate forms of Tenors: (i) 'TOT', (ii) date in the DD/MM/YY
    format, and (iii) from the tenor_list below. The latter case makes the use
    of the pf DataFrame to calculate a pivot table, split by the
    values of colnames, i.e., 'Tenor', 'Mkt (T)', and 'Mkt (T-1)', with the sum
    of 'DTD SGD' values for each triplet, named "df_tenor".
    The df_tenor rows are then sorted based on the duration of 'Tenor' values.
    A col 'Rate_move' is added as the difference of 'Mkt (T)' and 'Mkt (T-1)'.
    Also, a col 'pc' is appended as the ratio of 'DTD SGD' values and their sum.
    The rows with pc values above 0.01 are filtered to "df_tenor2", whose rows
    with min & max 'Tenor' durations are recorded, so that a list sel_tenor_lst
    can be filled with the 'Tenor' values between these two (the two included).
    The "df_tenor" (NOT "df_tenor2") rows whose 'Tenor' values are in this list are
    filtered to df_tenor_select and their 'RatesMovement' values are used to
    evaluate the avg_ratemove.

    Parameters:
    pf_any (DataFrame):    either pfccy_sensi or pfccy_2_sensi pre-filtered for
                           a specific portfolio and one of its top yield curves
    dtd_pl (number):       the value from 'DTD_PL' col from the matching row
                           of ReportsQueryPL...csv. Only the SIGN matters.

    Returns:
    tenor_max (str):       the longest tenor contributing to this Node/Component/Yield Curve triplet
    tenor_min (str):       the shortes tenor contributing to this Node/Component/Yield Curve triplet
    avg_ratemove (float):  the average of the two
    the_amount (float):    the $ amount at risk
    """
    cnt_date = pf_any['Sensi Date'].nunique()
    tenor_list = ['O/N', 'T/N', '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M',
                  '10M', '11M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '11Y',
                  '12Y', '13Y', '14Y', '15Y', '16Y', '17Y', '18Y', '19Y', '20Y', '30Y', '40Y', '50Y'
                  ]
    # get_tenor_idx = lambda x: tenor_list.index(x) + 1

    def get_tenor_idx(tenor):
        """
        Evaluates the ordinal number of the tenor in the tenor_list
        :param tenor: str
        :return:      int
        """
        try:
            return tenor_list.index(tenor) + 1
        except ValueError:
            return None

    pf_any['Tenor'].replace({'12M': '1Y'}, inplace=True)
    pf_any['Tenor'].replace({str(i) + 'WK': str(i) + 'W' for i in range(1, 10)}, inplace=True)
    pattern = '^([0-2][0-9]|3[0-1])(\/)(0[1-9]|1[0-2])(\/)(\d{2})$'   # DD/MM/YY
    tenor_min = None
    tenor_max = None
    avg_ratemove = None
    the_amount = None
    if pf_any['Tenor'].str.match('TOT').all():
        avg_ratemove = abs(round(pf_any['Rate_move'].mean() * 100, 2))
        the_amount = round(pf_any['Sensi Risk (OCY)'].mean(), 0)
    elif pf_any['Tenor'].str.match(pattern).all():
        if dtd_pl > 0:
            dmax = pf_any['DTD SGD'].idxmax()
        else:
            dmax = pf_any['DTD SGD'].idxmin()
        tenor_max = pf_any.loc[dmax, 'Tenor']
        avg_ratemove = abs(round(pf_any.loc[dmax, 'Rate_move'] * 100, 2))
        the_amount = round(pf_any['Sensi Risk (OCY)'].sum(), 0)
    elif pf_any['Tenor'].str.match(pattern).any() and pf_any['Tenor'].isin(tenor_list).any():
        print("Maturity date(s) mixed up with Tenor(s)!")
    elif pf_any['Tenor'].isin(tenor_list).all():
        df_tenor = pf_any.pivot_table(index=['Tenor', 'Mkt (T)', 'Mkt (T-1)', 'Rate_move'],
                                      columns='Product Type',
                                      values=['Sensi Risk (OCY)', 'DTD SGD'],
                                      aggfunc=np.sum,
                                      margins=True,
                                      margins_name='-Total')
        df_tenor = df_tenor.reset_index()
        df_tenor.index.name = None
        df_tenor.columns.name = None
        df_tenor.columns = [x[0] + x[1] for x in df_tenor.columns]
        the_total_dtd = df_tenor['DTD SGD-Total'].values[-1]
        df_tenor = df_tenor.head(-1)
        df_tenor['ord'] = df_tenor['Tenor'].map(get_tenor_idx).\
            where(cond=df_tenor['Tenor'].notna(), other=0)
        df_tenor = df_tenor.sort_values('ord', ascending=True).reset_index()
        df_tenor['pc'] = df_tenor['DTD SGD-Total'].div(the_total_dtd)
        df_tenor.drop(columns='index', inplace=True)

        df_tenor2 = df_tenor.query('abs(pc) >= 0.05').reset_index()
        imax = df_tenor2['ord'].idxmax()
        imin = df_tenor2['ord'].idxmin()
        itenor = df_tenor2.columns.get_loc('Tenor')
        tenor_max = df_tenor2.iloc[imax, itenor]
        tenor_min = df_tenor2.iloc[imin, itenor]

        start_idx = 0
        end_idx = -1
        for idx, value in enumerate(tenor_list):
            # if only a single tenor found, it is both min & max:
            if (value == tenor_min) & (value == tenor_max):
                start_idx = idx
                end_idx = idx
            elif value == tenor_min:
                start_idx = idx
            elif value == tenor_max:
                end_idx = idx

        sel_tenor_lst = []
        for i, tenor in enumerate(tenor_list):
            if start_idx <= i <= end_idx:
                sel_tenor_lst.append(tenor)

        df_tenor_select = df_tenor.query('Tenor in @sel_tenor_lst')
        sum_ratemove = df_tenor_select['Rate_move'].sum()
        the_amount = round(df_tenor_select['Sensi Risk (OCY)-Total'].sum() / cnt_date, 0)
        cnt_tenor = df_tenor_select['Tenor'].nunique()
        avg_ratemove = round(abs(sum_ratemove / cnt_tenor * 100) / cnt_date, 2)
        # return df_tenor, tenor_max, tenor_min, avg_ratemove, old_format(the_amount)
    else:   # inspect what's going on
        input(pf_any.head())
        print("A trouble here?\n", pf_any['Tenor'])
    return tenor_max, tenor_min, avg_ratemove, the_amount


def go_irdelta_comments(irdframe, dtd_pl):
    """
    Summary:
    For Credit Trading only,
    Adds a new column 'comment' to the irdframe for IR Delta contributions,
    prefills it with a string, and then applies a rule-based workflow to
    evaluate the remaining part(s) of the string.
    Finally, it prints the DataFrame for irdelta portfolios contributing to the P&L.

    Parameters:
    irdframe (DataFrame):  A table that summarises the evaluated properties
                           of all the contributing IR Delta portfolios.
    dtd_pl (number):       the value from 'DTD_PL' col from the matching row
                           of ReportsQueryPL...csv. Only the SIGN matters.
    """
    # Pre-fill the 1st part of the commentary (c.f. "groupby without aggregation"):
    comm_groups = irdframe.groupby(['DTD_SGD', 'Node', 'Raw_Component', 'Ccy'], as_index=False)

    def start_comment(group):
        """
        Evaluates either the beginning of the commentary, if that is the 1st appearance
        of the node/portfolio with the biggest $ amount at risk, or "and" for the smaller
        $ amount at risk associated with the same node/portfolio and its DTD P/L.
        :param group: DataFrame  the rows of DataFrame associated with a group from DataFrameGroupBy
        :return: DataFrame       the rows of DataFrame updated with the new 'comment' column
        """
        group['comment'] = np.where(group.index == abs(group['AtRisk']).idxmax(),
                                    "SGD{} from {} {} where bulk of PL is from ", " and ")
        return group

    irdframe = comm_groups.apply(start_comment)
    # And now for the rest of the commentary:
    template = [
        "{} as yields {} by an average of {}bps for tenors {} to {} on net {} IRPV01 of {} {}.\n",
        "from benchmark {} Gov bond as yields {} by {}bps on net {} IRPV01 of {} {}.\n",
        "{} as yields {} by {}bps for {} contracts.\n",
        "benchmark {} Gov bond yield {} by an average of {}bps.\n"]

    for irow in irdframe.iterrows():
        idx = irow[0]
        row = irow[1]
        missing_vals = row[row.isna()].index.tolist()
        # if row['DTD_SGD'] > 0:
        if dtd_pl > 0:
            dec_inc = 'increased'
        else:
            dec_inc = 'decreased'
        if not missing_vals:
            # No missing values in this row. Also, its comment string is the 1st part (no 'and') of
            # the commentary. Insert the DTD, Node and Component values in the pre-filled string.
            # Use the template[0] for the rest:
            if row['comment'] != ' and ':
                irdframe.iloc[idx, -1] = (row['comment'] + template[0]).\
                    format(old_format(row['DTD_SGD']),
                           row['Node'],
                           ' '.join(row['Raw_Component'].split()[:2]),
                           row['Yield_Curve'],
                           dec_inc,
                           row['Avg_ratemove'],
                           row['Tenor_min'],
                           row['Tenor_max'],
                           'long' if row['AtRisk'] > 0 else 'short',
                           row['Ccy'],
                           old_format(row['AtRisk']))
            # No missing values in this row.
            # But, it's the 2nd part ('and') of a commentary. Use the template[0]:
            else:
                irdframe.iloc[idx, -1] = (row['comment'] + template[0]).\
                        format(row['Yield_Curve'],
                               dec_inc,
                               row['Avg_ratemove'],
                               row['Tenor_min'],
                               row['Tenor_max'],
                               'long' if row['AtRisk'] > 0 else 'short',
                               row['Ccy'],
                               old_format(row['AtRisk']))
        # If the row is the 2nd part ('and') of a commentary and the Yield Curve is from Gov bonds,
        # the only tenor being 'TOT',  use the template[1]:
        elif (row['comment'] == ' and ') & ('BD' in row['Yield_Curve']) &\
                ('Tenor_min' in missing_vals) & ('Tenor_max' in missing_vals):
            irdframe.iloc[idx, -1] = (row['comment'] + template[1]).\
                    format(row['Yield_Curve'][:3],
                           dec_inc,
                           row['Avg_ratemove'],
                           'long' if row['AtRisk'] > 0 else 'short',
                           row['Ccy'],
                           old_format(row['AtRisk']))
        # If the associated tenor is actually only a maturity date, use the template[2]
        elif missing_vals == ['Tenor_min']:
            irdframe.iloc[idx, -1] = (row['comment'] + template[2]).\
                format(old_format(row['DTD_SGD']),
                       row['Node'],
                       ' '.join(row['Raw_Component'].split()[:2]),
                       row['Yield_Curve'],
                       dec_inc,
                       row['Avg_ratemove'],
                       row['Tenor_max'])
        # If the row is the start of the commentary, the Yield Curve is from Gov bonds,
        # and the only tenor being 'TOT',  use the template[3]:
        elif ('BD' in row['Yield_Curve']) &\
                ('Tenor_min' in missing_vals) & ('Tenor_max' in missing_vals):
            irdframe.iloc[idx, -1] = (row['comment'] + template[3]).\
                    format(old_format(row['DTD_SGD']),
                           row['Node'],
                           ' '.join(row['Raw_Component'].split()[:2]),
                           row['Yield_Curve'],
                           dec_inc[:-1],
                           row['Avg_ratemove'])
        else:
            print("No comment. Something is not right here:\n", row)

    tmp_tot = irdframe['DTD_SGD'].sum()
    input(irdframe)
    print("\nIR Delta commentary list:")
    # Join the related comments into single strings via a whitespace.
    # Then drop the full stop, newline and the whitespace characters in the middle of the string:
    commentary = irdframe.groupby(['DTD_SGD', 'Node', 'Raw_Component', 'Ccy'], as_index=False)\
                                  ['comment'].apply(' '.join).str.replace('\.\\n ', '').\
                                  reset_index().sort_values('DTD_SGD', ascending=(tmp_tot < 0))
    # The IR Delta commentary output:
    # input(commentary.iloc[:, -1].tolist())

    # EXTRA: combine the two (or more) commentaries for the same portfolio into a single commentary:
    # Add the DTD amounts up for the row with the same nodes/portfolios
    # and merge the amounts with the other columns:
    pf_groups = irdframe.groupby('Node', as_index=False)[['DTD_SGD']].sum()
    df_tmp = pd.merge(commentary, pf_groups, on='Node', how='left')
    # Update the original DTD $ and drop the extra DTD (_y) column:
    df_tmp['DTD_SGD_x'] = df_tmp['DTD_SGD_y']
    df_tmp.drop('DTD_SGD_y', axis=1, inplace=True)
    df_tmp.rename(columns={'DTD_SGD_x': 'DTD_SGD', 0: 'comment'}, inplace=True)
    # Replace the DTD $ amount with the properly human-formatted amount inside the comment:
    pattern = "SGD.[0-9]+\.*[0-9]*[A-Za-z]*"  # e.g. "SGD-176.5K"
    helper = lambda x: re.sub(pattern, "SGD" + old_format(x['DTD_SGD']), x['comment'])
    df_tmp['comment'] = df_tmp.apply(helper, axis=1)
    # Join the related comments into single strings.
    # Then drop the full stop and newline in the middle of the string.
    # Finally, replace the 2nd occurrence of " S$ is " amount in the string with "and":
    commentary2 = df_tmp.groupby(['DTD_SGD', 'Node'], as_index=False)['comment'].\
            apply(' '.join).str.replace('\.\\n ', ' ').\
                    str.replace(" SGD.*is from", " and from").\
                    reset_index().sort_values('DTD_SGD', ascending=(tmp_tot < 0))
    # The new IR Delta commentary output:
    print(*commentary2.iloc[:, -1].tolist(), sep='\n')


def get_irdeltas(y_curve, p_folio, pf_table, pfc_sensi, nfc_sensi, pfccy_sensi, dtd_pl,
                 getter, dd_dict, dd_list, irdeltas):
    """
    Summary:
    For Credit Trading only. It processes the Nodes/portfolio with the key
    P&L contributions (in the dtd_pl direction), and stores their
    properties (required for the commentary) in the irdeltas DataFrame.
    The excessive single contributions (> 1.25 DTD_SGD) from a Yield Curve for
    a given Node, are offset by a contribution(s) in the opposite direction.
    ----------
    Parameters:
    ----------
    y_curve (str):          the Yield Curve
    p_folio (str):          the portfolio/Node
    pf_table (DataFrame):   either pfccy_sensi or pfccy_2_sensi pre-filtered for
                            a specific portfolio and one of its top yield curves
    pfc_sensi (DataFrame):  pf_sensi distilled by each portfolio keeping only
                            its top Raw Component value. The only portfolios
                            are those contributing to dtd_pl (of the same sign).
    nfc_sensi (DataFrame):  pf_sensi distilled by each portfolio keeping only
                            its top Raw Component value. The portfolios are
                            those NOT contributing to dtd_pl (opposite sign).
    pfccy_sensi (DataFrame): pf_sensi with each portfolio keeping only its top
                             Raw Component value, PL Currency and Yield Curve.
    dtd_pl (number):       the value from 'DTD_PL' col from the matching row
                           of ReportsQueryPL...csv. Only the SIGN matters.
    getter (function):     get_tenor_rates() function
    dd_dict (dict):        an empty dictionary
    dd_list (list):        a dictionary of lists, from irdeltas
    irdeltas (DataFrame):  the table where the summary of IR Delta portfolios is
    ----------
    Returns:
    ----------
    dd_dict (dict):        an empty dictionary
    dd_list (list):        the updated a dictionary of lists, from irdeltas
    irdeltas (DataFrame):  the updated table where the summary of IR Delta
                           Nodes/portfolios is stored
    """
    if (not pf_table.empty) & ('Sensi Date' in irdeltas.columns):
        sdate = pf_table['Sensi Date'].iloc[0]
    if getter.__name__ == 'get_tenors_rates':
        tmax, tmin, avrm, amount = getter(pf_table, dtd_pl)
    elif getter.__name__ == 'get_tenors_rates_by_day':
        tmp = getter(pf_table, dtd_pl)
        tmax, tmin, avrm, amount, sdate = (tmp[c].values for c in tmp.columns)
    else:
        tmax, tmin, avrm, amount, sdate = '', '', None, None, None
    dsgd_c = pfc_sensi.query("Node == @p_folio")['DTD SGD'].sum()
    dsgd_ccy = pfccy_sensi.query("Node == @p_folio")['DTD SGD'].sum()
    if 0 < (dsgd_c / dsgd_ccy) < 0.8:
        # an excessive contribution (dsgd_ccy) is coming from this Yield Curve
        # it needs to be offset by a contribution(s) in the opposite direction:
        tmp_dict = {'DTD_SGD': dsgd_ccy, 'Node': p_folio,
                    'Raw_Component': pf_table['Raw Component'].iloc[0],
                    'Yield_Curve': y_curve,
                    'Ccy': pf_table['PL Currency'].iloc[0],
                    'Tenor_max': tmax, 'Tenor_min': tmin,
                    'Avg_ratemove': avrm, 'AtRisk': amount}
        if 'Sensi Date' in irdeltas.columns:
            tmp_dict['Sensi Date'] = sdate
        # Get the slice of first 5 items from the tmp_dict,
        # and if it's not in dd_list, append the tmp_dict to irdeltas,
        # and update the dd_list with these 5 items:
        if dict(islice(tmp_dict.items(), 5)) not in dd_list:
            irdeltas = irdeltas.append(tmp_dict, ignore_index=True)
            dd_list = irdeltas[['DTD_SGD', 'Node', 'Raw_Component', 'Yield_Curve', 'Ccy']].\
                    to_dict('records', into=dd_dict)
        ndsgd_ccy = 0
        n_cnt = 0
        while 0 < (dsgd_c / (dsgd_ccy + ndsgd_ccy)) < 0.8:
            ny_curve = nfc_sensi.query("Node == @p_folio")['Yield Curve'].iloc[n_cnt]
            nf_table = nfc_sensi.query("(Node == @p_folio) & (`Yield Curve` == @ny_curve)")
            ndsgd_tmp = nf_table['DTD SGD'].sum()
            if getter.__name__ == 'get_tenors_rates':
                tmax, tmin, avrm, amount = getter(nf_table, dtd_pl)
            elif getter.__name__ == 'get_tenors_rates_by_day':
                tmp = getter(nf_table, dtd_pl)
                tmax, tmin, avrm, amount, sdate = (tmp[c].values for c in tmp.columns)
            tmp_dict = {'DTD_SGD': ndsgd_tmp, 'Node': p_folio,
                        'Raw_Component': nf_table['Raw Component'].iloc[0],
                        'Yield_Curve': ny_curve,
                        'Ccy': nf_table['PL Currency'].iloc[0],
                        'Tenor_max': tmax, 'Tenor_min': tmin,
                        'Avg_ratemove': avrm, 'AtRisk': amount}
            if 'Sensi Date' in irdeltas.columns:
                tmp_dict['Sensi Date'] = sdate
            if dict(islice(tmp_dict.items(), 5)) not in dd_list:
                irdeltas = irdeltas.append(tmp_dict, ignore_index=True)
                dd_list = irdeltas[['DTD_SGD', 'Node', 'Raw_Component', 'Yield_Curve', 'Ccy']].\
                    to_dict('records', into=dd_dict)
            ndsgd_ccy += ndsgd_tmp
            n_cnt += 1
    # else:  # no issue with the DTD SGD contribution from a Yield Curve being perhaps too large:
    elif not pf_table.empty:
        tmp_dict = {'DTD_SGD': dsgd_c, 'Node': p_folio,
                    'Raw_Component': pf_table['Raw Component'].iloc[0],
                    'Yield_Curve': y_curve, 'Ccy': pf_table['PL Currency'].iloc[0],
                    'Tenor_max': tmax, 'Tenor_min': tmin,
                    'Avg_ratemove': avrm, 'AtRisk': amount}
        if 'Sensi Date' in irdeltas.columns:
            tmp_dict['Sensi Date'] = sdate
        if dict(islice(tmp_dict.items(), 5)) not in dd_list:
            irdeltas = irdeltas.append(tmp_dict, ignore_index=True)
            dd_list = irdeltas[['DTD_SGD', 'Node', 'Raw_Component', 'Yield_Curve', 'Ccy']].\
                to_dict('records', into=dd_dict)

    return dd_dict, dd_list, irdeltas


def trim_irdeltas(irdeltas, pfc_sensi, pfccy_sensi, pfccy_2_sensi):
    """
    Summary:
    For Credit Trading only. It trims away the Yield Curve contributions
    to the Node/portfolio DTD_SGD if they fall below 20% of their DTD_SGD.
    Ditto for the associated commentaries.

    Parameters:
    irdeltas (DataFrame):  the table where the summary of IR Delta portfolios is
                           stored
                            a specific portfolio and one of its top yield curves
    pfc_sensi (DataFrame):  pf_sensi distilled by each portfolio keeping only
                            its top Raw Component value. The only portfolios
                            are those contributing to dtd_pl (of the same sign).
                            those NOT contributing to dtd_pl (opposite sign).
    pfccy_sensi (DataFrame): pf_sensi with each portfolio keeping only its top
                             Raw Component value, PL Currency and Yield Curve.
    pfccy_2_sensi (DataFrame): pf_sensi with each portfolio keeping only its 2nd
                               best Yield Curve.
    Returns:
    irdeltas (DataFrame):  the updated (trimmed) table where the summary of the
                           IR Delta portfolios is stored
    """
    # pf_contrib = lambda x: pfc_sensi.query("Node == @x")['DTD SGD'].sum()
    # yc1_contrib = lambda x: pfccy_sensi.query("Node == @x")['DTD SGD'].sum()
    # yc2_contrib = lambda x: pfccy_2_sensi.query("Node == @x")['DTD SGD'].sum()
    pfc_o = PassDaFrame(pfc_sensi)
    pfccy_o = PassDaFrame(pfccy_sensi)
    pfccy_2_o = PassDaFrame(pfccy_2_sensi)

    def yc_contrib(group):
        """
        Evaluates the fractional contribution from a Yield Curve for a Node/portfolio
        with the common DTD $ amount arising from two (or more) Yield Curves.
        :param group: DataFrame     the rows of DataFrame associated with a group from GroupBy
        :return: DataFrame          the rows of DataFrame updated with the new 'yc_frac' column
        """
        # split the rows by the amount-at-risk, the high(est) and low(est):
        crit_max = group.index == abs(group['AtRisk']).idxmax()
        crit_min = group.index == abs(group['AtRisk']).idxmin()
        # For the high(est) amount-at-risk evaluate the yc_frac based on the top Yield Curve, and
        # for the low(est) amount-at-risk evaluate the yc_frac based on the 2nd best Yield Curve:
        group['yc_frac'] = np.where(
            crit_max, pfccy_o.tab_pl_contrib(group.iloc[crit_max.tolist().index(True), 1]),
            pfccy_2_o.tab_pl_contrib(group.iloc[crit_min.tolist().index(True), 1]))
        return group

    comm_groups = irdeltas.groupby(['DTD_SGD', 'Node'], as_index=False)
    irdeltas = comm_groups.apply(yc_contrib)
    # irdeltas['yc_frac'] /= irdeltas['DTD_SGD']
    irdeltas['yc_frac'] /= irdeltas['Node'].map(pfc_o.tab_pl_contrib)
    irdeltas = irdeltas.query("yc_frac > 0.2").reset_index(drop=True)
    input(irdeltas)
    return irdeltas


def go_bonds_comments(pfc_sensi, dtd_pl):
    """
    Summary:
    ----------
    For Credit Trading only. Considers only the Credit Delta portfolios and
    for those extracts the top 3 P&L contributing Industry groups...
    Only the Nodes with all 3 groups contribution in the dtd_pl direction
    survive the sunsequent "distillation".
    Finally, it prints the commetaries for the Bond Spreads portfolios/Nodes
    contributing to the P&L.
    ----------
    Parameters:
    ----------
    pfc_sensi (DataFrame): The pre-distilled SENSI table with only the top P&L
                           contributing Raw Component for each portfolio.
    dtd_pl (number):       the value from 'DTD_PL' col from the matching row
                           of ReportsQueryPL...csv. Only the SIGN matters.
    """
    bondspreads = pfc_sensi.query("(`Raw Component` == 'Credit Delta Opening')")
    bondspreads = pivtab_best(bondspreads, 'Node', 'Product Type', 1, dtd_pl)
    bond_list = bondspreads.query("`Product Type` == 'Bond'")['Node'].tolist()
    if pfc_sensi.query("(`Raw Component` == 'Credit Delta Opening') & (Node in @bond_list)")\
            ['Industry'].isna().all():
        print(f"No Industry available for {bond_list}")
        return
    indqy = "(`Raw Component` == 'Credit Delta Opening') & (Node in @bond_list)"
    # Get top 3 contributing Industries for each Node/portfolio:
    bondustry = pivtab_best(pfc_sensi.query(indqy), 'Node', 'Industry', 3, dtd_pl)
    # Get the top Industry (if contributing in the right P/L direction):
    bond_spread_sensi = get_top_contribution(bondustry[['Node', 'Industry']],
                                             pfc_sensi, 'Node', 'Industry')
    bond_spread_sensi.drop(columns='index', inplace=True)
    ind = bond_spread_sensi.groupby(['Node', 'Instrument', 'Industry']).\
        agg({'Rate_move': 'mean', 'DTD SGD': 'sum'}).\
        sort_values('DTD SGD', ascending=(dtd_pl < 0)).\
        query("`DTD SGD` * @dtd_pl > 0")
    # Get the 2nd Industry (if contributing in the right P/L direction):
    bond_spread_sensi = get_top_contribution(bondustry[['Node', 'Industry_2']],
                                             pfc_sensi, 'Node', 'Industry_2')
    bond_spread_sensi.drop(columns='index', inplace=True)
    ind2 = bond_spread_sensi.groupby(['Node', 'Instrument', 'Industry']).\
        agg({'Rate_move': 'mean', 'DTD SGD': 'sum'}).\
        sort_values('DTD SGD', ascending=(dtd_pl < 0)).\
        query("`DTD SGD` * @dtd_pl > 0")
    ind = ind.append(ind2)
    # Get the 3rd Industry (if contributing in the right P/L direction):
    bond_spread_sensi = get_top_contribution(bondustry[['Node', 'Industry_3']],
                                             pfc_sensi, 'Node', 'Industry_3')
    bond_spread_sensi.drop(columns='index', inplace=True)
    ind2 = bond_spread_sensi.groupby(['Node', 'Instrument', 'Industry']).\
        agg({'Rate_move': 'mean', 'DTD SGD': 'sum'}).\
        sort_values('DTD SGD', ascending=(dtd_pl < 0)).\
        query("`DTD SGD` * @dtd_pl > 0")
    ind = ind.append(ind2).sort_values('DTD SGD', ascending=(dtd_pl < 0))
    # The *** loss if -dtd_pl > 0 else profit *** making bond spread portfolios:
    bond_set = {n[0] for n in ind2.index.tolist()}
    print("\nBond spread commentaries:")
    for bds in bond_set:
        indus3 = bondustry.loc[bondustry["Node"] == bds, ['Industry', 'Industry_2', 'Industry_3']].\
                values[0].tolist()
        comment = "SGD" + old_format(pfc_sensi.query('Node == @bds')['DTD SGD'].sum())
        comment += " from " + f"{bds}" + " credit delta where bulk of PL is from "
        comment += "{}, {}, {}".format(*indus3) + " where spreads"
        # if top12['Rate_move'].sum() > 0:
        lower_bound = abs(ind.query('Node == @bds')['Rate_move'].head(15).min())
        higher_bound = abs(ind.query('Node == @bds')['Rate_move'].head(10).max())
        if ind.query('Node == @bds')['Rate_move'].sum() > 0:
            comment += f" widened from {lower_bound:.2f}bps to {higher_bound:.2f}"
        else:
            comment += f" tightened from {higher_bound:.2f}bps to {lower_bound:.2f}"
        comment += 'bps.\n'
        print(comment)


def ct_demo(node_id, ddate, entity, pub_holiday, files):
    """
    Summary:
    --------
    Runs A demo of Credit Trading P&L attribution.
    ----------
    Parameters:
    ----------
    node_id (str):       the trading desk ('CREDIT TRADING' here)
    ddate (string):      the Business Date in the string format, e.g. '20190430'
    entity (string):     the trading DBS entity (typically 'DBSSG')
    pub_holiday (*):     if supplied (i.e. not None), it flags the previous day
                         was a public holiday
    Example: -------------------------------------------------------------------
    $ ./credtrade.py -d 20190412 -n CREDIT\ TRADING -e DBSSG
    """
    curdate = ddate
    ddate = parse(curdate)
    pf_sensi, plexp, _ = get_clean_portfolio(curdate=curdate,
                                             entity=entity,
                                             node_id=node_id,
                                             pub_holiday=pub_holiday,
                                             files=files)

    comment_raw = files + 'Controllers_PL_Commentary_'
    comment_raw += month_abbr[ddate.month] + '_' + str(ddate.year) + '.ods'
    df_comment_raw = read_ods(comment_raw, 1)
    df_comment_raw = df_comment_raw.head(-2)
    dtd_pl = get_dtd_pl(df_comment_raw, curdate, node_id)
    pf_sensi, _, _ = diag_sensi_vs_pl(pf_sensi, plexp, dtd_pl, n_rows=8)
    # portfolios' top Raw Component:
    # pf_comp = pivotTable(pf_sensi, 'Node', 'Raw Component', 1, dtd_pl)
    non_open = ('IR Delta', 'Credit Delta', 'IR Delta Derived', 'Credit Delta Derived',
                'IR Basis Delta', 'IR Basis Delta Derived')
    tmp_sensi = pf_sensi.drop(pf_sensi.query("`Raw Component` in @non_open").index, axis=0)
    pf_comp = pivtab_best(tmp_sensi, 'Node', 'Raw Component', 1, dtd_pl)
    # Now, filter only the top Raw Component from pf_sensi:
    pfc_sensi = get_top_contribution(pf_comp, pf_sensi, 'Node', 'Raw Component')
    pfc_sensi.drop(columns='index', inplace=True)
    input(pfc_sensi.groupby(['Node', 'Raw Component'])[['DTD SGD']].
          sum().sort_values('DTD SGD', ascending=(dtd_pl < 0)))

    # portfolio-RawComponent pairs' top PL Currency:
    tmp_sensi = pivtab_best(pfc_sensi, 'Node', 'PL Currency', 1, dtd_pl)
    # Now, filter only the top PL Currency from pf_sensi:
    pfcc_sensi = get_top_contribution(tmp_sensi, pfc_sensi, 'Node', 'PL Currency')
    pfcc_sensi.drop(columns='index', inplace=True)
    input(pfcc_sensi.groupby(['Node', 'Raw Component', 'PL Currency'])[['DTD SGD']].
          sum().sort_values('DTD SGD', ascending=(dtd_pl < 0)))

    # portfolio-RawComponent-PLCurrency triplets' top Yield Curves:
    tmp_sensi = pivtab_best(pfcc_sensi, 'Node', 'Yield Curve', 2, dtd_pl)
    colist = ['Node', 'Raw Component', 'PL Currency']
    # Now, filter the top Yield Curve from pf_sensi (note: pfc_sensi has only the top currency):
    pfccy_sensi = get_top_contribution(tmp_sensi, pfc_sensi, 'Node', 'Yield Curve')
    pfccy_sensi.drop(columns='index', inplace=True)
    yc1_pf = pfccy_sensi.groupby(colist + ['Yield Curve'])[['DTD SGD']].\
        sum().sort_values('DTD SGD', ascending=(dtd_pl < 0))
    pf_yc1 = yc1_pf.reset_index(level=(colist[1:] + ['Yield Curve']))[['Yield Curve']]
    # And now filter for the 2nd best Yield Curve:
    pfccy_2_sensi = get_top_contribution(tmp_sensi, pfc_sensi, 'Node', 'Yield Curve_2')
    yc2_pf = pfccy_2_sensi.groupby(colist + ['Yield Curve_2'])[['DTD SGD']].\
        sum().sort_values('DTD SGD', ascending=(dtd_pl < 0))
    pf_yc2 = yc2_pf.reset_index(level=(colist[1:] + ['Yield Curve_2']))[['Yield Curve_2']]
    input(pd.merge(pf_yc1, pf_yc2, left_index=True, right_index=True, how='outer'))
    # easier with a dict {Node: [list of Yield Curves]} than the DataFrame:
    yc12_dict = {i[0]: [i[3]] for i in yc1_pf.index}
    for i in yc2_pf.index:
        if i[0] in yc12_dict.keys():
            yc12_dict[i[0]].extend([i[3]])
        else:
            yc12_dict.update({i[0]: i[3]})

    nfc_sensi = pfc_sensi.query("`DTD SGD` * @dtd_pl < 0").\
        sort_values('DTD SGD', ascending=(dtd_pl > 0))
    colist = ['DTD_SGD', 'Node', 'Raw_Component', 'Ccy', 'Yield_Curve',
              'Tenor_max', 'Tenor_min', 'Avg_ratemove', 'AtRisk']
    irdeltas = pd.DataFrame(columns=colist)
    dd_dict = dict()
    dd_list = irdeltas[colist[:5]].to_dict('records', into=dd_dict)

    cnt_date = pf_sensi['Sensi Date'].nunique()
    if cnt_date > 1:
        daily_pl = pfc_sensi.groupby('Sensi Date')[['DTD SGD']].sum().reset_index()
        the_total = abs(daily_pl['DTD SGD'].sum())
        pl_dates = daily_pl.query("abs(`DTD SGD`) > 0.05 * @the_total")['Sensi Date'].tolist()
        sirdeltas = pd.DataFrame(columns=['Sensi Date'] + colist)
        ss_dict = dict()
        ss_list = sirdeltas[['Sensi Date'] + colist[:5]].to_dict('records', into=ss_dict)

    for p_folio in yc12_dict.keys():
        for y_curve in yc12_dict[p_folio]:
            if not y_curve:
                continue
            pf_table = pfccy_sensi.query("(Node == @p_folio) & (`Yield Curve` == @y_curve)")
            if pf_table.empty:
                continue
            dd_dict, dd_list, irdeltas = get_irdeltas(
                y_curve, p_folio, pf_table, pfc_sensi, nfc_sensi, pfccy_sensi, dtd_pl,
                get_tenors_rates, dd_dict, dd_list, irdeltas)

            if cnt_date > 1:
                for sdate in pl_dates:
                    spf_table = pf_table.query("`Sensi Date` == @sdate")
                    spfc_sensi = pfc_sensi.query("`Sensi Date` == @sdate")
                    snfc_sensi = nfc_sensi.query("`Sensi Date` == @sdate")
                    spfccy_sensi = pfccy_sensi.query("`Sensi Date` == @sdate")
                    ss_dict, ss_list, sirdeltas = get_irdeltas(
                        y_curve, p_folio, spf_table, spfc_sensi, snfc_sensi, spfccy_sensi,
                        dtd_pl, get_tenors_rates, ss_dict, ss_list, sirdeltas)

    irdeltas.sort_values('AtRisk', ascending=False, inplace=True)
    irdeltas.sort_values('DTD_SGD', ascending=(dtd_pl < 0), ignore_index=True, inplace=True)
    irdeltas = trim_irdeltas(irdeltas, pfc_sensi, pfccy_sensi, pfccy_2_sensi)
    go_irdelta_comments(irdeltas, dtd_pl)
    if cnt_date > 1:
        for sdate in pl_dates:
            tmpird = sirdeltas.query("`Sensi Date` == @sdate").drop(columns='Sensi Date')
            tmpird.sort_values('AtRisk', ascending=False, inplace=True)
            tmpird.sort_values('DTD_SGD', ascending=(dtd_pl < 0), ignore_index=True, inplace=True)
            print("\n", sdate)
            tmpird = trim_irdeltas(tmpird, pfc_sensi, pfccy_sensi, pfccy_2_sensi)
            go_irdelta_comments(tmpird, dtd_pl)

    go_bonds_comments(pfc_sensi, dtd_pl)
    if cnt_date > 1:
        for sdate in pl_dates:
            print("\n", sdate)
            pfc_sensi = pfc_sensi.query("`Sensi Date` == @sdate")
            go_bonds_comments(pfc_sensi, dtd_pl)
