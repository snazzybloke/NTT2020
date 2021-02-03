#!/usr/bin/env python3
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from dateutil.parser import parse
import pathlib
from FXO.fxo_xl import pl_by_components
from FXO.enquire5 import *
from FXO.common_futils_lcy import get_idxminmax2, get_selection2, sign_it, lst_unpack
from IRD.irvega_xl import get_tenors_ordered
from Common_Flow.commentary_funtions import common_function
from openpyxl.styles import Font
from config.configs import getLogger
from config import configs
env = configs.env
logger = getLogger("run_fsoci.py")


def fsdelta_u(pf_fsoci, u):
    """the contributing Underlying values in FS Delta
    """
    pf_fsoci.rename({"RiskDiff": "Risk Difference", "DiffInLcy": "Sensi_OCI_LCY",
        "DtdDay": "DTD Day", "T_mkt": "T", "T_1_mkt": "T-1"},
            axis=1, inplace=True)
    fs_u= pf_fsoci.query("(Component == 'FS Delta') & (Underlying == @u)")
    get_tenors_ordered(fs_u, 'Tenor')
    fs_u['ord'] -= fs_u['ord'].min() - 1
    fs_upiv = fs_u.pivot_table(index=['ord', 'Tenor', 'DTD Day', 'T', 'T-1'],
            values=['Risk Difference', 'Sensi_OCI_LCY'],
            aggfunc=sum,
            fill_value=0,
            margins=True,
            margins_name='Total')
    #fs_upiv.sort_values('ord', inplace=True)
    # Boolean array:
    # bool_array = fs_upiv.index.droplevel([1, 2, 3, 4]) == 'Total'
    fs_total = fs_upiv.loc['Total', 'Sensi_OCI_LCY'].iloc[0]
    if np.isclose([fs_total], [0])[0]:
        print(f"Nothing to see here:\n{fs_upiv}")
    else:
         fs_upiv.drop(fs_upiv.query("ord == 'Total'").index, axis=0, inplace=True)
         print(f"FS Delta for Underlying {u} breakdown:\n{fs_upiv}")
         fs_upiv.reset_index(inplace=True)
   
         if (fs_upiv.groupby('Tenor')['T'].count() > 1).any():
             # Assign 'Rate_move' to market rate 'T' if the date is the latest
             # and negative previous market rate 'T-1' for other dates:
             tmp = sorted(fs_upiv['DTD Day'].unique().tolist())
             day = [tmp[0], tmp[-1]]
             crt = fs_upiv['DTD Day'] == day[1]
             fs_upiv = fs_upiv.assign(Rate_move=100 * fs_upiv['T'].where(crt, fs_upiv['T-1']))
         else:
             fs_upiv = fs_upiv.assign(Rate_move=100 * (fs_upiv['T'] - fs_upiv['T-1']))
   
         fnl = fs_upiv.groupby(['ord', 'Tenor'], as_index=False).agg({
             'DTD Day': max, 'Risk Difference': sum, 'Sensi_OCI_LCY': sum, 'Rate_move': sum
             })
   
         print(get_selection2(fnl, fs_total, 'ord', 'Sensi_OCI_LCY', [0.8, 0.1, 0.8, 1.2, 0.0]))
         fnl = fnl.assign(frac=fnl['Sensi_OCI_LCY'] / fs_total)
         print(get_idxminmax2(fnl, 'Tenor', 'frac'))


def doit_doit(pf_fsoci, xclfile, node_id=None, ddate=None):
    """
    pf_fsoci is a PLSensiDay1VsOCI table
    """
    pf_fsoci.rename({"Tenor GPC": "Tenor"}, axis=1, inplace=True)
    get_tenors_ordered(pf_fsoci, 'Tenor')

    xclwrt = pd.ExcelWriter(xclfile, mode="w", engine='openpyxl')

    # rename the columns to the "standard" PL Sensi names:
    ren_dict = {"Day": "Business Date", "DTD Day": "Sensi Date", "Portfolio": "Node",
                "Underlying": "Yield Curve",
                "Und.Maturity": "Underlying Maturity", "Sensi_OCI_LCY": "DTD LCY",
                "T": "Mkt (T)", "T-1": "Mkt (T-1)", "Risk Difference": "Sensi Risk (OCY)",
                "DtdDay": "Sensi Date", "RawComponent": "Raw Component",
                "ProductType": "Product Type", "UndMaturity": "Underlying Maturity",
                "DiffInLcy": "DTD LCY", "T_mkt": "Mkt (T)", "T_1_mkt": "Mkt (T-1)",
                "RiskDiff": "Sensi Risk (OCY)"}
    # Note: keep pf_fsoci columns, and renamed columns in df_fsoci
    df_fsoci = pf_fsoci.rename(ren_dict, axis=1)
    df_fsoci = df_fsoci.assign(LCY=["SGD"] * len(df_fsoci))

    by_nodes = df_fsoci.pivot_table(index='Node',
                                    values='DTD LCY',
                                    aggfunc=sum,
                                    margins=True,
                                    margins_name='Total')

    by_comps = df_fsoci.pivot_table(index=['Business Date', 'Component'],
                                    values='DTD LCY',
                                    aggfunc=sum,
                                    margins=True,
                                    margins_name='Total')
    total = by_comps.loc[('Total', ''), 'DTD LCY']

    pd.DataFrame().to_excel(xclwrt, sheet_name='FSOCI_SENSI', startrow=0, startcol=0)
    wsheet = xclwrt.book["FSOCI_SENSI"]
    selected = pl_by_components(df_fsoci, total, "The SENSI way", xclwrt, wsheet)
    # The key component:
    bigc = (by_comps.query("Component in @selected")['DTD LCY'] / total).idxmax()[1]

    if "IR Vega" in selected:
        # shift it to the last spot, due to USD filtering and column name changes:
        selected.remove("IR Vega")
        selected.append("IR Vega")
    # if "Fx Delta" in selected:
        # Unfortunately the FXO SENSI data often missing for the same date :-(
        # selected.remove("Fx Delta")
    print(f"Selected components: {selected}")

    if not ddate:
        ddate = df_fsoci["Sensi Date"].mode().iloc[0]
    npth = 'GFM>DBS>DBS Singapore>DBSSG>Treasury>Sales and Trading>'

    df_results = pd.DataFrame(columns=['Component', 'PnL', 'Comment'])
    ddt = parse(ddate).strftime("%d-%m-%Y")
    tmp_val = ['All', total,
            f"{sign_it(total, 'SGD')} (FSOCI) for {node_id} desk for {ddt} " +
                    lst_unpack("mainly from", selected)]
    tmp_dic = {c: v for  c, v in zip(df_results.columns, tmp_val)}
    df_results = df_results.append(tmp_dic, ignore_index=True)

    for comp in selected:
        print(f"Component {comp}")
        sensi_nc = df_fsoci.query("Component == @comp")
        if comp == "IR Vega":
            npth += 'Interest Rate Desk>Interest Rate Derivatives & Structuring'
            sensi_nc = sensi_nc[sensi_nc['Yield Curve'].str.contains("USD", na=False)]
            # add new columns 'Currency', 'Strike' and 'Type'. Rename two:
            sensi_nc = sensi_nc.assign(Currency=["USD"] * len(sensi_nc))
            sensi_nc = sensi_nc.assign(Strike=[np.infty] * len(sensi_nc))
            sensi_nc.rename({'Yield Curve': 'Underlying', 'Currency': 'PL Currency'}, axis=1, inplace=True)
            sensi_nc = sensi_nc.assign(Type = '')
            sensi_nc['Type'] = np.where(sensi_nc['Tenor'].isna(),
                    ["PL Restore – Imported"]*len(sensi_nc), ['']*len(sensi_nc))
        elif comp == "IR Delta":
            npth += 'Credit Desk>CREDIT TRADING'
        elif (comp == "New deals") or ("Fx" in comp):
            npth += 'FX Desk>FX Options>'
            sensi_nc = sensi_nc.assign(Entity=["DBSSG"] * len(sensi_nc))
        # elif comp == "Credit Delta":
        #     npth += 'Interest Rate Desk>Interest Rate Trading'
        #  OR
        #     npth += 'Credit Desk>CREDIT TRADING'
        # elif comp == ...???
        #     npth += 'T&M Structured Funding'
        # elif comp == "CreditDefaultSwap":
        #     npth += 'Credit Desk>Credit Derivatives'
        elif comp == "FS Delta":
            # comp = "IR Delta"
            # sensi_nc['Component'].replace('FS Delta', 'IR Delta', inplace=True)
            npth += 'Credit Desk>CREDIT TRADING'
        else:
            pass
        pl_comp = sensi_nc['DTD LCY'].sum()
        comment, xclwrt = common_function(sensi=sensi_nc, component=comp, pl_com=pl_comp,
                                          writer=xclwrt, date=ddate, nodepath=None)
        tmp_dict = {'Component': comp, 'PnL': pl_comp, 'Comment': comment}
        df_results = df_results.append(tmp_dict, ignore_index=True)

    # Are there mutiple Yield Curves (i.e., Underlying values) in FS Delta?
    und_lst = pf_fsoci.loc[pf_fsoci['Component'] == 'FS Delta', 'Underlying'].unique()
    print(f"FS Delta unique Underlying values: {und_lst}")
    for u in und_lst:
        fsdelta_u(pf_fsoci, u)

    print('\n', df_results)
    df_results.to_excel(xclwrt, sheet_name='Commentary', index=False, startrow=3, startcol=1)
    bold_font = Font(bold=True, size=12)
    wsheet = xclwrt.book["Commentary"]
    wsheet["B1"] = "SUMMARY"
    wsheet["B1"].font = bold_font
    wsheet["B3"] = "Selection Table"
    # Generate the final commentary from the last column:
    comment = df_results.iloc[0, -1] + " "
    # Iterate over the rows of Series df_results.iloc[1:, -1]:
    for i_com in df_results.iloc[1:, -1].iteritems():
        comment += " \n[" + str(i_com[0]) + "] "
        comment += i_com[1]
    comment += "."
    i_row = df_results.shape[0] + 8
    print('\n', comment, '\n')
    wsheet["B" + str(i_row + 8)] = "COMMENTARY"
    wsheet["B" + str(i_row + 8)].font = bold_font
    wsheet["B" + str(i_row + 9)] = ' '.join(comment)
    wsheet["B" + str(i_row + 9)].font = bold_font
    xclwrt.close()
    return df_results, selected, comment


def fsoci_demo(dfile, ddate=None):
    """
    FSOCI table from a csv file dfile
    """
    path = pathlib.Path().cwd() / "meta_data"
    if ddate:
        xclfile = path / f"{parse(ddate).strftime('%Y-%m-%d')}_FSOCI.xlsx"
    else:
        xclfile = path / "tmp_FSOCI.xlsx"
    pf_fsoci = pd.read_csv(dfile)
    node_id = "Unknown"
    df_results, selected, comment = doit_doit(pf_fsoci, xclfile, node_id, ddate)
    comm_dic = {'commentary': comment, 'file_name': xclfile, 'components': selected}
    return comm_dic


def fsoci_db(ddate, npth=None, tup5db=None):
    """
    Summary:
    -----------
    Same as fsoci_demo() above, but relying on iWork DB to extract the data.

    Parameters:
    -----------
    ddate (str):    the Business Date in the string format, e.g. '2019-04-30'
    npth (str):     the trading desk
    tup5db (tuple): the 5-tuple carrying the user, password, IP address, port and DB name details

    Returns:
    -----------
    df_results (DataFrame): the table with 5 rows (for New deals, Fx Delta/Gamma/Vega/smiles) and 6
                            cols: Component, Selected, PL, Ccy/Instrument, Product Type, Commentary

    """
    path = pathlib.Path().cwd() / "meta_data"
    xclfile = path / f"{parse(ddate).strftime('%Y-%m-%d')}_FSOCI.xlsx"

    if not npth:
        npth = "GFM>DBS>DBS Singapore>DBSSG>Treasury>Sales and Trading>Interest Rate Desk>Interest Rate Derivatives & Structuring"

    entity, node_id = np.array(npth.split('>'))[[3, -1]]
    if not tup5db:
        tup5db = (env['username3'], env['password3'], env['host3'], env['port3'], env['database3'])
    mydb = PyMyDB(tup5db)
    mysql = TheQueries(mydb, ddate, entity, node_id)
    mysql.get_node_tree_path()
    dtd_pl = mysql.get_dtd_pl()
    pf_fsoci = mysql.get_fsoci()
    if pf_fsoci.empty:
        xclwrt.save()
        return {"commentary": [], "file_name": xclfile, "components": []}

    df_results, selected, comment = doit_doit(pf_fsoci, xclfile, node_id, ddate)
    comm_dic = {'commentary': comment, 'file_name': xclfile, 'components': selected}
    return comm_dic


if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--date', '-d',
                    required=False, help="date in this format '2019-04-30'")
    AP.add_argument('--node_id', '-n', choices=['IRD', 'CREDIT TRADING', 'FX Options'],
                    required=False, help="Trading Desk (e.g. 'IRD', 'Credit Trading', 'Fx Options'")
    AP.add_argument('--file', '-f',
                    required=False, help="Path to the input FSOCI table file")
    ARG = vars(AP.parse_args())

    if ARG['file']:
        cdic = fsoci_demo(ARG['file'], ARG['date'])
        print(cdic)
    elif ARG['date']:
        cdic = fsoci_db(ARG['date'], ARG['node_id'])
        print(cdic)
    else:
        pass
