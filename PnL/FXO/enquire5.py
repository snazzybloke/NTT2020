# coding: utf-8

####################################################
# Author: Ante Bilic                               #
# Since: May 10, 2020                              #
# Copyright: The PLC Project                       #
# Version: N/A                                     #
# Maintainer: Ante Bilic                           #
# Email: ante.bilic.mr@gmail.com                   #
# Status: N/A                                      #
####################################################

"""
Two classes for  dealing with the iWork DB.
"""

from config import configs
env = configs.env
from config.configs import getLogger
logger = getLogger("enquire5.py")
import pymysql.cursors
import pandas as pd
import datetime


class PyMyDB():
    """
    Methods relevant for connecting to the iWork DB and
    extracting the data out of it.Uses the pymysql.
    """
    def __init__(self, tup5db=None):
        """connects to the DB"""
        self._db_connection = None
        if not tup5db:
            tup5db = ("iworkro",
                      "ArdW8veX5+",
                      "10.68.21.146",
                      6603,
                      "iwork")
            # tup5db = (env['username3'],
            #           env['password3'],
            #           env['host3'],
            #           env['port3'],
            #           env['database3'])
            logger.info("Not connecting to MySQL DB")
        try:
            self._db_connection = pymysql.connect(
                user=tup5db[0],
                passwd=tup5db[1],
                host=tup5db[2],
                port=tup5db[3],
                db=tup5db[4],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor)
            self._db_cur = self._db_connection.cursor()
            logger.info("Connection to MySQL DB successful")
        except Exception as ex0:
            logger.info(f"The error '{ex0}' occurred.")

    def query(self, query):
        """Executes the SQL query and returns the raw result"""
        try:
            with self._db_cur as cursor:
                return cursor.execute(query)
        finally:
            logger.info('Closing the DB connection')
            self._db_connection.close()

    def get_dataframe(self, query, chunksize=None):
        """Executes the SQL query and returns the result as a DataFrame table"""
        table = pd.read_sql_query(query, con=self._db_connection, chunksize=chunksize)
        return table

    def __del__(self):
        self._db_connection.close()


class TheQueries:
    """
    Methods relevant for getting the data from the iWork DB and
    exchange the parameters using the common object.
    The constructor expects a PyMyDB object to work with.
    The only difference between the two is the format for the str_to_date()
    which is '%%Y-%%m-%%d' for the former and '%Y-%m-%d' latter.
    NODE_PATH, similarly, gets appended with '>%%' for the former, whereas with '>%' for the atter.
    """

    def __init__(self, my_db_obj, sdate='2020-03-16', entity='DBSSG', node_id='FX Options'):
        self._db_obj = my_db_obj
        try:
            self.s2date = "str_to_date('" + sdate + "', '%Y-%m-%d')"
        except Exception:
            self.s2date = "str_to_date('" + str(sdate) + "', '%Y-%m-%d')"
        self.entity = entity
        self.node_id = node_id
        self.treenbr = None
        self.nodenbr = None
        self.node_path = None
        self.sdate = sdate
        self.v_tree_id = 'GFM'
        self.s3date = datetime.datetime.strptime(sdate, "%Y-%m-%d").date()


    def get_node_tree_path(self):
        """Executes the SQL query which generates the values for the parameters TREENBR,
        NODENBR and NODE_PATH for the given day/nodeid"""
        query = f"""
        SELECT TREENBR,NODENBR,NODE_PATH from tree_hierarchy 
        where TREEID= '{self.v_tree_id}'
        and DAY = {self.s2date}
        and nodeid='{self.node_id}'
        and NODE_LEVELID='Sub Product Lines'
        and NODE_PATH like '%%>{self.entity}>%%'
        order by NODE_PATH      
        LIMIT 1;
        """
        df_ntp = self._db_obj.get_dataframe(query=query)
        if not df_ntp.empty:
            self.treenbr = df_ntp['TREENBR'].iloc[0]
            self.nodenbr = df_ntp['NODENBR'].iloc[0]
            self.node_path = df_ntp['NODE_PATH'].iloc[0] + '>%'


    def get_dtd_pl(self):
        """Executes the SQL query which generates the Day-to-day PnL and returns the number"""
        query = f"""
        SELECT SUM(DTD_SGD) FROM pl_todate p
        WHERE p.day = {self.s2date}
        AND TREENBR = {self.treenbr} AND NODENBR = {self.nodenbr}
        AND TYPE IN ('PL') and version=0 and p.status='A'
        """
        df_dtd = self._db_obj.get_dataframe(query=query)
        if not df_dtd.empty:
            return df_dtd.iloc[0, 0]
        else:
            return None


    def get_sensi(self):
        """Executes the SQL query which generates the SENSI data for the given date/node/entity and
        returns their DataFrame table"""
        query = f"""
        SELECT     p.DAY                AS "Business Date",
                   p.DTD_DAY            AS "Sensi Date",
                   port.nodeid          AS "Node",
                   e.ENTITYID                   AS "Entity",
                   c.`PL_COMPID`        AS "Component",
                   p.PRODUCT_TYPE       AS "Product Type",
                   (select currid from currency where currnbr=p.currnbr) "PL Currency",
                   p.val_group          AS "Val Group",
               p.instrument         AS "Instrument",
               (select currid from currency where currnbr=e.RPT_CURRNBR) "LCY",
                   m.`COMPONENT`                AS "Raw Component",
                   p.product_group      AS "Product Group",
               p.issuer_curve       AS "Issuer Curve",
               p.issuer                 AS "Issuer Name",
               p.seniority              AS "Seniority",
               p.underlying             AS "Underlying",
               p.yield_curve            AS "Yield Curve",
               p.tenor                  AS "Tenor",
               p.und_maturity       AS "Underlying Maturity",
               p.quotation              AS "Quotation",
               p.multiplier             AS "Multiplier",
               (SELECT
                                        case
                                                when p.`TYPE` = 'trade' then 'PL Sensi - trade'
                                                when p.`TYPE` = 'imported' then 'PL Explain - imported'
                                                when p.`TYPE` = 'derived' then 'PL Sensi - derived'
                                                ELSE p.`TYPE`
                                    end )AS Type ,
               p.expected_explain       AS "PL Explain",
               p.actual_sensi           AS "PL Sensi",
               p.mkt_data_strike                 AS "Strike",
               p.t_mkt                  AS "Mkt (T)",
               p.t_1_mkt                AS "Mkt (T-1)",
               p.value                  AS "Sensi Risk (OCY)",
               p.dtd_orig               AS "DTD ORIG",
               p.dtd_local              AS "DTD LCY",
               p.dtd_sgd                AS "DTD SGD",
               (
                                SELECT INDUSTRY from price pr where pr.DAY= '2020-03-25'
                                and `TYPENBR` IN (select paramnbr from param p where p.paramid='Party')
                                AND pr.priceid = p.issuer
                                ) "Industry"

        from pl_sensi p,
         (
         select th.NODENBR, th.NODEID from tree_hierarchy th where  th.DAY = {self.s2date}
          AND th.treeid='{self.v_tree_id}'
          AND th.NODE_PATH LIKE '{self.node_path}'
          AND th.NODE_LEVELID='Portfolio'
          ) as port,
          pl_compmap m, pl_comp c, node n, entity e
          where p.day= {self.s2date}
          AND TREENBR=0
          AND p.nodenbr=port.nodenbr
          AND p.type<>'intermediate'
          AND p.PL_COMPMAPNBR=m.pl_compmapnbr
          AND m.PL_COMPNBR=c.PL_COMPNBR
          AND p.status='A' AND p.VERSION=0
          AND n.NODENBR=p.NODENBR AND n.ENTITYNBR = e.ENTITYNBR
          """
        pf_sensi = self._db_obj.get_dataframe(query)
        if not pf_sensi.empty:
            cols = ['Business Date', 'Sensi Date', 'Node', 'Entity', 'Component', 'Product Type',
                    'PL Currency', 'Val Group', 'Instrument', 'LCY', 'Raw Component',
                    'Product Group', 'Issuer Curve', 'Issuer Name', 'Seniority', 'Underlying',
                    'Yield Curve', 'Tenor', 'Underlying Maturity', 'Quotation', 'Multiplier',
                    'Type', 'PL Explain', 'PL Sensi', 'Strike', 'Mkt (T)', 'Mkt (T-1)',
                    'Sensi Risk (OCY)', 'DTD ORIG', 'DTD LCY', 'DTD SGD', 'Industry']
            ### pf_sensi = pf_sensi.drop_duplicates(subset=cols, keep='first')
            pf_sensi['DTD SGD'] = pd.to_numeric(pf_sensi['DTD SGD'])
            pf_sensi["Mkt (T)"] = pd.to_numeric(pf_sensi["Mkt (T)"])
            pf_sensi["Mkt (T-1)"] = pd.to_numeric(pf_sensi["Mkt (T-1)"])
            pf_sensi = pf_sensi.assign(Rate_move=pf_sensi["Mkt (T)"] - pf_sensi["Mkt (T-1)"])
            return pf_sensi
        else:
            return None


    def get_plva(self):
        """Executes the SQL query which generates the PLVA data for the given date/node/entity and
        returns their DataFrame table"""
        query = f"""
             select p.day as "Business Date",
             port.nodeid as "Node",
             port.NODE_LEVELID as "Node Level",
             e.entityid as "Entity",
             c.pl_compid as "Component",
             p.product_type as "Product Type",
             p.instrument AS "Instrument",
             (select currid from currency where currnbr=p.currnbr) "PL Currency",
             (select currid from currency where currnbr=e.RPT_CURRNBR) "LCY",
             p.dtd_local as "DTD LCY",
             p.wtd_local as "WTD LCY",
             p.mtd_local as "MTD LCY",
             p.ytd_local as "YTD LCY",
             p.dtd_sgd as "DTD SGD",
             p.wtd_sgd as "WTD SGD",
             p.mtd_sgd as "MTD SGD",
             p.ytd_local as "YTD SGD"
             from
             pl_explain p,
             (
                  select NODENBR, NODEID, NODE_LEVELID from tree_hierarchy th where  th.DAY = {self.s2date}
                  AND treeid= '{self.v_tree_id}'
                  AND NODE_PATH LIKE '{self.node_path}'
                  AND NODE_LEVELID='Portfolio'
             ) as port,
              pl_compmap m, pl_comp c,node n, entity e
              where p.day={self.s2date}
                AND TREENBR=0
                AND p.nodenbr=port.nodenbr  
                AND p.PL_COMPMAPNBR=m.pl_compmapnbr
                AND m.`PL_COMPNBR`=c.`PL_COMPNBR`
                AND p.status='A' AND p.VERSION=0
                and n.`NODENBR`=p.`NODENBR`
                and n.`ENTITYNBR`=e.`ENTITYNBR`
        """
        df_plva = self._db_obj.get_dataframe(query)
        if not df_plva.empty:
            return df_plva
        else:
            return None


    def plva_query(self):
        """
        query fetches the whole PL Sensi data for the node_id and business day combination

        return PL Sensi as dataframe object
        """

        query = f"""
        SELECT
            p.day as "Business Date",
            port.nodeid as "Node",
            port.NODE_LEVELID as "Node Level",
            e.entityid as "Entity",
            c.pl_compid as "Component",
            p.product_type as "Product Type",
            p.instrument AS "Instrument",
            (select currid from currency where currnbr=p.currnbr) "PL Currency",
            (select currid from currency where currnbr=e.RPT_CURRNBR) "LCY",
            p.dtd_local as "DTD LCY",
            p.wtd_local as "WTD LCY",
            p.mtd_local as "MTD LCY",
            p.ytd_local as "YTD LCY",
            p.dtd_sgd as "DTD SGD",
            p.wtd_sgd as "WTD SGD",
            p.mtd_sgd as "MTD SGD",
            p.ytd_sgd as "YTD SGD"
        from
        pl_explain p,
        (
         select NODENBR, NODEID, NODE_LEVELID from tree_hierarchy th 
         where  th.DAY = {self.s2date}
         AND treeid = '{self.v_tree_id}'
         AND NODE_PATH LIKE '{self.node_path}'
         AND NODE_LEVELID='Portfolio'
        ) as port,
         pl_compmap m, pl_comp c,node n, entity e
         where p.day={self.s2date}
           AND TREENBR=0
           AND p.nodenbr=port.nodenbr  
           AND p.PL_COMPMAPNBR=m.pl_compmapnbr
           AND m.`PL_COMPNBR`=c.`PL_COMPNBR`
           AND p.status='A' AND p.VERSION=0
           AND n.`NODENBR`=p.`NODENBR`
           AND n.`ENTITYNBR`=e.`ENTITYNBR`
          """

        df_plva = self._db_obj.get_dataframe(query)
        if not df_plva.empty:
            return df_plva
        else:
            return None


    def get_fsoci(self):
        """Executes the SQL query which generates the Day1VsOCI"""
        query = f"""
        With tbl_node_path as (
        SELECT DISTINCT TH.nodenbr as nodenbr
        FROM tree_hierarchy th
        WHERE th.DAY BETWEEN '{self.s3date}' AND '{self.s3date}'
        AND th.status='A'
        AND th.NODE_PATH LIKE '{self.node_path}'
        ),
        tbl_node as (
        SELECT n.nodenbr,n.nodeid,e.rpt_currnbr,n.pl_method
        FROM node n, entity e
        WHERE n.entitynbr=e.entitynbr
        AND n.status='A'
        AND n.node_levelnbr=0
        AND n.nodenbr IN (
        SELECT DISTINCT pr.nodenbr
        FROM pl_sensi_OCI pr
        WHERE pr.DAY BETWEEN '{self.s3date}' AND '{self.s3date}'
        AND pr.status='A')
        AND n.nodenbr IN ( SELECT nodenbr FROM tbl_node_path )
        AND e.entityId = 'DBSSG'
        ),
        prevail_raw as (
        select ps.nodenbr, ps.dtd_local, ps.dtd_sgd, ps.product_type, ps.underlying, ps.yield_curve, c.currid, ps.quotation, ps.issuer_curve,
        ps.value, ps.T_mkt, ps.T_1_mkt, pc.PL_COMPID, pcm.component, ps.UND_MATURITY, ps.tenor, day, dtd_day
        from PL_SENSI_OCI ps, pl_comp pc, PL_COMPMAP pcm, CURRENCY c
        where ps.DAY BETWEEN '{self.s3date}' AND '{self.s3date}'
        AND ps.nodenbr IN (SELECT NODENBR FROM tbl_node)
        AND ps.pl_compmapnbr = pcm.PL_COMPMAPNBR
        AND pc.PL_COMPNBR = pcm.PL_COMPNBR
        AND pcm.status = 'A'
        AND ps.status = 'A'
        AND ps.version = 0
        AND ps.status = 'A'
        AND ps.treenbr = 0
        AND ps.currNbr = c.currNbr
        AND c.status = 'A'
        ),
        prevail AS (
        select nodenbr, dtd_local, dtd_sgd, product_type, currid,
        case
        when pl_compid = 'FS Delta' THEN yield_curve
        when pl_compid = 'Fx Delta' THEN currid
        when pl_compid = 'IR Delta' THEN yield_curve
        when pl_compid = 'IR Basis Delta' THEN yield_curve
        when pl_compid = 'Fx Vega' THEN quotation
        when pl_compid = 'Credit Delta' AND product_type in ('CreditDefaultSwap', 'CDS') then issuer_curve
        ELSE Underlying
        End as underlying, tenor, UND_MATURITY, value, T_mkt, T_1_mkt, day, dtd_day, pl_compid, component
        from prevail_raw
        ),
        prevail_by_comp AS (
        SELECT pr.nodenbr, coalesce(pr.product_type, '') AS product_type, coalesce(pr.UNDERLYING, '') AS underlying, coalesce(pr.tenor, '-') as tenor,
        sum(coalesce(pr.dtd_local,0)) AS dtd_local, sum(coalesce(pr.dtd_sgd,0)) AS dtd_sgd, sum(coalesce(pr.value,0)) as value, sum(coalesce(pr.T_mkt,0)) as T_mkt,
        max(coalesce(pr.T_1_mkt,0)) as T_1_mkt, pr.DAY, pr.DTD_DAY, pr.PL_COMPID, pr.Component, coalesce(pr.UND_MATURITY, '-') as und_maturity
        FROM prevail pr
        GROUP BY pr.nodenbr, pr.product_type, pr.underlying, pr.DAY, pr.DTD_DAY, pr.PL_COMPID, pr.component, pr.UND_MATURITY, pr.tenor
        ),
        incept_raw as (
        select ps.nodenbr, ps.dtd_local, ps.dtd_sgd, ps.product_type, ps.underlying, ps.yield_curve, c.currid, ps.quotation, ps.issuer_curve,
        ps.value, ps.T_mkt, ps.T_1_mkt, pc.PL_COMPID, pcm.component, ps.UND_MATURITY, ps.tenor, ps.day, ps.dtd_day
        from PL_SENSI ps, pl_comp pc, PL_COMPMAP pcm, CURRENCY c
        where ps.DAY BETWEEN '{self.s3date}' AND '{self.s3date}'
        AND ps.nodenbr IN (SELECT NODENBR FROM tbl_node)
        AND ps.pl_compmapnbr = pcm.PL_COMPMAPNBR
        AND pc.PL_COMPNBR = pcm.PL_COMPNBR
        AND pcm.status = 'A'
        AND ps.status = 'A'
        AND ps.version = 0
        AND ps.status = 'A'
        AND ps.treenbr = 0
        AND ps.currNbr = c.currNbr
        AND c.status = 'A'
        ),
        incept AS (
        select nodenbr, dtd_local, dtd_sgd, product_type, currid,
        case
        when pl_compid = 'FS Delta' THEN yield_curve
        when pl_compid = 'Fx Delta' THEN currid
        when pl_compid = 'IR Delta' THEN yield_curve
        when pl_compid = 'IR Basis Delta' THEN yield_curve
        when pl_compid = 'Fx Vega' THEN quotation
        when pl_compid = 'Credit Delta' AND product_type in ('CreditDefaultSwap', 'CDS') then issuer_curve
        ELSE Underlying
        End as underlying, tenor, UND_MATURITY, value, T_mkt, T_1_mkt, day, dtd_day, pl_compid, component
        from incept_raw
        ),
        incept_by_comp AS (
        SELECT ic.nodenbr, coalesce(ic.product_type, '') AS product_type, coalesce(ic.UNDERLYING, '') AS underlying, coalesce(ic.tenor, '-') as tenor,
        sum(coalesce(ic.dtd_local,0)) AS dtd_local, sum(coalesce(ic.dtd_sgd,0)) AS dtd_sgd, sum(coalesce(ic.value,0)) as value, sum(coalesce(ic.T_mkt,0)) as T_mkt,
        max(coalesce(ic.T_1_mkt,0)) as T_1_mkt, ic.DAY, ic.DTD_DAY, ic.PL_COMPID, ic.Component, coalesce(ic.UND_MATURITY, '-') as und_maturity
        FROM incept ic
        GROUP BY ic.nodenbr, ic.product_type, ic.underlying, ic.DAY, ic.DTD_DAY, ic.PL_COMPID, ic.component, ic.UND_MATURITY, ic.tenor
        ),
        sensi_key AS (
        SELECT DISTINCT DAY, DTD_DAY, nodenbr, product_type, underlying, pl_compid, component, tenor, UND_MATURITY FROM prevail_by_comp
        ),
        incept_prevail_by_comp AS (
        SELECT k.DAY as Day, k.DTD_DAY as DtdDay, n.nodeid as Portfolio, k.pl_compid as Component, k.component as RawComponent, k.product_type as ProductType,
        k.underlying as Underlying, k.tenor as Tenor, k.UND_MATURITY as UndMaturity,
        prevail.dtd_local as PrevailFSLcy, incept.dtd_local as InceptFSLcy,
        (coalesce(prevail.dtd_local, 0) - coalesce(incept.dtd_local, 0)) as DiffInLcy, (coalesce(prevail.dtd_sgd, 0) - coalesce(incept.dtd_sgd, 0)) as DiffInSgd,
        prevail.T_mkt, prevail.T_1_mkt, prevail.value as PrevailRisk, incept.value as InceptRisk, (coalesce(prevail.value, 0) - coalesce(incept.value, 0)) as RiskDiff
        FROM sensi_key k
        inner JOIN tbl_node n on k.nodenbr = n.nodenbr
        LEFT JOIN incept_by_comp incept ON k.nodenbr = incept.nodenbr and k.DAY = incept.DAY and k.DTD_DAY = incept.DTD_DAY AND k.pl_compid = incept.pl_compid ANd k.component = incept.component AND k.product_type = incept.product_type
        AND k.underlying = incept.underlying AND k.tenor = incept.tenor AND k.UND_MATURITY = incept.UND_MATURITY
        LEFT JOIN prevail_by_comp prevail ON k.nodenbr = prevail.nodenbr and k.DAY = prevail.DAY and k.DTD_DAY = prevail.DTD_DAY AND k.pl_compid = prevail.pl_compid AND k.component = prevail.component AND k.product_type = prevail.product_type
        AND k.underlying = prevail.underlying AND k.tenor = prevail.tenor AND k.UND_MATURITY = prevail.UND_MATURITY
        ) SELECT * FROM incept_prevail_by_comp
        """

        df_fsoci = self._db_obj.get_dataframe(query)
        if not df_fsoci.empty:
            return df_fsoci
        else:
            return None

