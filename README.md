# NTT2020

This is only a part of the PRIVATE repo at https://github.com/snazzybloke/NTT2020,
which shows the work I carried out over 2020 while at DBS Bank, Singapore.
The files legally belong to the Bank, even though every single line of those
was written by me only.
The common files (to which I contributed with my team mates) are not included, because
I was not the only contributor to those.
Similarly, I cannot provide any input (i.e., Bank's trading data) to demonstrate the use
of the provided files because I left all the data with the Bank (for obvious reasons),

The directory tree shows following files:
```bash
├── CT
│   └── ct_red.py
├── EffectivePC.pdf
├── FXO
│   ├── common_futils_lcy.py
│   ├── enquire5.py
│   └── fxo_xl.py
├── FXO_example.pdf
├── IRD
│   └── irvega_xl.py
├── IRD_example.pdf
└── README.md
```
The 3 subfolders CT, FXO, and IRD comprise the Python files that implement the
process known as "Product Control PnL Commentary"  (PnL = Profit and Loss).
Namely, every financial institution has to have this process in place, typically
carried out by human controllers.
The value of day-to-day trades (carried out by the Traders) is subject to
variations (there are many variables that can affect the value, e.g., interest
rate changes, FX pair rate changes, time effects on options etc...).
Hence, the PnL amounts considered here are not REALIZED profits/losses, but
rather "paper" profits/losses due to the variations, but they still affect the
Bank's balance sheet.
The accounting/reporting/risk management technique to properly attribute those
profits/losses to their causes is the so-called "sensitivities method"
(see https://en.wikipedia.org/wiki/PnL_Explained).
The sensitivities method has been implemented here for three DBS trading desks:  
(1) "Credit Trading" (see the CT/ct_red.py file)  
(2) "FX Options" (see the FXO/fxo_xl.py file)  
(3) "Interest Rate Derivatives" (see the IRD/ird_xl.py file).  
I also wrote the two common files, used by all three trading desks:  
(4) FXO/common_futils_lcy.py, which contains the common functions;  
(5) FXO/enquire5.py, which contains a class for connecting to the iWork database,
                 and another class to extract the data from iWork via the SQL
                 queries shown in the file (I have removed the correct iWork
                 credentials, for obvious reasons).

The PnL process is usually executed by the common start.py script (not provided),
which should sit in this directory and for the given trading desk, date, and
business entity (only DBSSG for Singapore currently implemented).
Alternatively, one may simply copy one of the three trading desk's Python
files to this folder and run it by providing the required command line
arguments (the date or, optionally, file location, if the tables have been
downloaded from the iWork database).
(Note: one can safely comment out the four lines at the start  
from config.configs import getLogger  
from config import configs  
env = configs.env  
logger = getLogger("....py")  
because the common logger is not provided here)


The "Credit Trading" is concerned mainly with the components known as
"IR Delta" (IR = interest rate, for Yield Curves) and "Credit Delta" (for Bonds).
The file produces the so-called "PnL Commentary" as a plain text
(i.e., not the whole Excel document, like "FX Options" and "Interest Rate
Derivatives", because the addition of Excel output was assigned to a Junior Data
Scientist to complete, while I moved on to the "FX Options" desk.

The "FX Options" is concerned mainly with the components known as FX Delta,
FX Gamma, FX Smiles and FX Vega (see Investopedia if interested), as well as
"New deals". The file produces an Excel document with a number of sheets.
An example is provided in the PDF file FXO_example.pdf.
The first sheet is "Workflow", and is always the same, but is shown as
a reminder to a human controller, if they wish to check the output.
It is advisable to read the "Workflow" to understand the work being done:
the subsequent pages shows the breakdowns for each component.
The selected rows from each component are highlighted in "yellow"
(and also "orange" for column selection).
The final Excel worksheet, "Summary", shows the summary of the process and the
generated "Commentary". The commentary needs to be filed with the monetary
authority of the jurisdiction (MAS in Singapore, while in Australia it may be RBA or APRA).

The "Interest Rate Derivatives" is concerned with the Caplets and Swaptions
(see Investopedia if interested) for DBS arising from Minisys and MUREX
portfolios (nodes).
The only component of interest is the so-called IR Vega. For almost all
trading days I have seen the only contributing currency ("CCY" or "ccy") is the USD.
However, I have deliberately made provisions to make sure that any other currency
contribution (even if zero $) does not crash the execution.
This has, regrettably, contributed to the extra cognitive complexity.
An example of the Excel report is provided in the PDF file IRD_example.pdf.
Again, the first sheet is "Workflow", and is always the same.
The rest shows the breakdowns and row selections, first for the Caplets,
then for Swaptions. The final Excel worksheet, "Summary", shows the summary
of the process and the generated "Commentary".

I am not sharing the code documentation.
If interested, one may extract all the docstrings using sphinx of pydoc.

Finally, if interested, one can learn more about the Product Control process
and PnL attribution from the recent book by P. Nash, "Effective Product Control",
Wiley (2018). The first ten pages (showing the Contents) are provided in the
PDF file EffectivePC.pdf. I am unable to share the whole book (for obvious reasons).
