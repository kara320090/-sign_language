
from pathlib import Path
import pandas as pd

path = Path("data/raw/Error_LOG_Sheet.xls")

sheets = pd.read_excel(path, sheet_name=None)

for sheet_name, df in sheets.items():
    print("\n==============================")
    print("SHEET:", sheet_name)
    print("COLUMNS:", list(df.columns))
    print(df.head(10))