"""散乱因子の数値テーブル(tools/gen_scattering_tables.py で生成)。"""

from kaisetsu.scattering.tables.it1992 import IT1992
from kaisetsu.scattering.tables.sears1992 import SEARS1992
from kaisetsu.scattering.tables.wk1995 import WK1995

__all__ = ["IT1992", "SEARS1992", "WK1995"]
