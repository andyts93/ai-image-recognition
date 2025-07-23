""" Script per estrarre i part id per le prove con Pollini """
import pandas as pd
from config import *

df = pd.read_csv(CSV_PATH)
unique_part_ids = df['part_id'].unique()
print(*unique_part_ids, sep=',')