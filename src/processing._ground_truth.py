import os
import pandas as pd

from src.utils import get_base_dir

dataset = "dooway"
region = "EmiliaRomagna" #Basilicata
base_dir = os.path.join(get_base_dir(), dataset, region)

#df = pd.read_csv(os.path.join(base_dir, "reviews_output_llm_basilicata_50_rev_days.csv"))
df = pd.read_csv(os.path.join(base_dir, "reviews_output_llm_emilia_romagna_50_rev_days.csv"))
print()