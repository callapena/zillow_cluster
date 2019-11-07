import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prepare import prep_zillow

zillow = pd.read_csv('zillow.csv')
zillow = prep_zillow(zillow)