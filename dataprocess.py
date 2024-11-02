import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Group data by site
# Sites are characterised by a unique (lon, lat) combination. To group sites together, 
# we can firstly create a new column with the combined longitude and latitude.
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# We can now perform groupby operations, e.g. computing mean values across all models
data = data.groupby('lon_lat').mean().drop(columns='model')

# Display a summary
data

###Fig 1###
#A map showing variability in model predictions across the 12 configurations (e.g. where the configurations closely agree, and where they differ).


###Fig 2###



###Fig 3###

