import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import pandas as pd
import os

# Load the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Group data by site
# Sites are characterised by a unique (lon, lat) combination. To group sites together, 
# we can firstly create a new column with the combined longitude and latitude.
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# Display a summary
#data

##########################################
# Fig. 1                                 #
##########################################
#A map showing variability in model predictions across the 12 configurations (e.g. where the configurations closely agree, and where they differ).
#Laod map library
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# Calculate the mean and std of models in each lon_lat
data_mean = data.groupby('lon_lat').mean()
data_std = data.groupby('lon_lat').std()

# sepearte lon_lat into depentdent column again for map plot
data_mean[['longitude', 'latitude']] = pd.DataFrame(data_mean.index.tolist(), index=data_mean.index)
data_std[['longitude', 'latitude']] = pd.DataFrame(data_std.index.tolist(), index=data_std.index)

# Set the central longitude and visualization range
central_lon = 170
central_lat = 0
width = 140
height = 45

# Create the figure using GridSpec layout, divided into 2 rows and 2 columns, with the right side for the color bar
f = plt.figure(constrained_layout=True, figsize=(13.8, 10))
gs = GridSpec(2, 2, figure=f, width_ratios=[1, 0.05])

# Define data, color mapping, and labels
data_values = [data_std['coral_cover_2020'], data_std['coral_cover_2100']]
colormaps = ['Reds', 'Reds']
titles = ["2020", "2100"]

# Calculate the minimum and maximum of all variability values to ensure a consistent color scale
vmin = min(data_values[0].min(), data_values[1].min())
vmax = max(data_values[0].max(), data_values[1].max())

# Use a loop to draw subplots
for i in range(2):
    # Create subplot and set projection
    ax = f.add_subplot(gs[i, 0], projection=ccrs.Robinson(central_longitude=central_lon))
    ax.set_extent([central_lon - width, central_lon + width, central_lat - height, central_lat + height], ccrs.PlateCarree())

    # Add coastline and land features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#b6cbcf', edgecolor='#57868f', linewidth=0.5)

    # Add latitude and longitude lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="-", color='#57868f', alpha=0.2, linewidth=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 210, 30)) # Sets fixed intervals for longitude gridlines every 30 degrees, starting from -180 up to 210.
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 100, 30))
    gl.xlabel_style = {'size': 16, 'color': '#57868f'}
    gl.ylabel_style = {'size': 16, 'color': '#57868f'}

    # Plot variability data points, setting the same vmin and vmax to keep colors consistent
    # Use hexbin to show density
    hexbin = ax.hexbin(
        data_std['longitude'],  # Longitude values for each data point
        data_std['latitude'],  # Latitude values for each data point
        C=data_values[i], # The value associated with each data point
        gridsize=100,  # Adjust grid size, larger smaller
        cmap=colormaps[i], # Specifies the color maps
        mincnt=1,  # Show only cells with at least 1 point
        vmin=vmin, # Sets the minimum value for color mapping
        vmax=vmax, # Sets the maximum value for color mapping
        reduce_C_function=np.mean,  # Specifies that values in each hexagon should be averaged rather than summed
        transform=ccrs.PlateCarree(), #coordinates are in Plate Carree projection
        alpha=1
    )

    # Display the year title at the bottom left of the map
    ax.text(
        0.05, 0.1, titles[i], 
        transform=ax.transAxes, 
        fontsize=24, 
        color='#57868f', 
        fontweight='bold',
        ha='left', 
        va='top'
    )

# Add the overall title for the figure
f.suptitle("Variability in Coral Cover Predictions across Configurations", fontsize=24, fontweight='bold')

# Add a unified color bar on the right side
cax = f.add_subplot(gs[:, 1])  # Use the right column in GridSpec for the color bar location
cbar = plt.colorbar(hexbin, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Standard Deviation", fontsize=16)

# Save the image to the output folder with dpi= 300
output_dir = "output"
output_path = os.path.join(output_dir, "Fig1.png")
plt.savefig(output_path, dpi=300)