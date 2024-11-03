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

# 設置中心經度和可視化範圍
central_lon = 170
central_lat = 0
width = 140
height = 45

# 創建圖表，使用 GridSpec 佈局，將圖表分為 2 行 2 列，右側為顏色條
f = plt.figure(constrained_layout=True, figsize=(13.8, 10))
gs = GridSpec(2, 2, figure=f, width_ratios=[1, 0.05])

# 定義資料、顏色映射和標籤
data_values = [data_std['coral_cover_2020'], data_std['coral_cover_2100']]
colormaps = ['Reds', 'Reds']
titles = ["2020", "2100"]

# 計算所有變異性的最大最小值，確保顏色刻度一致
vmin = min(data_values[0].min(), data_values[1].min())
vmax = max(data_values[0].max(), data_values[1].max())

# 使用迴圈來繪製子圖
for i in range(2):
    # 創建子圖並設定投影
    ax = f.add_subplot(gs[i, 0], projection=ccrs.Robinson(central_longitude=central_lon))
    ax.set_extent([central_lon - width, central_lon + width, central_lat - height, central_lat + height], ccrs.PlateCarree())

    # 加入海岸線和陸地特徵
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#b6cbcf', edgecolor='#57868f', linewidth=0.5)

    # 添加經緯度線
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="-", color='#57868f', alpha=0.2, linewidth=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 210, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 100, 30))
    gl.xlabel_style = {'size': 16, 'color': '#57868f'}
    gl.ylabel_style = {'size': 16, 'color': '#57868f'}

    # 繪製變異性資料點，設置相同的 vmin 和 vmax 來保持顏色一致
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

    # 在地圖左下角顯示年份標題
    ax.text(
        0.05, 0.1, titles[i], 
        transform=ax.transAxes, 
        fontsize=24, 
        color='#57868f', 
        fontweight='bold',
        ha='left', 
        va='top'
    )

# 添加圖表的總標題
f.suptitle("Variability in Coral Cover Predictions across Configurations", fontsize=24, fontweight='bold')

# 添加統一的顏色條，位於右側
cax = f.add_subplot(gs[:, 1])  # 使用 GridSpec 的右側欄位作為顏色條的位置
cbar = plt.colorbar(hexbin, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Standard Deviation", fontsize=16)

# 保存圖像到 output 資料夾，設定 dpi=400
output_dir = "output"
output_path = os.path.join(output_dir, "fig1.png")
plt.savefig(output_path, dpi=300)

# 圖說 地理範圍、年分意義、點的大小、顏色意義 越濃差異越大、投影坐標