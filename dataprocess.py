import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Group data by site
# Sites are characterised by a unique (lon, lat) combination. To group sites together, 
# we can firstly create a new column with the combined longitude and latitude.
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# Display a summary
#data

###Fig 1###
#A map showing variability in model predictions across the 12 configurations (e.g. where the configurations closely agree, and where they differ).
#Laod map library
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 計算每個地點的各模型配置的均值和標準差
data['lon_lat'] = list(zip(data.longitude, data.latitude))
data_mean = data.groupby('lon_lat').mean().drop(columns='model')
data_std = data.groupby('lon_lat').std().drop(columns='model')

# 將經緯度重新拆分為單獨的欄位
data_mean[['longitude', 'latitude']] = pd.DataFrame(data_mean.index.tolist(), index=data_mean.index)
data_std[['longitude', 'latitude']] = pd.DataFrame(data_std.index.tolist(), index=data_std.index)

# 設置中心經度和可視化範圍
central_lon = 170
central_lat = 0
width = 140
height = 45

# 創建圖表，設定等距圓柱投影和子圖布局
f = plt.figure(constrained_layout=True, figsize=(13.8, 5))
ax = f.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=central_lon))

# 設置地圖顯示範圍
ax.set_extent([central_lon - width, central_lon + width, central_lat - height, central_lat + height], ccrs.PlateCarree())

# 加入地圖的海岸線和陸地特徵
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='#b6cbcf', edgecolor='#57868f', linewidth=0.5)

# 添加經緯度線
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.top_labels = False  # 移除頂部的緯度標籤
gl.right_labels = False  # 移除右側的經度標籤
gl.xlocator = mticker.FixedLocator(np.arange(-180, 210, 30))  # 設定經度間隔
gl.ylocator = mticker.FixedLocator(np.arange(-90, 100, 30))   # 設定緯度間隔
gl.xlabel_style = {'size': 10, 'color': '#57868f'}
gl.ylabel_style = {'size': 10, 'color': '#57868f'}

# 繪製變異性資料（以 coral_cover_2100 的標準差為例）
variability = data_std['coral_cover_2100'].values
scatter = ax.scatter(
    data_std['longitude'],
    data_std['latitude'],
    c=variability,
    cmap='coolwarm',
    s=10,
    transform=ccrs.PlateCarree(),
    edgecolor='k',
    linewidths=0.1
)

# 加入顏色條並放置於底部
cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05)
cbar.set_label('Variability in Coral Cover Predictions (km²)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 設定標題並顯示地圖
plt.title("Variability in Coral Cover Predictions across Configurations (2100)", fontsize=14)
plt.show()
#############




###Fig 2###



###Fig 3###

