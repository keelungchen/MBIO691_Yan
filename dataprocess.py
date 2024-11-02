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

##########################################
# Fig. 1                                 #
##########################################
#A map showing variability in model predictions across the 12 configurations (e.g. where the configurations closely agree, and where they differ).
#Laod map library
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# 計算每個地點的各模型配置的均值和標準差
data_mean = data.groupby('lon_lat').mean()
data_std = data.groupby('lon_lat').std()

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
ax.add_feature(cfeature.COASTLINE, linewidth=0.1)
ax.add_feature(cfeature.LAND, facecolor='#b6cbcf', edgecolor='#57868f', linewidth=0.1)

# 添加經緯度線
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="-", color='#57868f', alpha=0.2)
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
    cmap='Reds',
    s=5,
    transform=ccrs.PlateCarree()
)

# 加入顏色條並放置於底部
cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05)
cbar.set_label('Variability in Coral Cover Predictions (km²)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 設定標題並顯示地圖
plt.title("Variability in Coral Cover Predictions across Configurations (2100)", fontsize=14)


#############
## 計算每個地點的各模型配置的均值和標準差
data_mean = data.groupby('lon_lat').mean()
data_std = data.groupby('lon_lat').std()

# 將經緯度重新拆分為單獨的欄位
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
titles = ["Variability in Coral Cover Predictions (2020)", "Variability in Coral Cover Predictions (2100)"]

# 計算所有變異性的最大最小值，確保顏色刻度一致
vmin = min(data_values[0].min(), data_values[1].min())
vmax = max(data_values[0].max(), data_values[1].max())

# 使用迴圈來繪製子圖
for i in range(2):
    # 創建子圖並設定投影
    ax = f.add_subplot(gs[i, 0], projection=ccrs.Robinson(central_longitude=central_lon))
    ax.set_extent([central_lon - width, central_lon + width, central_lat - height, central_lat + height], ccrs.PlateCarree())

    # 加入海岸線和陸地特徵
    ax.add_feature(cfeature.COASTLINE, linewidth=0.1)
    ax.add_feature(cfeature.LAND, facecolor='#b6cbcf', edgecolor='#57868f', linewidth=0.1)

    # 添加經緯度線
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="-", color='#57868f', alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 210, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 100, 30))
    gl.xlabel_style = {'size': 10, 'color': '#57868f'}
    gl.ylabel_style = {'size': 10, 'color': '#57868f'}

    # 繪製變異性資料點，設置相同的 vmin 和 vmax 來保持顏色一致
    scatter = ax.scatter(
        data_std['longitude'],
        data_std['latitude'],
        c=data_values[i],
        cmap=colormaps[i],
        s=5,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    # 設置子圖標題
    ax.set_title(titles[i], fontsize=14)

# 添加統一的顏色條，位於右側
cax = f.add_subplot(gs[:, 1])  # 使用 GridSpec 的右側欄位作為顏色條的位置
cbar = plt.colorbar(scatter, cax=cax, orientation='vertical')

###Fig 2###



###Fig 3###

