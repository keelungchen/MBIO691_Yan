import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import seaborn as sns
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
        transform=ccrs.PlateCarree(),
        edgecolor='none', 
        alpha=0.5
    )
    # 在地圖左下角顯示年份標題
    ax.text(
        0.05, 0.1, titles[i], 
        transform=ax.transAxes, 
        fontsize=14, 
        color='#57868f', 
        fontweight='bold',
        ha='left', 
        va='top'
    )

# 添加圖表的總標題
f.suptitle("Variability in Coral Cover Predictions across Configurations", fontsize=12, fontweight='bold')

# 添加統一的顏色條，位於右側
cax = f.add_subplot(gs[:, 1])  # 使用 GridSpec 的右側欄位作為顏色條的位置
cbar = plt.colorbar(scatter, cax=cax, orientation='vertical')
cbar.set_label("Standard Deviation", fontsize=12)

# 保存圖像到 output 資料夾，設定 dpi=400
output_dir = "output"
output_path = os.path.join(output_dir, "variability_coral_cover_predictions.png")
plt.savefig(output_path, dpi=400)

# 圖說 地理範圍、年分意義、點的大小、顏色意義 越濃差異越大、投影坐標

###Fig 2###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 載入數據
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Group data by site
# Sites are characterised by a unique (lon, lat) combination. To group sites together, 
# we can firstly create a new column with the combined longitude and latitude.
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# 根據 'lon_lat' 分組並計算平均值
data_mean = data.groupby('lon_lat').mean()

# 計算珊瑚覆蓋率變化百分比
data_mean['coral_cover_change'] = ((data_mean['coral_cover_2100'] - data_mean['coral_cover_2020']) / data_mean['coral_cover_2020']) * 100

# 計算 SST 和 pH 變化
data_mean['SST_change'] = data_mean['SST_2100'] - data_mean['SST_2020']
data_mean['pH_change'] = data_mean['pH_2100'] - data_mean['pH_2020']

# 移除異常值 (範圍設為 -100% 到 100%)
filtered_data = data_mean[(data_mean['coral_cover_change'] >= -100) & (data_mean['coral_cover_change'] <= 100)]

# 6. 計算 'coral_cover_change' 的 99% 分位數
coral_cover_99th_percentile = filtered_data['coral_cover_change'].quantile(0.99)

# 7. 過濾掉超過 99% 分位數的離群值
filtered_data = filtered_data[filtered_data['coral_cover_change'] <= coral_cover_99th_percentile]

# Classify SST_seasonal into categories
filtered_data['SST_seasonal_class'] = pd.cut(
    filtered_data['SST_seasonal'],
    bins=[-np.inf, 0.5, 1, np.inf],
    labels=['<=0.5 std', '0.5-1 std', '>1 std']
)

# Define marker styles for each class
marker_styles = {
    '<=0.5 std': 'o',  # circle
    '0.5-1 std': 's',  # square
    '>1 std': 'D'      # diamond
}

# 清除當前圖形
plt.figure()  # 開啟新圖形，避免重疊
# Plot scatter plot with different markers for SST_seasonal classes
for sst_class, marker in marker_styles.items():
    subset = filtered_data[filtered_data['SST_seasonal_class'] == sst_class]
    plt.scatter(
        subset['SST_change'], 
        subset['coral_cover_change'], 
        c=subset['pH_change'], 
        s=subset['SST_seasonal'],  # Scale size for visual effect
        cmap='BuPu_r', 
        alpha=0.7,
        marker=marker,
        edgecolor='none',
        label=sst_class
    )


# 保存圖像到 output 資料夾，設定 dpi=400
output_dir = "output"
output_path = os.path.join(output_dir, "Fig2.png")
plt.savefig(output_path, dpi=400)

scatter = plt.scatter(
    filtered_data['SST_change'], 
    filtered_data['coral_cover_change'], 
    c=filtered_data['pH_change'], 
    marker=filtered_data['SST_seasonal_class'],  # 調整大小以增強視覺效果
    cmap='BuPu_r', alpha=0.7
)


# 使用 scatter 畫散佈圖，設定:
# X 軸為 SST_change，Y 軸為 coral_cover_change
# 點的顏色依據 pH_change，並使用 'viridis' 色譜（由低到高顏色變化）
# 點的大小依據 SST_seasonal（乘以 10 是為了增加視覺效果）
scatter = plt.scatter(
    filtered_data['SST_change'], 
    filtered_data['coral_cover_change'], 
    c=filtered_data['pH_change'], 
    s=filtered_data['SST_seasonal'],  # 調整大小以增強視覺效果
    cmap='BuPu_r', alpha=0.7
)

# 加入顏色條，用於表示 pH_change 的數值範圍
plt.colorbar(scatter, label='pH change')

# 設定圖標題及軸標籤
plt.xlabel('SST change (degrees C)')
plt.ylabel('Coral cover change (%)')
plt.title('珊瑚覆蓋率變化與 SST 變化的關係\n(以 pH 變化為顏色, SST 季節變化為點大小)')




###Fig 3###

