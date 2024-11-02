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

# Display a summary
#data

###Fig 1###
#A map showing variability in model predictions across the 12 configurations (e.g. where the configurations closely agree, and where they differ).
#Laod map library
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 計算每個地點的各模型配置的均值和標準差
# 透過 groupby 進行分組，並計算所有模型的均值和標準差
data_mean = data.groupby('lon_lat').mean().drop(columns='model')
data_std = data.groupby('lon_lat').std().drop(columns='model')

# 將經緯度重新拆分為單獨的欄位
data_mean[['longitude', 'latitude']] = pd.DataFrame(data_mean.index.tolist(), index=data_mean.index)
data_std[['longitude', 'latitude']] = pd.DataFrame(data_std.index.tolist(), index=data_std.index)

# 創建等距圓柱投影，設置中心經度為180度
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

# 加入海岸線和國界
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# 繪製變異性（例如 coral_cover_2100 的標準差）
variability = data_std['coral_cover_2100'].values
scatter = ax.scatter(
    data_std['longitude'],
    data_std['latitude'],
    c=variability,
    cmap='coolwarm',
    s=10,
    transform=ccrs.PlateCarree()
)

# 加入顏色條並設定在下方顯示
cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05)  # 使用水平顯示並調整距離
cbar.set_label('Variability in Coral Cover Predictions (km²)')
plt.show()

###Fig 2###



###Fig 3###

