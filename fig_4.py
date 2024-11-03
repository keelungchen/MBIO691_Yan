##########################################
# Fig. 4                                 #
##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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


plt.figure()  # 開啟新圖形，避免重疊
# 建立子圖形區域，將圖分為 1 行 2 列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# 左邊的散佈圖：SST_change vs coral_cover_change，以 pH_change 進行上色
scatter1 = ax1.scatter(
    filtered_data['SST_change'], 
    filtered_data['coral_cover_change'], 
    c=filtered_data['pH_change'], 
    s=2, 
    cmap='Purples_r', 
    edgecolor='none'
)
# 在圖的下方加上顏色條，表示 pH_change 的數值範圍
cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.2)
cbar1.set_label('pH change')  # 設定顏色條標籤

# 右邊的散佈圖：SST_change vs coral_cover_change，以 SST_seasonal 進行上色
scatter2 = ax2.scatter(
    filtered_data['SST_change'], 
    filtered_data['coral_cover_change'], 
    c=filtered_data['SST_seasonal'], 
    s=2, 
    cmap='Oranges', 
    edgecolor='none'
)
# 在圖的下方加上顏色條，表示 SST_seasonal 的數值範圍
cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.2)
cbar2.set_label('SST seasonal')  # 設定顏色條標籤

# 統一設定 X 軸和 Y 軸的標籤
fig.text(0.55, 0.25,'SST change (degrees C)', ha='center')  # 設置 X 軸標籤於圖中央下方
fig.text(0, 0.5,'Coral cover change (%)', va='center', rotation='vertical')  # 設置 Y 軸標籤於圖左側中間

# 添加圖表的總標題
fig.suptitle("Coral cover change with SST, pH change and seasonal SST", fontsize=12,fontweight='bold')


# 保存圖像到 output 資料夾，設定 dpi=400
output_dir = "output"
output_path = os.path.join(output_dir, "Fig4.png")
plt.savefig(output_path, dpi=400)
