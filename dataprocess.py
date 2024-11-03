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

##########################################
# Fig. 2                                 #
##########################################
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
output_path = os.path.join(output_dir, "Fig2.png")
plt.savefig(output_path, dpi=400)


##########################################
# Fig. 3                                 #
##########################################
# 導入所需的套件
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# 讀取數據集
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# 創建一個新的列，表示唯一的 (longitude, latitude) 組合來標識不同的地點
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# 根據地點（lon_lat）分組並計算每個地點的變量平均值
data_mean = data.groupby('lon_lat').mean().reset_index()

# 計算珊瑚覆蓋變化的百分比 (以 2100 年相比 2020 年)
data_mean['coral_cover_change'] = ((data_mean['coral_cover_2100'] - data_mean['coral_cover_2020']) / 
                                   data_mean['coral_cover_2020']) * 100

# 計算 SST 和 pH 變化
data_mean['SST_change'] = data_mean['SST_2100'] - data_mean['SST_2020']
data_mean['pH_change'] = data_mean['pH_2100'] - data_mean['pH_2020']


# 定義函數來剔除離群值
def remove_outliers(df, columns, threshold=1.5):
    for column in columns:
        Q1 = df[column].quantile(0.25)  # 第一四分位數
        Q3 = df[column].quantile(0.75)  # 第三四分位數
        IQR = Q3 - Q1  # 計算四分位範圍
        # 使用 IQR 來定義下限和上限
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # 篩選出在上下限範圍內的數據
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# 剔除 'coral_cover_change', 'SST_2100', 'SST_seasonal', 'pH_2100', 'PAR' 列的離群值
data_mean_clean = remove_outliers(data_mean, ['coral_cover_change', 'SST_change', 'SST_seasonal', 'pH_change', 'PAR'])

# 繪製清理後的數據
plt.figure(figsize=(10, 6))

# 使用子圖繪製每個變量（SST變化, pH變化, PAR）與珊瑚覆蓋變化之間的關係
for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)  # 2x2 子圖布局
    x = data_mean_clean[var]
    y = data_mean_clean['coral_cover_change']
    
    # 繪製灰色散點
    plt.scatter(x, y, color='gray', alpha=0.6, s=0.5, edgecolor='none')
    
    # 使用線性回歸添加相關性線
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot(x, slope * x + intercept, color='black', linestyle='--', linewidth=1)
    
    # 顯示相關性係數
    plt.xlabel(var)  # 設置 X 軸標籤
    plt.ylabel('Coral Cover Change (%)')  # 設置 Y 軸標籤
    plt.title(f'{var} vs Coral Cover Change\nR² = {r_value**2:.2f}')  # 顯示 R² 值於標題


# 保存圖像到 output 資料夾，設定 dpi=400
output_dir = "output"
output_path = os.path.join(output_dir, "Fig3.png")
plt.savefig(output_path, dpi=400)

# 顯示圖表
plt.show()


##########################################
# Fig. 4                                 #
##########################################
# 導入所需的套件
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# 讀取數據集
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# 計算珊瑚覆蓋變化率 (2100年的覆蓋率 - 2020年的覆蓋率) / 2020年的覆蓋率
data['coral_cover_change'] = (data['coral_cover_2100'] - data['coral_cover_2020']) / data['coral_cover_2020'] * 100

# 過濾掉不合理的變化率 (假設合理範圍為 -100% 到 100%)
data = data[(data['coral_cover_change'] >= -100) & (data['coral_cover_change'] <= 100)]

# 定義函數來剔除離群值
def remove_outliers(df, columns, threshold=1.5):
    for column in columns:
        Q1 = df[column].quantile(0.25)  # 第一四分位數
        Q3 = df[column].quantile(0.75)  # 第三四分位數
        IQR = Q3 - Q1  # 計算四分位範圍
        # 使用 IQR 來定義下限和上限
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # 篩選出在上下限範圍內的數據
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# 剔除 'coral_cover_change', 'SST_2100', 'SST_seasonal', 'pH_2100', 'PAR' 列的離群值
data = remove_outliers(data, ['coral_cover_change'])

# 將緯度分組為 0.5 的間隔
data['latitude_bin'] = (data['latitude'] // 1) * 1

# 計算每個 1 度緯度區間和每個 model 的平均珊瑚覆蓋變化率
latitude_change = data.groupby(['latitude_bin', 'model'])['coral_cover_change'].mean().reset_index()

# 計算所有 model 的平均值
mean_change = latitude_change.groupby('latitude_bin')['coral_cover_change'].mean().reset_index()

# 設置圖形大小
plt.figure(figsize=(20, 12))
# 使用顏色映射，讓每個 model 顯示更明顯的顏色
colors = cm.get_cmap('tab20', 12)  # 使用 12 種顏色

# 為不同的 model 繪製折線圖，每個 model 使用不同顏色
for model_num in sorted(latitude_change['model'].unique()):
    model_data = latitude_change[latitude_change['model'] == model_num]
    plt.plot(model_data['latitude_bin'], model_data['coral_cover_change'], 
             label=f'Model {model_num}', linewidth=1.5, color=colors(model_num))

# 繪製平均變化率折線，使用較粗的黑線
plt.plot(mean_change['latitude_bin'], mean_change['coral_cover_change'], 
         label='Mean', linewidth=2.5, color='black')

# 添加英文標題和坐標軸標籤
plt.title('Average Coral Cover Change Rate by Latitude',fontweight='bold')
plt.xlabel('Latitude')
plt.ylabel('Average Coral Cover Change Rate (%)')

# 顯示圖例在圖外部，並按 Model 0 到 Model 11 排序
plt.legend(title="Model Number", bbox_to_anchor=(1.05, 1), loc='upper left')

output_dir = "output"
# 設定輸出路徑
output_path = os.path.join(output_dir, "Fig4.svg")

# 儲存圖形，提升解析度和圖像品質
plt.savefig(output_path, dpi=400, format='svg', bbox_inches='tight', transparent=True)
