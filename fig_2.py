##########################################
# Fig. 2                                 #
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
plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout=True 自動調整佈局

# 設置主標題
plt.suptitle("Relationships between Variables and Coral Cover Change", fontsize=14, fontweight='bold')

# 定義要繪製的變數列表
variables = ['SST_change', 'SST_seasonal', 'pH_change', 'PAR']
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
    
    # 添加 R^2 值於子圖內部
    plt.text(0.05, 0.9, f'$R^2 = {r_value**2:.2f}$', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # 顯示相關性係數
    plt.xlabel(var)  # 設置 X 軸標籤
    plt.ylabel('Coral Cover Change (%)')  # 設置 Y 軸標籤


# 保存圖像到 output 資料夾，設定 dpi=500
output_dir = "output"
output_path = os.path.join(output_dir, "Fig2.png")
plt.savefig(output_path, dpi=180)