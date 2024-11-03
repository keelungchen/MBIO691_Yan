##########################################
# Fig. 3                                 #
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
output_path = os.path.join(output_dir, "Fig3.svg")

# 儲存圖形，提升解析度和圖像品質
plt.savefig(output_path, dpi=400, format='svg', bbox_inches='tight', transparent=True)
