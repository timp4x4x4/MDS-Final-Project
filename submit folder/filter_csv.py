import pandas as pd

df = pd.read_csv("US_Accidents_March23.csv/US_Accidents_March23.csv")

df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')

df_2022 = df[df['Start_Time'].dt.year == 2022]

df_2022.to_csv("US_Accidents_2022.csv", index=False)

print(f"篩選出 {len(df_2022)} 筆 2022 年事故資料，並已儲存到 US_Accidents_2022.csv。")
