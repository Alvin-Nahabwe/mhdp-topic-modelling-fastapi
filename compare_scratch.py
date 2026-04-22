import pandas as pd

# Load topic info
semi_info = pd.read_csv("bertopic_semi_supervised/topic_info.csv")
sup_info = pd.read_csv("bertopic_supervised/topic_info.csv")

print("--- Semi-Supervised Model ---")
print(f"Total topics: {len(semi_info)}")
has_outlier_semi = -1 in semi_info['Topic'].values
print(f"Has Outlier Topic (-1): {has_outlier_semi}")
if has_outlier_semi:
    outlier_count_semi = semi_info[semi_info['Topic'] == -1]['Count'].values[0]
    print(f"Outlier count (-1): {outlier_count_semi}")
else:
    print("Outlier count (-1): 0")

print("\n--- Supervised Model ---")
print(f"Total topics: {len(sup_info)}")
has_outlier_sup = -1 in sup_info['Topic'].values
print(f"Has Outlier Topic (-1): {has_outlier_sup}")
if has_outlier_sup:
    outlier_count_sup = sup_info[sup_info['Topic'] == -1]['Count'].values[0]
    print(f"Outlier count (-1): {outlier_count_sup}")
else:
    print("Outlier count (-1): 0")

