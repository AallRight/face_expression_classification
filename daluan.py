import pandas as pd
import os
# from sklearn.utils import shuffle
import csv
import random

# data = pd.read_csv("data.csv")
# data = shuffle(data)
# data.to_csv("data_shuffle.csv")
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [row for row in reader]

random.shuffle(rows)

with open("data_shuffled.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)