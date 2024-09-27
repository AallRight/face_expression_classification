import pandas as pd
import os
import csv
savefile_name = "data.csv"
savepath = "."
df = pd.read_csv("datafornone.csv");
df.to_csv(savepath+'\\'+ savefile_name,encoding="utf_8_sig",index=False)
df = pd.read_csv("dataforopen.csv");
df.to_csv(savepath+'\\'+ savefile_name,encoding="utf_8_sig",index=False,header = False, mode = 'a+')
df = pd.read_csv("dataforsmile.csv");
df.to_csv(savepath+'\\'+ savefile_name,encoding="utf_8_sig",index=False,header = False, mode = 'a+')
df = pd.read_csv("dataforpouting.csv");
df.to_csv(savepath+'\\'+ savefile_name,encoding="utf_8_sig",index=False,header = False, mode = 'a+')