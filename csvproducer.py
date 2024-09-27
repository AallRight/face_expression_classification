import csv
dir = "2smile/"
fix = "smile.jpg"
with open('dataforsmile.csv','w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for i in range(1, 4842):
        num = str(i)
        address = dir + num + fix
        label = 2
        row_list = [address, label]
        writer.writerow(row_list);
print("success")
