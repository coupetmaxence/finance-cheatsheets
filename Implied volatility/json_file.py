import csv
import json

with open("S&P500.json", mode='w', encoding='utf-8') as f:
    json.dump([], f)

with open("S&P500.json", mode='r', encoding='utf-8') as f:
    data = json.load(f)

with open('data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for row in spamreader:
        print(row)
        print(len(row))
        entry = {'strike': float(row[3]), 'price': float(row[2])}
        data.append(entry)

with open("S&P500.json", mode='w', encoding='utf-8') as f:
    json.dump(data,f)

with open("hist_S&P500.json", mode='w', encoding='utf-8') as f:
    json.dump([], f)

with open("hist_S&P500.json", mode='r', encoding='utf-8') as f:
    data = json.load(f)

with open('hist.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        print(row)
        print(len(row))
        entry = {'date': row[0], 'adj_close': row[5]}
        data.append(entry)




with open("hist_S&P500.json", mode='w', encoding='utf-8') as f:
    json.dump(data,f)
