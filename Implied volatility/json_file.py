import csv
import json

option_prices = "option_prices"
hist_prices = "hist_prices"


# Storing option prices in a json file

with open(option_prices+".json", mode='w', encoding='utf-8') as f:
    json.dump([], f)

with open(option_prices+".json", mode='r', encoding='utf-8') as f:
    data = json.load(f)

with open(option_prices+'.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for row in spamreader:
        entry = {'strike': float(row[3]), 'price': float(row[2])}
        data.append(entry)

with open(option_prices+".json", mode='w', encoding='utf-8') as f:
    json.dump(data,f)



# Storing historical prices of the S&P500

with open(hist_prices+".json", mode='w', encoding='utf-8') as f:
    json.dump([], f)

with open(hist_prices+".json", mode='r', encoding='utf-8') as f:
    data = json.load(f)

with open(hist_prices+'.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        entry = {'date': row[0], 'adj_close': row[5]}
        data.append(entry)

with open(hist_prices+".json", mode='w', encoding='utf-8') as f:
    json.dump(data,f)
