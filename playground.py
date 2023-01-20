import csv
from tqdm import tqdm
from time import sleep

with open("train.csv", 'r') as f:
    reader = csv.DictReader(f)
    reader = tqdm(reader,total=223415, unit="pics")
    for row in reader:
        continue