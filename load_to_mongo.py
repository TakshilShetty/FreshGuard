"""Load a CSV into MongoDB. Usage: python load_to_mongo.py data/demo/demo_5day_3bananas.csv"""
import sys

import pandas as pd
from pymongo import MongoClient

df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
col = MongoClient("mongodb://localhost:27017")["freshguard"]["readings"]
col.drop()  # ponytail: reload replaces the collection; use upserts if you need to append
col.insert_many(df.to_dict("records"))
print(f"{col.count_documents({})} documents in freshguard.readings")
