import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
df = pd.read_csv("data/NF-UNSW-NB15-v3.csv")
df.head()
