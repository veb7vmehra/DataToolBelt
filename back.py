# hackathon T - Hacks 3.0
# flask backend of data-cleaning website
import matplotlib.pyplot as plt
import pandas as pd

def feature_pie(filename, feature1, feature2, class_size = 10):
    df = pd.read_csv (filename)
    sums = df.groupby(df[feature1])[feature2].sum()
    plt.axis('equal')
    explode = (0.1, 0, 0, 0, 0)  
    plt.pie(sums, labels=sums.index, explode = explode, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Pie chart on basis of "+feature1)
    plt.show()

def new_feature(filename, feature1, feature2, ):
