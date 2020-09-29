# hackathon T - Hacks 3.0
# flask backend of data-cleaning website
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import *
import os
from datetime import *
from subprocess import Popen, PIPE
from math import floor
import converter as con
from flask_ngrok import run_with_ngrok

def feature_pie(filename, feature1, feature2, class_size = 10):
    df = pd.read_csv (filename)
    sums = df.groupby(df[feature1])[feature2].sum()
    plt.axis('equal')
    explode = (0.1, 0, 0, 0, 0)  
    plt.pie(sums, labels=sums.index, explode = explode, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Pie chart on basis of "+feature1)
    plt.show()

def new_feature(filename, feature1, feature2, ):
    #function to create feature
    pass

def disp(filename):
    df = pd.read_csv(filename)
    n_row = str(len(df) + 1)
    n_col = str(len(df.axes[1]) + 1)
    col = []
    for c in df.columns:
        col.append(c)
    types = df.dtypes.tolist()
    f = open(filename, "r+")
    line0 = f.readline()
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()
    line4 = f.readline()
    line5 = f.readline()
    f.close()
    return n_row, n_col, col, types, line0, line1, line2, line3, line4, line5

def stat(filename, feature, func):
    df = pd.read_csv(filename)
    if func == "mean":
        ans = df[feature].mean()
    if func == "mx":
        ans = df[feature].max()
    if func == "mn":
        ans = df[feature].min()
    if func == "sm":
        ans = df[feature].sum()
    return ans

app = Flask(__name__)

#app.secret_key = 'maidoublequotesmelikhrhahu'

#run_with_ngrok(app)
@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        if request.files['file'].filename != '':
            f = request.files.get('file')
            varrr = "static/"+f.filename
            err=f.save(varrr)
            name = f.filename.split('.')
            ext = name[-1]
            name = name[0]
            if ext == "json":
                con.jsontocsv("static/"+f.filename, "static/"+name+".csv")
            elif ext == "xml":
                con.xmltocsv("static/"+f.filename, "static/"+name+".csv")
            n_row, n_col, col, types, line0, line1, line2, line3, line4, line5 = disp("static/"+f.filename)
            lists = []
            line0 = line0.split(',')
            lists.append(line0)
            line1 = line1.split(',')
            lists.append(line1)
            line2 = line2.split(',')
            lists.append(line2)
            line3 = line3.split(',')
            lists.append(line3)
            line4 = line4.split(',')
            lists.append(line4)
            line5 = line5.split(',')
            lists.append(line5)
            print(lists)
            return render_template("filedata.html", n_row = n_row, n_col = n_col, col = col, types = types, lists = "../static/"+name+".csv", name = f.filename)
    return render_template("upload.html")

@app.route('/stat', methods = ['GET', 'POST'])
def stats():
    if request.method == 'POST':
        filename = request.form['filename']
        name = filename('.')
        ext = name[-1]
        name = name[0]
        if ext == "json":
            con.jsontocsv("static/"+filename, "static/"+name+".csv")
        elif ext == "xml":
            con.xmltocsv("static/"+filename, "static/"+name+".csv")
        feature = request.form['feature']
        func = request.form['func']
        ans = stat(filename, feature, func)
        return render_template("filedata.html", filename = filename, feature = feature, func = func, ans = ans)

@app.route('/con', methods = ['GET', 'POST'])
def conv():
    if request.method == 'POST':
        filename = request.form['filename']
        name = filename.split('.')
        ext = name[-1]
        name = name[0]
        to = filename['to']
        if ext == "csv":
            if to == "json":
                con.csvtojson("static/"+filename, "static/"+name+"."+to)
            elif to == "xml":
                con.csvtoxml("static/"+filename, "static/"+name+"."+to)
        elif ext == "json":
            if to == "csv":
                con.jsontocsv("static/"+filename, "static/"+name+"."+to)
            elif to == "xml":
                con.jsontoxml("static/"+filename, "static/"+name+"."+to)
        elif ext == "xml":
            if to == "json":
                con.xmltojson("static/"+filename, "static/"+name+"."+to)
            elif to == "csv":
                con.xmltocsv("static/"+filename, "static/"+name+"."+to)

if __name__ == '__main__':
    app.run(debug=True)

