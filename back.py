# hackathon T - Hacks 3.0
# flask backend of data-cleaning website
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from flask import *
import os
from datetime import *
from subprocess import Popen, PIPE
from math import floor
import converter as con
from flask_ngrok import run_with_ngrok
from meanShift import Mean_Shift
from matplotlib import style
import seaborn as sns
style.use('ggplot')
from sklearn.model_selection import train_test_split
from datetime import datetime

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

colors = 10*['g', 'r', 'b', 'c', 'k']

from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)
import math
import operator

exprStack = []


def push_first(toks):
    exprStack.append(toks[0])


def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break


bnf = None


def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        # fnumber = Combine(Word("+-"+nums, nums) +
        #                    Optional("." + Optional(Word(nums))) +
        #                    Optional(e + Word("+-"+nums, nums)))
        # or use provided pyparsing_common.number, but convert back to str:
        # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")

        expr = Forward()
        expr_list = delimitedList(Group(expr))
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            addop[...]
            + (
                (fn_call | pi | e | fnumber | ident).setParseAction(push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        term = factor + (multop + factor).setParseAction(push_first)[...]
        expr <<= term + (addop + term).setParseAction(push_first)[...]
        bnf = expr
    return bnf


# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}

fn = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "abs": abs,
    "trunc": int,
    "round": round,
    "sgn": lambda a: -1 if a < -epsilon else 1 if a > epsilon else 0,
    # functionsl with multiple arguments
    "multiply": lambda a, b: a * b,
    "hypot": math.hypot,
    # functions with a variable number of arguments
    "all": lambda *a: all(a),
}


def evaluate_stack(s):
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op
    if op == "unary -":
        return -evaluate_stack(s)
    if op in "+-*/^":
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s) for _ in range(num_args)])
        return fn[op](*args)
    elif op[0].isalpha():
        raise Exception("invalid identifier '%s'" % op)
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return int(op)
        except ValueError:
            return float(op)

def test(s):
    val = "NA"
    exprStack[:] = []
    try:
        results = BNF().parseString(s, parseAll=True)
        val = evaluate_stack(exprStack[:])
    except ParseException as pe:
        print(s, "failed parse:", str(pe))
    except Exception as e:
        print(s, "failed eval:", str(e), exprStack)
    return val

def feature_pie(filename, feature1, feature2, class_size = 10):
    df = pd.read_csv(filename)
    sums = df.groupby(df[feature1])[feature2].sum()
    plt.axis('equal')
    plt.pie(sums, labels=sums.index, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Pie chart on basis of "+feature2)
    name = filename.split('.')
    plt.savefig(name[0]+".png")
    plt.close()

def feature_scatter(filename, feature1, feature2):
    df = pd.read_csv(filename)
    plt.axis('equal')
    plt.pie(feature1, feature2, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Scatter plot between "+feature1+" and "+feature2)
    name = filename.split('.')
    plt.savefig(name[0]+".png")
    plt.close()

def new_feature(filename, com, name):
    df = pd.read_csv(filename)
    com = com.split(',')
    formula = "_"
    temp = "_"
    for i, c in enumerate(com):
        if c == "formula":
            formula = com[i+1]
            temp = formula
    vals = []
    i = 0
    print(name)
    if name != " ":
        i = 1
    n = len(df)
    for j in range(n):
        for k, c in enumerate(com):
            if k%2 == 0:
                if c == "formula":
                    break
                formula = formula.replace(c, str(df.at[j, com[k+1]]))
        vals.append(test(formula))
        formula = temp
    col = len(df.axes[1])
    print(vals)
    df[name] = vals
    """
    if name != " ":
        df.insert(col, vals, True)
    else:
        df.insert(col, vals, True)
    """
    del df['Unnamed: 0']
    os.remove(filename)
    df.to_csv(filename) 

def disp(filename):
    df = pd.read_csv(filename)
    n_row = str(len(df))
    n_col = str(len(df.axes[1]))
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
    ans = 0
    print(filename,feature,func)
    print(df)
    if func == "mean":
        ans = df[feature].mean()
    if func == "max":
        ans = df[feature].max()
    if func == "min":
        ans = df[feature].min()
    if func == "sum":
        ans = df[feature].sum()
    return ans

def freq(filename, feature, condition):
    df = pd.read_csv(filename)
    condition = condition.split(' ')
    if condition[0] == "=":
        counts = df[feature].value_counts().to_dict()
        return counts[int(condition[1])]
    elif condition[0] == ">":
        count = 0
        df = pd.read_csv(filename)
        n = df.columns.get_loc(feature)
        for i in range(len(df)):
            if df.at[i, n] > int(condition[1]):
                count = count + 1
        return count
    elif condition[0] == "<":
        count = 0
        df = pd.read_csv(filename)
        n = df.columns.get_loc(feature)
        for i in range(len(df)):
            if df.at[i, n] < int(condition[1]):
                count = count + 1
        return count

def drop(filename, feature, condition):
    df = pd.read_csv(filename)
    condition = condition.split(' ')
    if condition[0] == "=":
        df.drop(df[df[feature] == int(condition[1])].index, inplace = True)
    elif condition[0] == ">":
        df.drop(df[df[feature] > int(condition[1])].index, inplace = True)
    elif condition[0] == "<":
        df.drop(df[df[feature] < int(condition[1])].index, inplace = True)

def ms(filename, feature1, feature2):
    name = filename.split('.')
    df = pd.read_csv(filename)
    n = df.columns.get_loc(feature1)
    mat1 = df.iloc[:, n].values
    m = df.columns.get_loc(feature2)
    mat2 = df.iloc[:, m].values
    combined = np.vstack((mat1, mat2)).T
    combined = combined.tolist()

    clf = Mean_Shift()
    clf.fit(combined)

    centroids = clf.centroids

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150, linewidths=5)

    plt.savefig("static/ms_"+name[0].split('/')[-1]+".png")
    plt.close()

def dataDivide(df, percent):
    train_df=df.sample(frac=percent,random_state=200) #random state is a seed value
    test_df=df.drop(train.index)
    return train_df, test_df

def scale(train_df, test_df, scale = 1):
    train_df["median_house_value"] /= scale_factor 
    test_df["median_house_value"] /= scale_factor
    return train_df, test_df

def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Compile the model topography into code that TensorFlow can efficiently
  # execute. Configure training to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model               

def train_model(model, df, feature, label, my_epochs, 
                my_batch_size=None, my_validation_split=0.1):
  """Feed a dataset into the model in order to train it."""

  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=my_batch_size,
                      epochs=my_epochs,
                      validation_split=my_validation_split)

  # Gather the model's trained weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the 
  # rest of history.
  epochs = history.epoch
  
  # Isolate the root mean squared error for each epoch.
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse, history.history

def plot_the_loss_curve(epochs, mae_training, mae_validation, filename):
  name = filename.split('.')
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
  plt.legend()
  
  # We're not going to plot the first epoch, since the loss on the first epoch
  # is often substantially greater than the loss for other epochs.
  merged_mae_lists = mae_training[1:] + mae_validation[1:]
  highest_loss = max(merged_mae_lists)
  lowest_loss = min(merged_mae_lists)
  delta = highest_loss - lowest_loss
  print(delta)

  top_of_y_axis = highest_loss + (delta * 0.05)
  bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.save("static/nn_"+name[0]+".png")

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
            if ext == "csv":
                con.csvtojson("static/"+f.filename, "static/"+name+".json")
                os.remove("static/"+f.filename)
                con.jsontocsv("static/"+name+".json", "static/"+f.filename)
            if ext == "json":
                con.jsontocsv("static/"+f.filename, "static/"+name+".csv")
            elif ext == "xml":
                con.xmltocsv("static/"+f.filename, "static/"+name+".csv")
            elif ext == "nc":
                con.netCDFtocsv("static/"+f.filename, "static/"+name+".csv")
            n_row, n_col, col, types, line0, line1, line2, line3, line4, line5 = disp("static/"+name+".csv")
            res = make_response(render_template("filedata.html", filename = f.filename, n_row = n_row, n_col = n_col, col = col, types = types, lists = "../static/"+name+".csv?"+str(datetime.now()), convertable=["json", "xml", "nc"]))
            res.set_cookie("filename", value=f.filename)
            return res
    return render_template("upload.html")

@app.route('/Info', methods=['GET', 'POST'])
def info():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    n_row, n_col, col, types, line0, line1, line2, line3, line4, line5 = disp("static/"+name[0]+".csv")
    return render_template("filedata.html", filename = filename, n_row = n_row, n_col = n_col, col = col, types = types, lists = "../static/"+name[0]+".csv?"+str(datetime.now()), convertable=["json", "xml", "nc"])

@app.route('/stat', methods=['GET', 'POST'])
def stats():
    if request.method == 'GET':
        filename = request.args.get('filename').split('/')[-1]
        name = filename.split('.')
        ext = name[-1]
        name = name[0]
        if ext == "json":
            con.jsontocsv("static/"+filename, "static/"+name+".csv")
        elif ext == "nc":
            con.netCDFtocsv("static/"+filename, "static/"+name+".csv")
        elif ext == "xml":
            con.xmltocsv("static/"+filename, "static/"+name+".csv")
        feature = request.args.get('feature')
        func = request.args.get('func')
        ans = stat("static/"+name+".csv", feature, func)
        print(ans,type(ans))
        return str(ans)
    return render_template("upload.html")

@app.route('/con', methods = ['GET', 'POST'])
def conv():
    if request.method == 'GET':
        filename = request.args.get('filename')
        name = filename.split('.')
        ext = name[-1]
        name = name[0]
        to = request.args.get('to')
        if ext == "csv":
            if to == "json":
                con.csvtojson("static/"+filename, "static/"+name+"."+to)
            elif to == "xml":
                con.csvtoxml("static/"+filename, "static/"+name+"."+to)
            elif to == "nc":
                con.csvtonetCDF("static/"+filename, "static/"+name+"."+to)
        elif ext == "json":
            if to == "csv":
                con.jsontocsv("static/"+filename, "static/"+name+"."+to)
            elif to == "xml":
                con.jsontoxml("static/"+filename, "static/"+name+"."+to)
            elif to == "nc":
                con.jsontonetCDF("static/"+filename, "static/"+name+"."+to)
        elif ext == "xml":
            if to == "json":
                con.xmltojson("static/"+filename, "static/"+name+"."+to)
            elif to == "csv":
                con.xmltocsv("static/"+filename, "static/"+name+"."+to)
            elif to == "nc":
                con.xmltonetCDF("static/"+filename, "static/"+name+"."+to)
        elif ext == "nc":
            if to == "json":
                con.netCDFtojson("static/"+filename, "static/"+name+"."+to)
            elif to == "csv":
                con.netCDFtocsv("static/"+filename, "static/"+name+"."+to)
            elif to == "xml":
                con.netCDFtoxml("static/"+filename, "static/"+name+"."+to)        
        return "../static/"+name+"."+to
    return render_template("upload.html")

@app.route('/analyse', methods = ['GET', 'POST'])
def analyse():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        feature1 = request.args.get('feature1')
        feature2 = request.args.get('feature2')
        if feature1 == None:
            return render_template("analysis.html", col = col)
        feature_pie("static/"+name+".csv", feature1, feature2)
        return str("../static/"+name+".png")
    return render_template("analysis.html", col = col)

@app.route('/anAdd', methods = ['GET', 'POST'])
def anAdd():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        kname = request.args.get('name')
        print(kname)
        com = request.args.get('formula')
        new_feature("static/"+filename, com, kname)
        feature1 = request.args.get('feature1')
        feature_pie("static/"+name+".csv", feature1, kname)
        return "../static/"+name+".png"

@app.route('/clean', methods = ['GET', 'POST'])
def clean():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'POST':
        feature1 = request.form['feature1']
        feature2 = request.form['feature2']
        feature_scatter("static/"+name+".csv", feature1, feature2)
        return render_template("clean.html", col = col, img = "static/"+name+".png")
    return render_template("clean.html", col = col)

@app.route('/clAdd', methods = ['GET', 'POST'])
def clAdd():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        kname = request.form['name']
        com = request.form['formula']
        new_feature("static/"+name+".csv", com, kname)
        feature_scatter("static/"+name+".csv", feature1, kname)
        return "../static/"+name+".png"

@app.route('/freq', methods = ['GET', 'POST'])
def fre():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        feature = request.args.get('feature')
        cond = request.args.get('cond')
        freqq = freq('static/'+name+".csv", feature, cond)
        return freqq
    return render_template("clean.html", col = col)

@app.route('/drop', methods = ['GET', 'POST'])
def dro():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        feature = request.args.get('feature')
        cond = request.args.get('cond')
        drop(filename, feature, cond)
        return
    return render_template("clean.html", col = col)

@app.route('/ms', methods = ['GET', 'POST'])
def mShift():
    filename = request.cookies.get('filename')
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    if request.method == 'GET':
        feature1 = request.args.get('feature1')
        feature2 = request.args.get('feature2')
        if feature1 == None:
            return render_template("meanShift.html", filename = filename, col = col)
        ms('static/'+filename, feature1, feature2)
        name = filename.split('.')
        return "../static/ms_"+name[0]+".png"
    return render_template("meanShift.html", filename = filename, col = col)

@app.route('/nn', methods = ['GET', 'POST'])
def neural():
    name = filename.split('.')
    name = name[0]
    df = pd.read_csv("static/"+name+".csv")
    col = []
    for c in df.columns:
        col.append(c)
    filename = request.cookies.get('filename')
    if request.method == 'GET':
        percent = request.args.get('percent')
        scale = request.args.get('scale')
        df = pd.read_csv("static/"+filename)
        train_df, test_df = dataDivide(df, percent)
        scale(train_df, test_df, scale)
        learning_rate = request.args.get('learning_rate')
        epochs = request.args.get('epochs')
        batch_size = request.args.get('batch_size')

        # Split the original training set into a reduced training set and a
        # validation set. 
        validation_split=request.args.get('validation_split')

        # Identify the feature and the label.
        my_feature=request.args.get('feature1')  # the median income on a specific city block.
        my_label=request.args.get('feature2') # the median value of a house on a specific city block.
        # That is, you're going to create a model that predicts house value based 
        # solely on the neighborhood's median income.  

        # Discard any pre-existing version of the model.
        my_model = None

        # Invoke the functions to build and train the model.
        my_model = build_model(learning_rate)
        epochs, rmse, history = train_model(my_model, train_df, my_feature, my_label, epochs, batch_size, validation_split)

        plot_the_loss_curve(epochs, history["root_mean_squared_error"], history["val_root_mean_squared_error"])

        return "../static/nn_"+name[0]+".png"
    return render_template("meanShift.html", filename = filename, col = col)

if __name__ == '__main__':
    app.run(debug=True)