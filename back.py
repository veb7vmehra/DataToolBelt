# hackathon T - Hacks 3.0
# flask backend of data-cleaning website
print("Hello")
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
    explode = (0.1, 0, 0, 0, 0)  
    plt.pie(sums, labels=sums.index, explode = explode, autopct='%1.1f%%', shadow=True, startangle=140)
<<<<<<< HEAD
    plt.title("Pie chart on basis of "+feature2)
    name = filename.split('.')
    plt.savefig(name[0]+".png")

def feature_scatter(filename, feature1, feature2):
    df = pd.read_csv(filename)
    sums = df.groupby(df[feature1])[feature2].sum()
    plt.axis('equal')
    plt.pie(sums, labels=sums.index, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Scatter plot between "+feature1+" and "+feature2)
=======
    plt.title("Pie chart on basis of "+feature1)
>>>>>>> 4d7aab5919b684f17091c5aab35d829db6139cd8
    name = filename.split('.')
    plt.savefig(name[0]+".png")

def new_feature(filename, com, name = " "):
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
    if name != " ":
        i = 1
    n = len(df)
    for j in range(i, n):
        for k, c in enumerate(com):
            if k%2 == 0:
                if c == "formula":
                    break
                formula.replace(c, df.at[j, com[k+1]])
        vals.append(test(formula))
        formula = temp
    col = len(df.axes[1]) + 2
    if name != " ":
        df.insert(col, name, vals, True)
    else:
        df.insert(col, vals, True)
    os.remove(filename)
    df.to_csv(filename) 

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
            elif ext == "nc"
                con.netCDFtocsv("static/"+f.filename, "static/"+name+".csv")
            n_row, n_col, col, types, line0, line1, line2, line3, line4, line5 = disp("static/"+f.filename)
            res = make_response(render_template("filedata.html", filename = f.filename, n_row = n_row, n_col = n_col, col = col, types = types, lists = "../static/"+name+".csv"))
            res.set_cookie("filename", value=f.filename)
            return res
    return render_template("upload.html")

@app.route('/stat', methods = ['GET', 'POST'])
def stats():
    if request.method == 'GET':
        filename = request.args.get('filename').split('/')[-1]
        name = filename.split('.')
        ext = name[-1]
        name = name[0]
        if ext == "json":
            con.jsontocsv("static/"+filename, "static/"+name+".csv")
<<<<<<< HEAD
        elif ext == "nc":
            con.netCDFtocsv("static/"+filename, "static/"+name+".csv")
=======
>>>>>>> 4d7aab5919b684f17091c5aab35d829db6139cd8
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
        n_row, n_col, col, types, line0, line1, line2, line3, line4, line5 = disp("static/"+name+".csv")
        return render_template("filedata.html", filename = name+"."+to, n_row = n_row, n_col = n_col, col = col, types = types, lists = "../static/"+name+".csv")
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
    if request.method == 'POST':
        feature1 = request.form['feature1']
        feature2 = request.form['feature2']
        feature_pie("static/"+name+".csv", feature1, feature2)
        return render_template("analysis.html", col = col, img = "static/"+name+".png")
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
        name = request.form['name']
        com = request.form['formula']
        new_feature(filename, com, name)
        return "../static/"+name+".png"

@app.route('/clean', methods = ['GET', 'POST'])
def analyse():
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
        name = request.form['name']
        com = request.form['formula']
        new_feature(filename, com, name)
        return "../static/"+name+".png"

if __name__ == '__main__':
    app.run(debug=True)