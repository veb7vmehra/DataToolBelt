import json
import csv
import pandas as pd
from json2xml import json2xml
from json2xml.utils import readfromurl, readfromstring, readfromjson
import xmltodict
import os
import xarray as xr
from netCDF4 import Dataset
import numpy as np

def jsontocsv(inp, out=""):
    data = json.load(open(inp,'r'))
    while len(data) == 1:
        data = data[list(data.keys())[0]]
    data = pd.json_normalize(data)
    if out == "":
        out = "output.csv"
    f = open(out, "w", newline='')
    f.write(data.to_csv())
    f.close()

def jsontoxml(inp, out=""):
    if out == "":
        out = "output.xml"
    f = open(out, "w", newline='')
    data = readfromjson(inp)
    f.write(json2xml.Json2xml(data).to_xml())
    f.close()

def jsontonetCDF(inp, out=""):
    if out == "":
        out = "output.nc"
    jsontocsv(inp,"temp.csv")
    fields = open("temp.csv", 'r').readline().split(',')
    fields[0] = 'ID'
    csvtonetCDF("temp.csv",out,field_list=fields)
    if os.path.exists("temp.csv"):
        os.remove("temp.csv")

def csvtojson(inp, out="", field_list=[]):
    reader = csv.DictReader(open(inp, 'r'), field_list)
    if out == "":
        out = "output.json"
    field_list += ["Field_"+str(i+1) for i in range(len(open(inp, 'r').readline().split(',')) - len(field_list))]
    f = open(out, "w", newline='')
    f.write(json.dumps([row for row in reader]))
    f.close()

def csvtoxml(inp, out="", field_list=[]):
    if out == "":
        out = "output.xml"
    csvtojson(inp,"temp.json")
    jsontoxml("temp.json")
    if os.path.exists("temp.json"):
        os.remove("temp.json")

def csvtonetCDF(inp, out="", field_list=[]):
    if out == "":
        out = "output.nc"
    data = np.array(pd.read_csv(inp, header=None))
    index_list = []
    field_list = [x.strip('\"') for x in field_list]
    field_list += ["Field_"+str(i+1) for i in range(len(open(inp, 'r').readline().split(',')) - len(field_list))]
    for i in range(len(data[0])):
        try:
            float(data[0][i])
            index_list.append(i)
        except:
            pass
    data = data[:,index_list]
    root_grp = Dataset(out, 'w', format='NETCDF4')
    root_grp.description = "DataToolbelt netCDF export"
    root_grp.createDimension("DBT", data.shape[0])
    var = []
    for i in range(data.shape[1]):
        var.append(root_grp.createVariable(field_list[i], 'd', ('DBT',)))
    for i in range(len(var)):
        var[i][:] = data[:, i]
    root_grp.close()

def xmltojson(inp, out=""):
    if out == "":
        out = "output.json"
    f = open(out, "w", newline='')
    f.write(json.dumps(xmltodict.parse(open(inp, 'r').read())))
    f.close()

def xmltocsv(inp, out=""):
    if out == "":
        out = "output.csv"
    xmltojson(inp,"temp.json")
    jsontocsv("temp.json")
    if os.path.exists("temp.json"):
        os.remove("temp.json")

def xmltonetCDF(inp, out=""):
    if out == "":
        out = "output.nc"
    xmltocsv(inp,"temp.csv")
    csvtonetCDF("temp.csv", out)
    if os.path.exists("temp.csv"):
        os.remove("temp.csv")

def netCDFtocsv(inp, out=""):
    if out == "":
        out = "output.csv"
    data = xr.open_dataset(inp)
    data = data.to_dataframe().reset_index()
    data.to_csv(out)

def netCDFtojson(inp, out=""):
    if out == "":
        out = "output.json"
    netCDFtocsv(inp, "temp.csv")
    fields = open("temp.csv", 'r').readline().split(',')
    fields[0] = 'ID'
    with open('temp.csv', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('temp.csv', 'w') as fout:
        fout.writelines(data[1:])
    csvtojson("temp.csv",out=out, field_list=fields)
    if os.path.exists("temp.csv"):
        os.remove("temp.csv")

def netCDFtoxml(inp, out=""):
    if out == "":
        out = "output.xml"
    netCDFtojson(inp,"temp.json")
    jsontoxml("temp.json",out)
    if os.path.exists("temp.json"):
        os.remove("temp.json")
