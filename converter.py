import json
import csv
import pandas as pd
from json2xml import json2xml
from json2xml.utils import readfromurl, readfromstring, readfromjson
import xmltodict
import os

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

def csvtojson(inp, out="", field_list=[]):
    reader = csv.DictReader(open(inp, 'r'), field_list)
    if out == "":
        out = "output.json"
    field_list += ["Field_"+str(i+1) for i in range(len(open(inp, 'r').readline().split(',')) - len(field_list))]
    f = open(out, "w", newline='')
    f.write(json.dumps([row for row in reader]))
    f.close()

def csvtoxml(inp, out=""):
    csvtojson(inp,"temp.json")
    jsontoxml("temp.json")
    if os.path.exists("temp.json"):
        os.remove("temp.json")


def xmltojson(inp, out=""):
    if out == "":
        out = "output.json"
    f = open(out, "w", newline='')
    f.write(json.dumps(xmltodict.parse(open(inp, 'r').read())))
    f.close()

def xmltocsv(inp, out=""):
    xmltojson(inp,"temp.json")
    jsontocsv("temp.json")
    if os.path.exists("temp.json"):
        os.remove("temp.json")

#jsontocsv("dummy.json")
#jsontoxml("dummy.json")
xmltocsv("dummy.xml")