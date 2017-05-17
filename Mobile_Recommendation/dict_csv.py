# -*- coding: utf-8 -*
    
'''
@author: PY131
'''

import csv 


####
#convert csv file to dict
####

# convert csv file to dict(key-value pairs each column)
def csv2dict(csv_file, key, value):
    new_dict = {}
    with open(csv_file,'r')as f:
        reader = csv.reader(f, delimiter=',')
#         fieldnames = next(reader)
#         reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=',')
        for row in reader:
            new_dict[row[key]] = row[value]
    return new_dict

# convert csv file to dict(key-value pairs each row)
def row_csv2dict(csv_file = ""):
    new_dict = {}
    with open(csv_file)as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            new_dict[row[0]] = row[1]
    return new_dict

####
#convert dict to csv file
####

# convert dict to csv file(key-value pairs each column)
def dict2csv(raw_dict = {}, csv_file = ""):
    with open(csv_file,'w') as f:
        w = csv.writer(f)
        # write all keys on one row and all values on the next
        w.writerow(raw_dict.keys())
        w.writerow(raw_dict.values())
        
# convert dict to csv file(key-value 1-1 pairs each row)       
def row_dict2csv(raw_dict = {}, csv_file = ""):
    with open(csv_file,'w') as f:
        w = csv.writer(f)
        w.writerows(raw_dict.items())

# convert dict to csv file(key-[value] 1-M pairs each row)       
def row2_dict2csv(raw_dict = {}, csv_file = ""):
    with open(csv_file,'w') as f:
        w = csv.writer(f)
        for k,v in raw_dict.items():
            w.writerows([k,v])
        
        

        