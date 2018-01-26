import mysql.connector
import numpy as np
import pandas as pd
import csv

## GLOBALS

wPWD = 'password'
rPWD = 'password'
DB_NAME = 'iris'
D_FILE = 'iris.data'
CNK_SIZE = 100

# Tables Creation Statements
TABLES = {}
TABLES['data'] = (
        "CREATE TABLE `data` ("
        "    `sepal-length` FLOAT(10),"
        "    `sepal-width` FLOAT(10),"
        "    `petal-length` FLOAT(10),"
        "    `petal-width` FLOAT(10),"
        "    `class` VARCHAR(20)"
        ")")


## FUNCTIONS

# Creates a database or returns an error
def create_database(cursor):
    try: cursor.execute(
            'CREATE DATABASE {}'.format(DB_NAME))
    except mysql.connector.Error as err:
        print('Failed creating database: {}'.format(err))
        exit(1)


## MAIN

# Connect to the server and grab the cursor
#conn = mysql.connector.connect(host='CastawayCay', user='writeclient', password=wPWD)
conn = mysql.connector.connect(host='127.0.0.1', user='root', password=raw_input('mysql password:')
cursor = conn.cursor()

# Use the DB_NAME database or create it if doesn't exist
try: conn.database = DB_NAME
except mysql.connector.Error as err:
    if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        conn.database = DB_NAME
    else:
        print(err)
        exit(1)

# Iterate through table statements to create tables
for table in TABLES.keys():
    try:
        print 'Creating table %s: ' % table,
        cursor.execute(TABLES[table])
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_TABLE_EXISTS_ERROR:
            print 'already exists.'
        else:
            # TBD
            # write a loop to add chunked data
            print 'OK.'
    
    
with open(D_FILE) as csvfile:
    data = [row for row in csv.reader(csvfile)]
#    print data





cursor.close() # Close handle for SQL server
conn.close() # Close the connection to the SQL server
