import sqlalchemy as sa
import csv
import os
import pymysql
import numpy as np
import cPickle
import pandas as pd
from odo import odo


## CLASS DEFINITIONS

# DO NOT MODIFY! - Table structure information object
class tabEntry:
    def __init__(self, name=None, dfiles=None, cols=None):
        self.name=name # name of table
        self.dfiles=dfiles   # path to csv source (string)
        self.cols=cols # list of tuples [(cName, cType), ...]


## GLOBALS - Modify!

# dictionary of MySql server config files
d_sql = {'vm-ubuntu01': '/home/john/client-ssl/CastawayCay_MySQL/.writeclient.cnf',
         'john-desktop': '/home/john/Application_Files/MySQL/.local_sql.cnf'}
wCliOpt = d_sql[os.uname()[1]] # choose config file base on hostname
# dictionary of data directory
d_dat = {'vm-ubuntu01': '/home/john/AnacondaProjects/Y790_Cran/17s-cran/CIFAR10/data/data/',
         'john-desktop': '/home/john/Documents/IU_Cyber/17S-Y790-Cran/17s-cran/CIFAR10/data/data/'}
DDIR = d_dat[os.uname()[1]]
DB_NAME = '17s_cran'
TABLES = []
TABLES.append(tabEntry(name = 'CIFAR10',
                       dfiles = [DDIR+f for f in os.listdir(DDIR)],
                       cols = [('data', sa.Text),
                               ('labels', sa.Integer),
                               ('filenames', sa.Text)]))
OVERWRITE = True
B_LD_SZ = 100
blah = pd.DataFrame()



## FUNCTIONS

# Extract the dickled dict from dfile and insert it into the tbl
def load_cifar10_old(dfile, tbl):
    f = open(dfile, 'rb')
    d = cPickle.load(f)
    f.close()
    for i in range(len(d['data'])):
        # create the insert() statement
        ins = tbl.insert().values(data=cPickle.dumps(d['data'][i]),
                         labels=d['labels'][i],
                         filenames=d['filenames'][i]
                         )
        ins.execute() # execute the inserte() statement

# Improved load using odo
# d = cPickle.load(open(DDIR+'data_batch_1'))
def load_cifar10(dfile, tbl):
    f = open(dfile, 'rb')
    d = cPickle.load(f)
    f.close()
    df = pd.DataFrame()
    df['data'] = np.array([cPickle.dumps(i) for i in d['data']])
    df['labels'] = d['labels']
    df['filenames'] = d['filenames']
    odo(df, tbl)

# DO NOT MODIFY! - Creates a connection to MySQL Server
def mysql_connect_w():
    return pymysql.connect(read_default_file=wCliOpt)

# DO NOT MODIFY! - Creates a connection to MySQL Database
def db_connect_w():
    return pymysql.connect(read_default_file=wCliOpt, db=DB_NAME)

# DO NOT MODIFY! - Creates a database or returns an error
def create_database(cursor):
    print 'Database `{}`:'.format(DB_NAME),
    try: cursor.execute(
            'CREATE DATABASE {}'.format(DB_NAME))
    except pymysql.MySQLError as err:
        print 'already exists.'
    else:
        print 'OK.'

# DO NOT MODIFY! - Adds an additional auto-increment column to table
def add_autoinc_col(db, t):
    conn = db.raw_connection()
    cursor = conn.cursor()
    cursor.execute('ALTER TABLE `'+t+'` ADD `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST')
    cursor.close() # Close handle for SQL server
    conn.close() # Close the connection to the SQL server


## MAIN

# Create Database
sql_eng = sa.create_engine('mysql+pymysql://', creator=mysql_connect_w)
conn = sql_eng.raw_connection()
cursor = conn.cursor() 
create_database(cursor)
cursor.close() # Close handle for SQL server
conn.close() # Close the connection to the SQL server

# Creat the sqlalchemy engine
db_eng = sa.create_engine('mysql+pymysql://', creator=db_connect_w)
metadata = sa.MetaData(bind=db_eng) # holds database information

# Loop through all table entries
for te in TABLES:
    # create the sqlalchemy.table object
    tbl = sa.Table(
        te.name,
        metadata,
        *[sa.Column(c[0],c[1]) for c in te.cols])
    print 'Table `{}`:'.format(te.name),
    if tbl.exists(): 
        if OVERWRITE:
            print 'exists; over-writing.',
            tbl.drop(checkfirst=True)
        else:
            print 'exists; skipping.'
            break
    tbl.create()
    add_autoinc_col(db_eng, te.name)    # Add AUTOINCREMENT COLUMN
    for dfile in te.dfiles: 
        print '.',
        load_cifar10(dfile,tbl)                    # bulk-load the .csv using odo
    print 'OK.'
print 'Table loading complete!'




