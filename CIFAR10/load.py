import sqlalchemy as sa
import csv
import os
import pymysql
import cPickle
from odo import odo


## CLASS DEFINITIONS

# DO NOT MODIFY! - Table structure information object
class tabEntry:
    def __init__(self, name=None, dfiles=None, cols=None):
        self.name=name # name of table
        self.dfiles=dfiles   # path to csv source (string)
        self.cols=cols # list of tuples [(cName, cType), ...]


## GLOBALS - Modify!

wCliOpt = '/home/john/client-ssl/CastawayCay_MySQL/.writeclient.cnf' # from Tortuga VM
#wCliOpt = '/home/john/Application_Files/MySQL/.local_sql.cnf' # from John's laptop
DDIR = '/home/john/AnacondaProjects/Y790_Cran/17s-cran/CIFAR10/data/data/'
DB_NAME = '17s_cran'
TABLES = []
TABLES.append(tabEntry(name = 'CIFAR10',
                       dfiles = [DDIR+'data_batch_1', DDIR+'data_batch_2', 
                                DDIR+'data_batch_3', DDIR+'data_batch_4', 
                                DDIR+'data_batch_5', DDIR+'test_batch'],
                       cols = [('data', sa.Text),
                               ('labels', sa.Integer),
                               ('filenames', sa.Text)]))



## FUNCTIONS

# Performs rudimentary data cleaning on source file
#def clean(fname):
#    name, ext = os.path.splitext(fname)
#    from_file = csv.reader(open(fname, 'r'), delimiter=',')
#    to_file = csv.writer(open(name+'_cleaned'+ext, 'w+'), delimiter=',')
#    for r in from_file:
#        r = [s.replace('-','_') for s in r] # replace '-' with '_'
#        to_file.writerow(r)
#    return name+'_cleaned'+ext

def load_cifar10(dfile, tbl):
    f = open(dfile, 'rb')
    d = cPickle.load(f)
    f.close()
    for i in range(len(d['data'])):
        ds = cPickle.dumps(d['data'][i])
        ins = tbl.insert().values(data=ds,
                  labels=d['labels'][i],
                  filenames=d['filenames'][i]
                  )
        ins.execute()

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
        print 'exists; over-writing.',
        tbl.drop(checkfirst=True)
    tbl.create()
    add_autoinc_col(db_eng, te.name)    # Add AUTOINCREMENT COLUMN
    for dfile in te.dfiles: 
        print '.',
        load_cifar10(dfile,tbl)                    # bulk-load the .csv using odo
    print 'OK.'
print 'Table loading complete!'




