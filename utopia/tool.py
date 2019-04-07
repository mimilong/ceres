from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
import pymongo
import pandas as pd
import numpy as np
import time

def pub_geti_from_conf(conf, item):
    result = conf.items(item)
    return {i[0]:i[1] for i in result}

def load_data_through_ssh(sql,conf_db,conf_ssh,db_type='postgresql'):
    with SSHTunnelForwarder(
            ssh_address_or_host=(conf_ssh.get('host'), int(conf_ssh.get('port'))),
            ssh_username=conf_ssh.get('user'),
            ssh_password=conf_ssh.get( 'pwd'),
            remote_bind_address=(conf_db.get('host'),
                                 int(conf_db.get('port')))) as server:
        port = server.local_bind_port
        engine = create_engine('{}://{}:{}@127.0.0.1:{}/{}'.format(db_type,conf_db.get('user'),
                                                                        conf_db.get('pwd'),port,
                                                                        conf_db.get('db')),
                               echo=True)
        frame = pd.read_sql_query(sql, con=engine)
        return frame

# ------- 从mongo库获取运营商数据
def load_data_from_mongo(query,conf,table_name,condition=None):

    # mongo 连接信息
    conn = pymongo.MongoClient(conf.get('host'), conf.get('port'))
    db = conn.sctx_data_product
    db.authenticate(conf.get('user'), conf.get('pwd'))
    table_names = db.getCollectionNames()
    table_names = [t for t in table_names if t[len(table_name)-1] == table_name]

    df = []
    for t in table_names:
        collection = db.getCollection(t)
        frame1 = collection.find(*query)
        for d in frame1:
            df.append(dict({k:v for k,v in d.items() if k != 'handlerData'}, **d['handlerData']))
    df = pd.DataFrame(df)
    return df
