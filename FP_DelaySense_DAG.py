import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

RAW_PATH        = '/opt/airflow/dags/Train.csv'
NORMALIZED_PATH = '/opt/airflow/dags/dataset_DA.csv'

# buat koneksi ke postgres
username = 'grup4'
password = 'grup4'
host = 'postgres'
port = '5432'
database = 'postgres'
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')

def extract_data():

    # ambil data dari csv
    df = pd.read_csv(RAW_PATH)

    # normalisasi nama kolom
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r'[^\w]', '_', regex=True)
    )

    # kirim dataset ke postgres
    df.to_sql('logistic', engine, if_exists='replace') 

def transform_data():
    # baca tabel yang diambil dari postgres
    df = pd.read_sql('SELECT * FROM logistic', con=engine)

    # drop duplicates & missing
    df = df.drop_duplicates().dropna()
    
    # rename reached_on_time_y_n → reached_on_time
    df.rename(columns={'reached_on_time_y_n': 'reached_on_time'}, inplace=True)

    # handling cardinality
    def handle_cardinality(w):
        if w < 1000:            return 'Low Weight'
        elif w < 1500:          return 'Below Medium'
        elif w < 3000:          return 'Medium Weight'
        elif w < 4500:          return 'Intermediate Weight'
        elif w < 6000:          return 'High Weight'
        else:                   return 'Very High Weight'

    if 'weight_in_gms' in df.columns:
        df['weight_category'] = df['weight_in_gms'].apply(handle_cardinality)

    # save dataset ke csv
    df.to_csv(NORMALIZED_PATH, index=False)

def load_data():
    df = pd.read_csv(NORMALIZED_PATH)
    # save ke csv
    df.to_csv('/opt/airflow/dags/dataset_DS.csv', index=False)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

with DAG(
    dag_id='etl_transform_load_ml_split',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['ETL','ML','split'],
) as dag:

    t_extract   = PythonOperator(task_id='extract_data',   python_callable=extract_data)
    t_transform = PythonOperator(task_id='transform_data', python_callable=transform_data)
    t_load      = PythonOperator(task_id='load_data',      python_callable=load_data)

    t_extract >> t_transform >> t_load