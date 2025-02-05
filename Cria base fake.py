print("Iniciando fake")

# importando o pacote necessário para iniciar uma seção Spark
from pyspark.sql import SparkSession

# iniciando o spark context
sc = SparkSession.builder.master('local[*]').getOrCreate()

from pyspark.sql.functions import col, lit, upper

df = sc.read.csv(path = "C:\\Users\\pitu_\\Documents\\Faculdade\\Sistemas\\2023-1\\TCC I\\Base sintetica\\202401_NovoBolsaFamilia.csv",
                 inferSchema = True, header = True, sep = ';', encoding = "UTF-8")

#Corrigir Títulos das Colunas
df = df.withColumnRenamed("NOME FAVORECIDO", "NOME")
df = df.withColumnRenamed("NOME MUNIC�PIO", "MUNICIPIO")

df_mun = sc.read.csv(path = "C:\\Users\\pitu_\\Documents\\Faculdade\\Sistemas\\2023-1\\TCC I\\Base sintetica\\COD_MUN.csv",
                 inferSchema = True, header = True, sep = ';', encoding = "UTF-8")

df_mun = df_mun.withColumn('MUNICIPIO', upper(col('MUNICIPIO')))
df = df.join(df_mun,df.MUNICIPIO ==  df_mun.MUNICIPIO,"leftouter")

#Dividindo o dataframe para cerca de 1MM de registros mantendo a proporção dos estados
fractions = df.select("UF").distinct().withColumn("fraction", lit(0.05)).rdd.collectAsMap()
df_base = df.sampleBy("UF", fractions, seed=1991) #dataframe base
df_responsavel = df.sampleBy("UF", fractions, seed=312) #dataframe para extração dos responsáveis
df_base = df_base.select("COD_MUNICIPIO","NOME")
df_base_pandas = df_base.toPandas()
df_responsavel_pandas = df_responsavel.toPandas()
df_base_pandas["NOME_RESPONSAVEL"] = df_responsavel_pandas["NOME"]
rows_base = df_base.count()
rows_responsavel = df_responsavel.count()
if rows_base < rows_responsavel:
  rows = rows_base
else:
  rows = rows_responsavel

from sdv.datasets.local import load_csvs
#FOLDER_NAME = 'C:\\Users\\pitu_\\Documents\\Faculdade\\Sistemas\\2023-1\\TCC I\\Base sintetica\\Datas_fakes'
try:
  data = load_csvs(folder_name="C:\\Users\\pitu_\\Documents\\Faculdade\\Sistemas\\2023-1\\TCC I\\Base sintetica\\Data_fake")
except ValueError:
  print('You have not uploaded any csv files.')

from sdv.metadata import Metadata
metadata = Metadata.detect_from_dataframes(data)
metadata.visualize()

#Creating a Synthesizer
#An SDV synthesizer is an object that you can use to create synthetic data. It learns patterns from the real data and replicates them to generate synthetic data.
from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data['Datas_fake'])

synthetic_data = synthesizer.sample(num_rows=rows)
df_base_pandas["data_nasc"] = synthetic_data["data_nasc"]

df_base_pandas = df_base_pandas.dropna()

df_base_pandas.to_csv("C:\\Users\\pitu_\\Documents\\Faculdade\\Sistemas\\2023-1\\TCC I\\Base sintetica\\base_sintetica_oficial.csv")