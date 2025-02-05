print("Iniciando Spark")

# importando o pacote necessário para iniciar uma seção Spark
from pyspark.sql import SparkSession

# iniciando o spark context
sc = SparkSession.builder.master('local[*]').config("spark.driver.memory", "32g").getOrCreate()

#Importando as bibliotecas
from pyspark.sql.functions import udf, array, struct, explode, lit
from pyspark.sql.functions import sum as soma
import math
import random
import warnings
warnings.filterwarnings('ignore')
import json
from Functions import *
from Add_erros import erros
from Schema import schema_check, schema
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = sc.read.csv(path = './base_sintetica_oficial.csv',
    inferSchema = True, header = True, sep = ',', encoding = "UTF-8")

df_a = sc.createDataFrame(df.collect()[:20000])
df_a = df_a.withColumnRenamed("_c0","ID")
registros_a = df_a.count()
df_a = df_a.withColumn("COD_MUNICIPIO",df_a.COD_MUNICIPIO.cast(IntegerType()))
df_b_int = sc.createDataFrame(df.collect()[:20000])
df_b_inc = sc.createDataFrame(df.collect()[100000:100100])
registros_inc = df_b_inc.count()
df_b_int = df_b_int.union(df_b_inc)
df_b_int = df_b_int.withColumnRenamed("_c0","ID")
df_b_int = df_b_int.withColumn("COD_MUNICIPIO",df_b_int.COD_MUNICIPIO.cast(IntegerType()))
registros = df_b_int.count()
df_a = df_a.withColumn('vet',array('NOME','NOME_RESPONSAVEL','data_nasc','COD_MUNICIPIO'))

config_file = 'config_basefake.json'
f = open(config_file)
config_json = json.load(f)

for inter in range(1,2):
    config = config_json["config"+str(inter)]

    df_b = erros(df_b_int,config)
    df_b = df_b.withColumn('vet',array('NOME','NOME_RESPONSAVEL','data_nasc','COD_MUNICIPIO'))

    #definindo variáveis
    m = config['tamanho_BF'] #tamanho de vetor de bloom de cada atributo (testar alguns tamanho - range: 100 até 1000 - passo 100)
    a = m*(len(df.columns)-1) #tamanho do filtro de bloom completo
    k = config['quant_hash'] #quantidade de iterações no hash = qtd de posições do bloom que seram alteradas para 1 a cada bigrama (2 e 3)
    t_hash = config['tipo_hash'] #define o tipo de hash q será usado no programa

    #Informações para o Record Level Bloom Filter
    try:
        NumberPosition = int(config['porcent_rlb']*m) #número de posições a serem extraidas do filtro de bloom (10% 20% 30%)

        #Escolhendo as posições aleatórias entre 0 e m-1 para o Record Level Bloom Filter
        positions = []
        for i in range(len(df.columns)-1):
            randomlist = random.sample(range(0, m-1), NumberPosition)
            #Definindo em quais posições os bits do primeiro filtro de bloom irão para o segundo - https://blog.betrybe.com/python/python-random/
            for j in range(m):
                positions.append(random.choice(randomlist))
            randomlist.clear()

        #Posições do vetor de bloom final
        vetor = []
        for i in range(m*(len(df.columns)-1)):
            vetor.append(i)

        #Algoritmo de embaralhamento de Fisher–Yates
        for i in range(m*(len(df.columns)-1)):
            j = math.floor(random.random() * (i + 1))
            [vetor[i], vetor[j]] = [vetor[j], vetor[i]]
    except:
        NumberPosition = 0

    # record start time
    start = time.time()

    #https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
    abfUDF = udf(lambda z: abf(z,m,k,t_hash),ArrayType(IntegerType()))
    df_a_ABF = df_a.withColumn('bloom', abfUDF(df_a.vet))
    df_b_ABF = df_b.withColumn('bloom', abfUDF(df_b.vet))

    clkUDF = udf(lambda z: clk(z,a,k,t_hash),ArrayType(IntegerType()))
    df_a_CLK = df_a.withColumn('bloom', clkUDF(df_a.vet))
    df_b_CLK = df_b.withColumn('bloom', clkUDF(df_b.vet))

    rlbUDF = udf(lambda z: rlb(z,a,m,k,t_hash,positions,vetor),ArrayType(IntegerType()))
    df_a_RLB = df_a.withColumn('bloom', rlbUDF(df_a.vet))
    df_b_RLB = df_b.withColumn('bloom', rlbUDF(df_b.vet))

    dataframe_a = eval(config["tipo_bloom"]["A"])
    dataframe_b = eval(config["tipo_bloom"]["B"])

    name_list = dataframe_a.select("BF").collect()
    IDs_list = dataframe_a.select("ID").collect()

    arv = criarMBT(name_list,IDs_list,0,a,5)

    BuscaUDF = udf(lambda z: busca(z, arv), schema)
    dataframe_b = dataframe_b.withColumn('busca', BuscaUDF(dataframe_b.BF))
    dataframe_b = dataframe_b.withColumn('config_number', lit(inter))
    #dataframe_b.select('config_number','ID', 'Tipo_erro', 'busca.baseA', 'busca.dice').write.mode('append').csv(
        #"dataframes_basefake.csv", header=True)

    acertoUDF = udf(lambda z: acerto(z[0], z[1]), IntegerType())
    dataframe_b = dataframe_b.withColumn('acerto', acertoUDF(struct(dataframe_b.ID, dataframe_b.busca.baseA)))

    # https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
    y_real = dataframe_b.select('acerto').rdd.flatMap(lambda x: x).collect()
    y_pred = dataframe_b.select('busca.dice').rdd.flatMap(lambda x: x).collect()
    media = sum(y_real) / len(y_real)
    if media == 1:
        y_real.append(0)
        y_pred.append(0)
    fpr, tpr, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    list_erro = dataframe_b.select("Tipo_erro").rdd.flatMap(lambda x: x).collect()
    list_erro = list(dict.fromkeys(list_erro))
    list_erro.sort()

    for erro in list_erro:
        dataframe_b_erro = dataframe_b.filter(dataframe_b.Tipo_erro == erro)
        registros = dataframe_b_erro.count()
        if erro == "Sem erro":
            true_matches = registros - registros_inc
        else:
            true_matches = registros

        dataframe_check = dataframe_b_erro.select(dataframe_b_erro.ID, explode(dataframe_b_erro.busca.pares_candidatos))
        dataframe_check = dataframe_check.withColumn('acerto', acertoUDF(struct(dataframe_check.ID, dataframe_check.key)))
        checkUDF = udf(lambda z: check(z[0], z[1], optimal_threshold), schema_check)
        dataframe_check = dataframe_check.withColumn('pares', checkUDF(struct(dataframe_check.acerto, dataframe_check.value)))

        calcFNUDF = udf(lambda z: calcFN(z, registros), IntegerType())

        # Pair Completeness or Recall or Sensitivity
        calcPcomUDF = udf(lambda z: calcPcom(z, registros), FloatType())

        # Reduction Rate
        calcRedRateUDF = udf(lambda z: calcRedRate(z[0], z[1], z[2], z[3], registros_a, registros), FloatType())

        # Pair Quality
        calcPairQualUDF = udf(lambda z: calcPairQual(z[0], z[1], z[2], z[3]), FloatType())

        # Acuracia
        calcAccUDF = udf(lambda z: calcAcc(z[0], z[1], z[2], z[3]), FloatType())

        # Precisao
        calcPreUDF = udf(lambda z: calcPre(z[0], z[1]), FloatType())

        # F-score
        calcFscUDF = udf(lambda z: calcFsc(z[0], z[1]), FloatType())

        pair = dataframe_check.select(soma("pares.VP").alias("VP"), soma("pares.FP").alias("FP"), soma("pares.VN").alias("VN"))
        pair = pair.withColumn('FN', calcFNUDF(pair.VP))
        pair = pair.withColumns({'tamanho_BF': lit(config["tamanho_BF"]), 'quant_hash': lit(config["quant_hash"]),
                                 'tipo_hash': lit(config["tipo_hash"]), 'tipo_BF': lit(config["tipo_bloom"]["nome"]),
                                 'porcentagem_erro':lit(round(1 - config["peso_erro"]["base"],2)), 'porcentagem_rlb': lit(NumberPosition / m), 'threshold': lit(optimal_threshold),
                                 'roc_area': lit(roc_auc)})
        pair = pair.withColumns({'pair_completeness': calcPcomUDF(pair.VP),
                                 'reduction rate': calcRedRateUDF(struct(pair.VP, pair.VN, pair.FP, pair.FN)),
                                 'pair quality': calcPairQualUDF(struct(pair.VP, pair.VN, pair.FP, pair.FN)),
                                 'acuracia': calcAccUDF(struct(pair.VP, pair.VN, pair.FP, pair.FN)),
                                 'precisao': calcPreUDF(struct(pair.VP, pair.FP))})
        pair = pair.withColumns({'f-score': calcFscUDF(struct(pair.precisao, pair.pair_completeness)), 'tempo_execucao': lit(calc_time(start)),
                                 "Tipo_erro":lit(erro),'tam_baseB':lit(10)})
        #pair.write.mode('append').csv("result_basefake_100k_"+erro+".csv", header=True)
        pair.show()

# Stop the SparkSession
sc.stop()