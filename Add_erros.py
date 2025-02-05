from pyspark.sql.functions import udf, array, struct, lit
import random
from datetime import datetime, date, timedelta

#2 supressão de palavras
def supPalavra(z):
    word = ""
    for i in range(len(z)):
        if z[i] != " ":
            word += z[i]
        else:
            break
    temp = z[i+1:]
    for j in temp:
        if j != " ":
            temp = temp.replace(j,"",1)
        else:
            break
    return word + temp

def abr1Nome(z):
    word = ""
    cap = z[0].upper()
    for i in range(len(z)):
        if z[i] == " ":
            word = z[i+1:]
        break
    return cap + ". " + word

def abr2Nome(z):
    word = ""
    for i in range(len(z)):
        if z[i] != " ":
            word += z[i]
        else:
            break
    cap = z[i+1].upper()
    temp = z[i+2:]
    for j in temp:
        if j != " ":
            temp = temp.replace(j,"",1)
        else:
            break
    return word + " " + cap + "." + temp

#3 erro grafia
def grafia(z):
    zf = z.replace("SS","Ç")
    if zf != z:
        return zf
    zf = z.replace("SS","S")
    if zf != z:
        return zf
    zf = z.replace("Ç","SS")
    if zf != z:
        return zf
    zf = z.replace("Z","S")
    if zf != z:
        return zf
    zf = z.replace("K","C")
    if zf != z:
        return zf
    zf = z.replace("TH","T")
    if zf != z:
        return zf
    zf = z.replace("EI","E")
    if zf != z:
        return zf
    zf = z.replace("ANA","ANNA")
    if zf != z:
        return zf
    zf = z.replace("DE ","")
    if zf != z:
        return zf
    zf = z.replace("RR","R")
    if zf != z:
        return zf
    zf = z.replace("Y","I")
    if zf != z:
        return zf
    zf = z.replace("W","U")
    if zf != z:
        return zf
    return z

#5 troca de data
def trocaDate(z,days):
    d = datetime.strptime(str(z), '%d/%m/%Y') + timedelta(days=days)
    return d.strftime('%d/%m/%Y')

def erros(df,config):
    #Inserindo erros no dataframe B
    weights = [config['peso_erro']['base'], config['peso_erro']['erro1'],config['peso_erro']['erro2'],config['peso_erro']['erro3'],
               config['peso_erro']['erro4'],config['peso_erro']['erro5']]
    split = df.randomSplit(weights, seed = 1991)
    df_base = split[0]
    df_base = df_base.withColumn("Tipo_erro",lit("Sem erro"))

    #1 supressão de valores
    supressaoUDF = udf(lambda z: "")
    split_1 = split[1].randomSplit([config["peso_erro"]['sup_nome'],config["peso_erro"]['sup_nomeResp'],config["peso_erro"]['sup_data'],
                                    config["peso_erro"]['sup_mun']], seed=1991)
    df_sup_nome = split_1[0].withColumns({"NOME": supressaoUDF("NOME"),"Tipo_erro":lit("tipo 1")})
    df_sup_nomeResp = split_1[1].withColumns({"NOME_RESPONSAVEL": supressaoUDF("NOME_RESPONSAVEL"),"Tipo_erro":lit("tipo 1")})
    df_sup_data = split_1[2].withColumns({"data_nasc":supressaoUDF("data_nasc"),"Tipo_erro":lit("tipo 1")})
    df_sup_mun = split_1[3].withColumns({"COD_MUNICIPIO": supressaoUDF("COD_MUNICIPIO"),"Tipo_erro":lit("tipo 1")})
    df_int1 = df_sup_nome.union(df_sup_nomeResp)
    df_int2 = df_sup_data.union(df_sup_mun)
    df_erro1 = df_int1.union(df_int2)

    #2 supressão de palavra
    supPalavraUDF = udf(lambda z: supPalavra(z))
    abr1NomeUDF = udf(lambda z: abr1Nome(z))
    abr2NomeUDF = udf(lambda z: abr2Nome(z))

    split_2 = split[2].randomSplit([config["peso_erro"]['ret_nome'],config["peso_erro"]['ret_nomeResp'],config["peso_erro"]['abr_nome']
                                    ,config["peso_erro"]['abr_nomeResp'],config["peso_erro"]['abr_nomeM'],config["peso_erro"]['abr_nomeRespM']], seed=1991)
    df_ret_nome = split_2[0].withColumns({"NOME": supPalavraUDF("NOME"),"Tipo_erro":lit("tipo 2")})
    df_ret_nomeResp = split_2[1].withColumns({"NOME_RESPONSAVEL": supPalavraUDF("NOME_RESPONSAVEL"),"Tipo_erro":lit("tipo 2")})
    df_abr1_nome = split_2[2].withColumns({"NOME": abr1NomeUDF("NOME"),"Tipo_erro":lit("tipo 2")})
    df_abr1_nomeResp = split_2[3].withColumns({"NOME_RESPONSAVEL": abr1NomeUDF("NOME_RESPONSAVEL"),"Tipo_erro":lit("tipo 2")})
    df_abr2_nome = split_2[4].withColumns({"NOME": abr2NomeUDF("NOME"),"Tipo_erro":lit("tipo 2")})
    df_abr2_nomeResp = split_2[5].withColumns({"NOME_RESPONSAVEL": abr2NomeUDF("NOME_RESPONSAVEL"),"Tipo_erro":lit("tipo 2")})
    df_int1 = df_ret_nome.union(df_ret_nomeResp)
    df_int2 = df_abr1_nome.union(df_abr1_nomeResp)
    df_int3 = df_abr2_nome.union(df_abr2_nomeResp)
    df_erro2 = df_int1.union(df_int2)
    df_erro2 = df_erro2.union(df_int3)

    #3 erro de grafia
    grafiaUDF = udf(lambda z: grafia(z))
    df_erro3 = split[3].withColumns({"NOME": grafiaUDF("NOME"),"NOME_RESPONSAVEL": grafiaUDF("NOME_RESPONSAVEL"),"Tipo_erro":lit("tipo 3")})

    #4 troca de categoria
    trocaCatUDF = udf(lambda z: z + random.randint(0,10000))
    df_erro4 = split[4].withColumns({"COD_MUNICIPIO": trocaCatUDF("COD_MUNICIPIO"),"Tipo_erro":lit("tipo 4")})

    #5 troca de data
    trocaDateUDF = udf(lambda z: trocaDate(z,random.randint(0,365)))
    df_erro5 = split[5].withColumns({"data_nasc": trocaDateUDF("data_nasc"),"Tipo_erro":lit("tipo 5")})

    df_b_int1 = df_base.union(df_erro1)
    df_b_int2 = df_erro2.union(df_erro3)
    df_b_int3 = df_erro4.union(df_erro5)
    df_b = df_b_int1.union(df_b_int2)
    df_b = df_b.union(df_b_int3)

    return df_b