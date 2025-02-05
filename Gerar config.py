import itertools

def generate_configurations():
    # Lista de valores para cada parâmetro
    tamanhos_BF = [100, 200, 400]
    #quant_hashes = [2, 3]
    #tipos_hash = ["DoubleHash", "TripleHash", "EnhancedDHash"]
    nomes = ["ABF", "CLK", "RLB"]
    bases = [0.02, 0.05, 0.1]
    #rlb_porcs = [0.3, 0.4, 0.5]

    # Cria um iterador com todas as combinações possíveis
    combinations = itertools.product(tamanhos_BF, nomes, bases)

    # Itera sobre cada combinação e cria um dicionário de configuração
    i=0
    for combo in combinations:
        tamanho_BF, nome, base = combo
        if nome == "RLB":
            #for porct in {0.3, 0.4, 0.5}:
                i += 1
                config = {
                    "config"+str(i):{
                        "tamanho_BF": tamanho_BF,
                        "quant_hash": 2,
                        "tipo_hash": "TripleHash",
                        "tipo_bloom": {
                            "nome": nome,
                            "A": f"df_a_{nome}.select('ID','bloom').withColumnRenamed('bloom','BF')",
                            "B": f"df_b_{nome}.select('ID','bloom','Tipo_erro').withColumnRenamed('bloom','BF')"
                        },
                        "peso_erro": {
                            "base": 1-5*base,
                            "erro1": base,
                            "erro2": base,
                            "erro3": base,
                            "erro4": base,
                            "erro5": base,
                            "sup_nome": 0.25,
                            "sup_nomeResp": 0.25,
                            "sup_data": 0.25,
                            "sup_mun": 0.25,
                            "ret_nome": 0.15,
                            "ret_nomeResp": 0.15,
                            "abr_nome": 0.15,
                            "abr_nomeResp": 0.15,
                            "abr_nomeM": 0.15,
                            "abr_nomeRespM": 0.15
                        },
                        'porcent_rlb': 0.4
                    }
                }
                with open('output.txt', 'a') as f:
                    print(config, file=f)  # Imprime cada configuração no formato JSON
        else:
            i += 1
            config = {
                "config" + str(i): {
                    "tamanho_BF": tamanho_BF,
                    "quant_hash": 2,
                    "tipo_hash": "TripleHash",
                    "tipo_bloom": {
                        "nome": nome,
                        "A": f"df_a_{nome}.select('ID','bloom').withColumnRenamed('bloom','BF')",
                        "B": f"df_b_{nome}.select('ID','bloom','Tipo_erro').withColumnRenamed('bloom','BF')"
                    },
                    "peso_erro": {
                        "base": 1 - 5 * base,
                        "erro1": base,
                        "erro2": base,
                        "erro3": base,
                        "erro4": base,
                        "erro5": base,
                        "sup_nome": 0.25,
                        "sup_nomeResp": 0.25,
                        "sup_data": 0.25,
                        "sup_mun": 0.25,
                        "ret_nome": 0.15,
                        "ret_nomeResp": 0.15,
                        "abr_nome": 0.15,
                        "abr_nomeResp": 0.15,
                        "abr_nomeM": 0.15,
                        "abr_nomeRespM": 0.15
                    }
                }
            }
            with open('output.txt', 'a') as f:
                print(config,file=f)  # Imprime cada configuração no formato JSON

if __name__ == "__main__":
    generate_configurations()