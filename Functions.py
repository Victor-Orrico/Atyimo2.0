import hashlib
import time
import numpy as np
from pyspark.sql.types import *

#Double Hashing
#https://docs.python.org/3/library/hashlib.html
def dbhash(registro,cont,tam):
  vhash1 = hashlib.md5(registro.encode()).hexdigest()
  vhash2 = hashlib.sha256(registro.encode()).hexdigest()
  pos = (int(vhash1,16) + cont*int(vhash2,16))%tam
  return pos

#Triple Hashing
def tphash(registro,cont,tam):
  vhash1 = hashlib.md5(registro.encode()).hexdigest()
  vhash2 = hashlib.sha256(registro.encode()).hexdigest()
  vhash3 = hashlib.sha512(registro.encode()).hexdigest()
  pos = (int(vhash1,16) + cont*int(vhash2,16) + int((cont*(cont-1)/2))*int(vhash3,16))%tam
  return pos

#Enhanced Double Hashing
def endbhash(registro, cont,tam):
  vhash1 = hashlib.md5(registro.encode()).hexdigest()
  vhash2 = hashlib.sha256(registro.encode()).hexdigest()
  pos = (int(vhash1,16) + cont*int(vhash2,16) + int((cont^3-cont)/6))%tam
  return pos

#ABF UDF
def abf(registro,m,k,t_hash):
  bloom_f = []
  for item in registro:
    bloom = [0] * m
    if item is None:
      pass
    else:
      item = str(item)
      item = str('_') + item
      item = item + str('_')
      for j in range(len(item)-1):
        cont = 1
        while cont <=k:
          string = item[j:j+2]
          if t_hash == "DoubleHash":
            pos = dbhash(string,cont,m)
          elif t_hash == "TripleHash":
            pos = tphash(string,cont,m)
          else:
            pos = endbhash(string,cont,m)
          bloom[pos] = 1
          cont +=1
    bloom_f.extend(bloom) #concatenando os filtros de Bloom
  return bloom_f

def abf_block(registro):
  bloom = [0] * 30
  if registro is None:
    return bloom
  else:
    registro = str(registro)
    registro = str('_') + registro
    registro = registro + str('_')
    for j in range(len(registro)-1):
      cont = 1
      while cont <=3:
        string = registro[j:j+2]
        pos = dbhash(string,cont,30)
        bloom[pos] = 1
        cont +=1
  return bloom

#CLK UDF
def clk(registro,a,k,t_hash):
  bloom = [0] * a
  for item in registro:
    if item is None:
      pass
    else:
      item = str(item)
      item = str('_') + item
      item = item + str('_')
      for j in range(len(item)-1):
        cont = 1
        while cont <=k:
          string = item[j:j+2]
          if t_hash == "DoubleHash":
            pos = dbhash(string,cont,a)
          elif t_hash == "TripleHash":
            pos = tphash(string,cont,a)
          else:
            pos = endbhash(string,cont,a)
          bloom[pos] = 1
          cont +=1
  return bloom

#RLB UDF
def rlb(registro,a,m,k,t_hash,positions,vetor):
  bloom_int = []
  bloom_fin = [0] * a
  start = 0
  c = 0
  bloom = abf(registro,m,k,t_hash)
  while start < a:
    #Criando o novo filtro de Bloom só com as posições sorteadas
    for i in range(start,m+start):
      bloom_int.append(bloom[positions[i]+c])
    start += m
    c += m
  for k in range(len(bloom_int)):
    bloom_fin[k] = bloom_int[vetor[k]]
  return bloom_fin

class NodoArvore:
  def __init__(self, chave=None, esquerda=None, direita=None):
    self.chave = chave
    self.esquerda = esquerda
    self.direita = direita

  def __repr__(self):
    return '%s' % self.chave
#https://algoritmosempython.com.br/cursos/algoritmos-python/estruturas-dados/arvores/

def apresentacao (arv: NodoArvore):
  try:
    if arv.esquerda:
      apresentacao(arv.esquerda)
  except AttributeError:
      pass

  try:
    if arv.direita:
      apresentacao(arv.direita)
  except AttributeError:
      pass
  
  print(arv)
  return

def criarMBT(lista, IDs, cont, tam_bf, group):
  soma = [0] * tam_bf
  resul = [0] * tam_bf
  for i in range(tam_bf):  # a é o tamanho do filtro de bloom
    soma[i] = 0
    for j in range(len(lista)):
      if lista[j]["BF"][i] == 1:
        soma[i] += 1
    resul[i] = soma[i] - len(lista) / 2

  n = 0
  for i in range(tam_bf):
    if np.absolute(resul[i]) < np.absolute(resul[n]):
      n = i

  if np.absolute(resul[n]) == len(lista) / 2:
    # print("BF iguais")
    no = {}
    for x in range(len(lista)):
      no[IDs[x]['ID']] = lista[x]["BF"]
    return no

  no = NodoArvore()
  no.chave = n
  # print(n)

  name_list_dir = []
  IDs_dir = []
  name_list_esq = []
  IDs_esq = []

  for i in range(len(lista)):
    if lista[i]["BF"][n] == 1:
      name_list_dir.append(lista[i])
      IDs_dir.append(IDs[i])
    else:
      name_list_esq.append(lista[i])
      IDs_esq.append(IDs[i])

  if len(name_list_dir) > group and cont < tam_bf:
    cont += 1
    # print("no direita")
    no.direita = criarMBT(name_list_dir, IDs_dir, cont, tam_bf, group)
  else:
    no.direita = {}
    for x in range(len(name_list_dir)):
      no.direita[IDs_dir[x]['ID']] = name_list_dir[x]["BF"]
  if len(name_list_esq) > group and cont < tam_bf:
    cont += 1
    # print("no esquerda")
    no.esquerda = criarMBT(name_list_esq, IDs_esq, cont, tam_bf, group)
  else:
    no.esquerda = {}
    for x in range(len(name_list_esq)):
      no.esquerda[IDs_esq[x]['ID']] = name_list_esq[x]["BF"]
  return no

#https://gist.github.com/JDWarner/6730747
def dice(im1, im2):
  im1 = np.asarray(im1).astype(np.bool_)
  im2 = np.asarray(im2).astype(np.bool_)

  if im1.shape != im2.shape:
    raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

  # Compute Dice coefficient
  intersection = np.logical_and(im1, im2)
  return 2. * intersection.sum() / (im1.sum() + im2.sum())

def busca(regB, arv):
  simi = 0
  cand_vet = {}
  if hasattr(arv, 'chave'):
  #try:
    k = arv.chave
    if regB[k] == 1:
      #print("entrou na direita")
      return busca(regB, arv.direita)
    else:
      #print("entrou na esquerda")
      return busca(regB, arv.esquerda)

  else:
  #except AttributeError:
    #print("chegou na folha")
    if arv:
      for x in arv:
        simi_int = float(dice(regB,arv[x]))
        cand_vet[x] = simi_int
        if simi_int > simi:
          simi = simi_int
          pos = x
    return Row("out1", "out2", "out3")(pos, simi, cand_vet)

def busca_noindex(regB, lista, id):
  simi = 0
  cand_vet = {}
  for x in range(len(lista)):
    simi_int = float(dice(regB,lista[x]['BF']))
    cand_vet[id[x]['ID']] = simi_int
    if simi_int > simi:
      simi = simi_int
      pos = id[x]['ID']
  return Row("out1","out2","out3")(pos,simi,cand_vet)

def acerto(id1, id2):
  if id1 == id2:
    return 1
  else:
    return 0

def check(check,dice, threshold):
  VP = VN = FP = FN = 0
  if check == 1 and dice >= threshold:
    VP = 1
  elif check == 1 and dice < threshold:
    FN = 1
  elif check == 0 and dice >= threshold:
    FP = 1
  else:
    VN = 1
  return Row("out1","out2","out3","out4")(VP, VN, FP, FN)

def criarDict (key,value):
  dic = {}
  dic[key] = value
  return dic

'''
def busca(regB, arv):
  simi = 0
  cand_vet = {}
  pos = 0
  node = arv  # Start at the root
  while node:  # While the current node is not empty (None or {})
    if hasattr(node, 'chave'):  # If the node is a tree node
      key = node.chave
      if regB[key] == 1:
        node = node.direita  # Move to the right
      else:
        node = node.esquerda  # Move to the left
    else:  # If the node is a leaf node (a dictionary)
      for x in node:
        simi_int = float(dice(regB, node[x]))
        cand_vet[x] = simi_int
        if simi_int > simi:
          simi = simi_int
          pos = x
  return Row("out1", "out2", "out3")(pos, simi, cand_vet)
'''

def calcFN(VP, true_matches):
  FN = true_matches - VP
  return FN

def calcPcom(VP, true_matches):
  pcom = VP / true_matches
  return pcom

def calcRedRate(VP, VN, FP, FN, registros_a, registros_b):
  redRate = 1 - ((VP + VN + FN + FP) / (registros_a * registros_b))
  return redRate

def calcPairQual(VP, VN, FP, FN):
  pairQual = VP / (VP + VN + FP + FN)
  return pairQual

def calcAcc(VP, VN, FP, FN):
  acc = (VP + VN) / (VP + VN + FP + FN)
  return acc

def calcPre(VP, FP):
  if VP == 0:
    pre = 0
  else:
    pre = VP / (VP + FP)
  return pre

def calcFsc(pre, recall):
  fsc = (2 * pre * recall) / (pre + recall)
  return fsc

def calc_time(init):
  # https://www.geeksforgeeks.org/how-to-check-the-execution-time-of-python-script/
  # https://docs.python.org/3/library/timeit.html
  time_exec = time.time() - init
  return time_exec