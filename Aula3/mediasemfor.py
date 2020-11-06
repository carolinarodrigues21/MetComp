# -*- coding: utf-8 -*-
import numpy as np
from numpy import loadtxt, array
from math import sqrt

imp = loadtxt ("media.txt") #importa o arquivo com a lista de notas dos alunos
n = array(imp,int) #transforma a lista imp em um array com numeros inteiros
quadrado = n**2 #lista com o quadrado de cada numero

mi = sum(n)/len(n) #cálculo da média da turma: soma dos valores da lista dividido pelo tamanho da lista
eqm = np.mean(n) #calcula a média usando a ferramenta do numpy

desvio = sqrt(1/(len(n)-1) * sum(quadrado) - (len(n)/(len(n)-1) * mi**2)) #calcula o desvio padrão das notas
eqs = np.std(n) #calcula o desvio com a ferramenta do numpy

aprovados = 0
reprovados = 0 
for x in n:   # o interator x vai percorrer a toda a lista n para encontrar quantas notas se encaixam na situação abaixo
    if x >= 7:
        aprovados +=1 
    elif x < 3:
       reprovados +=1

print("essa turma tem %d alunos, a média de notas é %.1f e o desvio padrão é %.1f." %(len(n), mi, desvio))
print("a média com np.mean é %.1f" %(eqm))
print("o desvio com np.std é %.1f" %(eqs))

print("o número de alunos aprovados foi: %d" %(aprovados))
print("o número de alunos reprovados foi: %d" %(reprovados))
