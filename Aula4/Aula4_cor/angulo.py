'''
O programa roda mas não fornece resultados esperados para o caso de um ou dois vetores nulos e para os angulos 0 e 180.
Erro: A aluna não impôs uma condição que impede o cálculo do cosseno para o caso de um ou mais vetores possuirem módulo nulo (divisão por zero).
Obs: Por tratar-se de uma variável float, os números no computador são aproximados. 
Portanto, para que seja possível calcularmos o valor exato do cosseno para ângulos paralelos, 
anti-paralelos e perpendiculares, devemos realizar arredondamentos utilizando a função round(), por exemplo.
'''

# -*- coding: utf-8 -*-
"""angulo

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QDuEoIEzi7dgC4MwNdrgYJa1xh2ssrXm
"""
# -*- coding: utf-8 -*-
from math import *
import numpy as np

# este programa quer calcular o angulo entre dois vetores no R3

def angulo(x,y):

    """
    Calcula o angulo entre os vetores de 3 dimensoes x e y
    return o angulo entre os vetores em graus
    """

    escalar = x[0]*y[0] + x[1]*y[1] + x[2]*y[2]   #produto escalar entre os vetores
    modx = sqrt(x[0]**2 + x[1]**2 + x[2]**2)      #módulo do vetor x
    mody = sqrt(y[0]**2 + y[1]**2 + y[2]**2)      #módulo do vetor y
        
    cos = escalar / (modx * mody)                 #define o valor do cosseno entre esses vetores 
    teta = degrees(np.arccos(cos))                #converte o valor em radianos para graus
    return teta

vet1 = np.array([1,1,1])                    #define o vetor 1
vet2 = np.array([1,1,1])                    #define o vetor 2
angulo12 = angulo(vet1,vet2)                      #encontrar o angulo entre os vetores
print("o angulo entre")
print(vet1)
print ("e")
print(vet2)
print("é %d graus" %(angulo12))