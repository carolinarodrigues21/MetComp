'''
O programa roda mas não fornece os resultados esperados.
Erro: O valor das constantes G e M estão incorretos.
Obs 1: O gráfico não mostra o zero da função pois o eixo y esta em escala logaritmica e log(0) não está definido. Além disso,
o range do x está excessivamente extenso. Assim, a visualização do comportamento da curva em x=1.48e11 é prejudicada.
É recomendado usar, por exemplo x = np.linspace(1.46e11,1.49e11,200).
Obs 2:  Cuidado com divisões por zero. Caso o valor do chute inicial seja 0 ou 1.5e11 teriamos uma divisão por zero
na função e sua derivada.
'''

# -*- coding: utf-8 -*-
"""pontoslag

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QDuEoIEzi7dgC4MwNdrgYJa1xh2ssrXm
"""

import matplotlib.pyplot as plt
import numpy as np

print("Esse programa implementa o método Newton-Raphson de para achar os pontos de Lagrange do sistema Terra-Sol")

G = 6.614 * 10**-11       #Constante gravitacional
M = 1.89 * 10**30         #Massa do Sol
m = 5.9736 * 10**24       #Massa da Terra
R = 1.5 * 10**11          #Distância entre a Terra e o Sol
omega = 1.992 * 10**-7    #Velocidade angular da Terra e do Satélite

#função que desejamos encontrar a raiz
def f(r):
  return G*M/r**2 - G*m/(R-r)**2 - omega**2 * r

#derivada da funcao f(r) calculada analiticamente
def derivada(r):
  return -G*2*M/r**3 - G*2*m/(R-r)**3 - omega**2

#faz o gráfico da função acima com os valores de x e y definidos
x = np.linspace(0,R,200)
y = f(x)
plt.plot(x,y)
plt.yscale('log')
plt.grid(True)
plt.show()

r = R/2
limite = 100  #número maximo de iterações
precisao = 1*10**-7
Nit = 0  #contador para o número de iterações
delta = 100

while abs(delta) > precisao and Nit<limite:
  x = r - f(r)/derivada(r)       # valor do próximo x que será usado
  delta = (x - r)
  r = x
  Nit = Nit + 1

print("Para r = R/2, o número de iterações é %.d \ne a posição do L1 entre Terra e Sol é %.3f m" %(Nit,x))