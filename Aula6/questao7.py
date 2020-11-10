# -*- coding: utf-8 -*-
"""questao7

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NSaHpHs7uUCykuGoQH26_M_LzGf5GejG
"""

import matplotlib.pyplot as plt
import numpy as np

print("Esse programa implementa o método Newton-Raphson de para achar a primeira raiz positiva de f(x) = eˆ(-x) - cos(x*pi/2) \n")

#função que desejamos encontrar a raiz
def f(x):
  return np.exp(-x) - np.cos(np.pi*x/2)

#derivada da funcao f(x) calculada analiticamente
def derivada(x):
  return np.pi*np.sin(np.pi*x/2)/2 - np.exp(-x)

#faz o gráfico da função acima com os valores de x e y definidos
x = np.linspace(0,4,200)
y = f(x)
plt.plot(x,y)
plt.title("Função")
plt.grid(True)
plt.show()

xi = float(input("Olhe o gráfico acima e determine o x desejado \n"))
limite = 100             #número maximo de iterações
precisao = 1*10**-7
precisao_derivada = 1*10**-5
dxi = derivada(xi)

#essa parte nao esta funcionando NS PQ???
if abs(dxi) < precisao_derivada:  # numeros cujas derivadas não são zero, mas serão muito perto e darão uma raiz muito distante
  print("Não é possivel encontrar a raiz por este valor de x")

else:
  Nit = 0  #contador para o número de iterações
  delta = 100
  while abs(delta) > precisao and Nit<limite:
    x = xi - f(xi)/derivada(xi)        # valor próximo x que será usado
    delta = (x - xi)                   #função com o valor de x
    xi = x
    Nit = Nit + 1
  print("O número de iterações : %.d \ne a raiz é %.3f" %(Nit,x))