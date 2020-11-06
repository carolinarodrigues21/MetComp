'''
[linha 45] Deveria trabalhar com k[i+1] e k[i] pois quando i=0,
k[i-1] = k[-1] = 20 (primeiro teste). Essa também é a causa da
lentidão do programa. ** Nesse caso, precisaria adicionar 1 ao k
final pois o último a ser testado foi o k[i+1].
[linha  53 e 58] 'if' e 'else' desnecessários.
[linha 76] Precisa adicionar 1 ao k final pois o último a ser
comparado foi o k[j+1].

'''
import math as mt
import numpy as np

print("esse programa deseja calcular a integral da Gaussiana")

#função que deseja calcular a integral 
def fun(y):     
  return 1/mt.sqrt(2*mt.pi) * mt.exp(-y**2/2)

#definição do método dos trapézios
def trapezio(fun,a,b,k):
  delta = (b - a)/2**k
  t = 1 #é o j do somátorio (chama de t por ser do trapézio)
  ft=[] #fj da fórmula
  for t in range(1,2**k):
    f = fun(a + t*delta)
    ft.append(f)
    t += 1
  T = delta/2 * (fun(a) + 2*sum(ft) + fun(b))
  return T

#definição do método de Simpson
def simpson(Tki,Tki1):
  return Tki1 + (Tki1 - Tki)/3

a = -2                        #intervalo inferior 
b = 2                         #intervalo superior 
precisao = 10**-6
k = np.array(range(0,21,1))   #como 2**20 é um número grande, logo o k seria menor que esse valor

#achar a ordem de K pela convergência no método dos trapézios
i = 0
for i in k:
  if abs(trapezio(fun, a,b, k[i])- trapezio(fun, a, b ,k[i-1])) < precisao: 
    break
  else:
    i+=1

#achar a ordem de K pela convergência no método de Simpson usando o método dos trapézios
j = 0    
for j in k:
  if j == 0:
   Tkj = trapezio(fun, a,b, k[j])      #esse é o Tk-1 da formula
   Tkj1 = trapezio(fun, a,b, k[j+1])   #esse é o TK da formula
   Sx = simpson(Tkj,Tkj1)
  else: 
    Tkj = trapezio(fun, a,b, k[j])      
    Tkj1 = trapezio(fun, a,b, k[j+1])   
    Sx = simpson(Tkj,Tkj1)
    if abs(Sx - S_anterior) < precisao:
      break
    else:
      j+=1
  S_anterior = Sx   #como o primeiro valor de j é 0, o primeiro S_anterior(Sx-1) será o S0



""" Para otimizar o programa: quando a integral vai de -a até a podemos fazer 2 * integral da função em [0,a].
Contudo, isso funciona apenas em funções pares, em funções impares, como sen(x), a integral de [-a,a] dá 0, logo essa otimização não funcionaria."""
  

print("a integral da Gaussiana calculada pelo método dos trapézios nos intervalo [%.d,%.d] é %.3f e o k que gera a melhor aproximação é %d" %(a,b,trapezio(fun, a,b, k[i]),i))
print("pelo método de Simpson a integral é é %.3f e o k que gera a melhor aproximação é %d" %(Sx,j))
