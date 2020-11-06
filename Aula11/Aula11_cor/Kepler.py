'''
O programa roda mas não fornece os resultados esperados.
Erro 1: Os passos estão incorretos. Eles deveriam ser n=365, n=24*365 e n=60*24*365 para, respectivamente, 1 dia, 1 hora e 1 minuto no Método de Euler.
Erro 2: O passo para o Método de Runge Kutta está incorreto. Deveriamos ter n=365.
Obs 1: As perguntas da Q.3 não foram respondidas.
Obs 2: A Q.4c não foi respondida.
Obs 3: Gráficos sem label nos eixos.
'''

import matplotlib.pyplot as plt
import numpy as np

#tarefa feita em conjunto pel@s alun@s: Carolina Niklaus Moreira da Rocha Rodrigues e Gabriel Queiroz de Miranda



def drdt(x,y):          #definidas as funções derivada
    G= 6.67*(10**-11)
    Ms= 1.98 *(10**30)
    r= np.sqrt((x[0]**2)+(x[1]**2))
    dvx= -G*Ms*(x[0]/(r**3))
    dvy= -G*Ms*(x[1]/(r**3))
    return np.array([x[2],x[3],dvx,dvy])

def Euler(drdt,x,ti,tf,n):   #definida a função de Euler
    t= np.zeros(n)
    if isinstance (xi,(float, int)):
        x= np.zeros(n,len(xi))
    else:
        neq= len(xi)
        x= np.zeros((n,neq))
        x[0]= xi
        t[0]= ti
        h= (tf-ti)/n
        for k in range(n-1):
            t[k+1]= t[k] + h
            x[k+1]= x[k] + h*drdt(x[k],t[k])
    return t,x

n=86400    # utilizadas as precisões dadas a seguir de minuto, hora e dia respectivamente:60, 3600, 86400
ti= 0
tf= 31536000 
xi= np.array([1.496*(10**11),0,0,2.97*(10**4)])

from rk4 import RungeKutta4
x, t = RungeKutta4(drdt, xi, ti, tf, n)  #alterado o "f" para "drdt"
plt.plot(x[:,0], x[:,1])    #plotado o gráfico
plt.title("Órbita da Terra em volta do Sol")
plt.show() 


#abaixo foi deixada a chamada com a função Euler

#t,x = Euler(drdt,xi,ti,tf,n)    
#plt.plot(x[:,0], x[:,1])    #plotado o gráfico
#plt.title("Órbita da Terra em volta do Sol")
#plt.show()
    

