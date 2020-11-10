import numpy as np
def f(y):
    '''
    Definir a função gaussiana
    retunrn: A função Gaussiana
    '''
    fy = np.exp(-y**2 / 2)
    return fy

def simpson(f,a,b,k):
    '''
    Implementar o método do trapézio
    return: a função tk para calcular a integral
    '''
    deltak = (b - a)/2**k
    i_impar = 1
    i_par = 2
    soma_impar = 0
    soma_par = 0
    while i_impar <= (2**k-1):
        y = a + i_impar*deltak
        fj = f(y)
        soma_impar += fj
        i_impar += 1
    while i_par <= (2**k-1):
        y = a + i_par*deltak
        fj = f(y)
        soma_par += fj
        i_par += 1
    sk = deltak/3 *(f(a) + 4*soma_impar + 2*soma_par +f(b))
    return sk

a = 0
b = 2
k = 10

g = simpson(f, a, b, k)

simp = 1/np.sqrt(2*np.pi) * simpson(f, a, b, k)
print("O valor da integral da função Gaussiana pelo método de Simpson no intervalo de a até b é %.3f usando k igual a %d "%(simp, cont1))
