import math

X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y = [0, 0, 0, 1]

for i in range(4):
    print(X[i], y[i])

w1 = 0.2
w2 = 0.56
bias = 0.15

def sigmoid(x):
    return 1/ (1+ math.exp(-x))


taxa_aprendizagem = 0.1

for epoca in range (50000): #condicao de parada qualquer
    for i in range(4): #para cada s:t

        x1 = X[i][0]
        x2 = X[i][1]
        target = y[i]

        y_in = x1*w1 + x2*w2 + bias
        saida = sigmoid(y_in)

        print('===========================')
        
        print(X[i], saida)

        print('w1 = ', w1)
        print('w2 = ', w2)
        print('bias = ', bias)

        sinal_erro = target - saida

        print('target: ', target)
        print('sinal de erro: ', target,'-', saida, '=', sinal_erro)
        #valor_instantaneo_erro = 1/2 * sinal_erro * sinal_erro

        gradiente = sinal_erro * saida *  (1 - saida)

        #definindo pesos New
        w1 = w1 + taxa_aprendizagem*gradiente*x1
        w2 = w2 + taxa_aprendizagem*gradiente*x2
        bias = bias + taxa_aprendizagem*gradiente

        print('new w1 = ', w1)
        print('new w2 = ', w2)
        print('new bias = ', bias)


