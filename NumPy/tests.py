import numpy as np

def celotni_tok(tabela):
    return np.sum(np.prod(tabela.T, axis=1), axis = 0)

def celotni_tok_po_vklopu_realno_stena(tabela, sigma, E, t, v_max):
    for index in range(len(tabela[1])):
        nova = tabela[1, index] + ((E * tabela[2, index]) / sigma) * np.log(np.exp(sigma * t) + 1)
        if abs(nova) < v_max:
            tabela[1, index] = nova
        else:
            tabela[1, index] = 0

    return celotni_tok(tabela)


print(celotni_tok_po_vklopu_realno_stena(np.array([[1, 2], [-1, 2], [1, 0.1]]), 100, -10, 3, 2))