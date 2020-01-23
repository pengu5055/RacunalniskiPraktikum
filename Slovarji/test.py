def narcisoidi(s):
    n = set()
    for i in s:
        if i in s[i]:
            n.add(i)

    return n
# =====================================================================@021516=
# 2. podnaloga
# Sestavite funkcijo `ljubljeni`, ki sprejme slovar zaljubljenih in vrne
# _množico_ tistih, ki so ljubljeni.
# =============================================================================
def ljubljeni(s):
    l = set()
    for i in s:
        for r in s[i]:
            l.add(r)

    return l
# =====================================================================@021517=
# 3. podnaloga
# Sestavite funkcijo `pari`, ki sprejme slovar zaljubljenih in vrne _množico_
# vseh parov, ki so srečno zaljubljeni. Vsak par naj se pojavi samo enkrat in
# sicer tako, da sta zaljubljenca našteta po abecedi. Na primer, če sta Ana in
# Bine zaljubljena, dodamo par `('Ana', 'Bine')`.
# =============================================================================
#def pari(s):
#    x = 0
#    p = set()
#    ljub = list(ljubljeni(s))
#    print(ljub)
#    for oseba in s:
#        for i in range(len(ljub)):
#            print(s[oseba])
#            for q in range(len(s[oseba])):
#                t = list(s[oseba])
#                print("t: " + t[q])
#                print("ljub: " + ljub[i])
#                if t[q] == ljub[i]:
#                    p.add((oseba, ljub[i]))
#
#    return p

def pari(s):
    p = set()
    for u, i in s.items():
        for q in i:  # q je oseba ki jo mara i
            print("q " + q)
            print(s[q])
            print("u : " + u)

            if u in s[q]:
                x = sorted([u, q])
                print("X: "+ str(x))
                p.add((x[0], x[1]))
                x.clear()
    return p

s =    {
        'Ana': {'Bine', 'Cene'},
        'Bine': set(),
        'Cene': {'Bine', 'Ana'},
        'Davorka': {'Davorka'},
        'Eva': {'Bine'}
    }
s2 = {
    'Ana': {'Bine', 'Cene'},
    'Bine': {'Ana'},
    'Cene': {'Bine'},
    'Davorka': {'Davorka'},
    'Eva': {'Bine'}
}

def kdo_ljubi(oseba1, s):
    t = set()
    for u, i in s.items():
        x = list(i)
        for q in x:
            if oseba1 == q:
                #c = sorted([u, q])
                #t.add((c[0], c[1]))
                #x.clear()
                t.add(u)

    return t


def ustrezljivi(oseba, zaljubljeni):
    pos = 0
    ust = kdo_ljubi(oseba, zaljubljeni)
    x = list(ust)

    for i in x:
        print(i)
        y = list(kdo_ljubi(i, zaljubljeni))
        for e in range(len(y)):
            ust.add(y[e])
            pos += 1
    for i in x:
        print(i)
        y = list(kdo_ljubi(i, zaljubljeni))
        for e in range(len(y)):
            ust.add(y[e])
            pos += 1
    return ust

print(ustrezljivi("Ana", s2))