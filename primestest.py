def je_prastevilo(a):
    if a == 1 or a == 0:
        return a == 2
    elif a % 2 == 0:
        return False
    else:
        for i in range(2, int(a/2)):
            if (a % i) == 0:
                return False

        return True


def prastevilo(n):
    c = 0
    i = 0
    while c < n:
        i += 1
        if je_prastevilo(i):
            c += 1
    return i


def naslednje_prastevilo(n):
    i = 1
    while prastevilo(i) <= n:
        i += 1
    return prastevilo(i)


def prime_sum(n):
    total = 0
    c = str(prastevilo(n))
    print(c)
    for i in range(len(c)):
        total += int(c[i])
    return total


def prvo_prastevilo_z_vsoto_stevk_vsaj(n):
    i = 1
    while prime_sum(i) < n:
        i += 1
    return prastevilo(i)

print(prvo_prastevilo_z_vsoto_stevk_vsaj(8))