def naslednji_clen(n):
    if n % 2 == 0:
        return n/2
    else:
        return 3*n + 1

def dolzina_zaporedja(n):
    c = 1
    while n != 1:
        n = naslednji_clen(n)
        c += 1

    return c

print(dolzina_zaporedja(199))