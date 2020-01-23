s = "+-+---+++"
d = 3
s = list(s)
obdelano = [s[x:x+d] for x in range(0, len(s), d)]
print(obdelano)