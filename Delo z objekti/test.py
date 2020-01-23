def sestej_stevke(stevke):
    t = []
    for i in range(len(stevke)):
        t.append(stevke[i]+stevke[i+1])

    return t


print(sestej_stevke([2, 0, 0, 4]))
