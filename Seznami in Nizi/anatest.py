s1 = [2, 4]
s2 = [1, 3]

def vsi_vecji(s1, s2):
    if len(s1) and len(s2) == 0:
        return True
    elif min(s1) >= max(s2):
        return True
    else:
        return False

#print(vsi_vecji([5, 8], [1, 2, 4]))
for i in range(0,5):
    for r in range(1,6):
        print("I: " + str(i), "R: " + str(r))