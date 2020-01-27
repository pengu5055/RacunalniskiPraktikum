from statistics import mean
def nabor_list(s):
    s = s.replace("\n", "") # Filter new line
    data = s.split(",")
    for i in range(len(data)):
        if data[i].isdigit():
            data[i] = int(data[i])

    return data

def rezultati(vhodna, izhodna):
    with open(vhodna, "r", encoding="utf-8") as f:
        data = sorted([nabor_list(line) + [sum(nabor_list(line)[1:])] for line in f.readlines()],
                      key=lambda a: (a[0].split(" ")[1]))


    with open(izhodna, "w+", encoding="utf-8") as f:
        for element in data:
            f.write(",".join([str(a) for a in element]) + "\n")

        + f.write(",".join([str(b) +"0" if len(str(b)) < 4 else str(b) for b in ["POVPRECEN STUDENT"] +
                          [float(round(mean(i[a] for i in data), 2)) for a in range(1, len(data[1]))]]))