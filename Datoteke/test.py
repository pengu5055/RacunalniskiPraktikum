def prepisi_in_zagrozi(f_in, f_out):
    with open(f_in, "r", encoding="utf-8") as f:
        content = f.read()
        f.close()

    content = content.replace(".", "!")

    with open(f_out, "w+", encoding="utf-8") as f:
        f.write(content.upper())

prepisi_in_zagrozi("in.txt", "out.txt")