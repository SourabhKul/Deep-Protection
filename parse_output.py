required_lines = []
with open('./output100.txt') as fin:
    for line in fin:
        if "Non-adversarial prior obtained at iteration" in line:
            x = line.replace("Non-adversarial prior obtained at iteration ", '')
            x = x.replace(" top", '')
            x = x.replace(" /5", '')
            required_lines.append(x)
        elif "Summary of attack <bound method Attack.name of" in line:
            required_lines.append(line)

with open('output100_parsed.txt', 'w') as f:
    for item in required_lines:
        f.write("%s" % item)