required_lines = []
with open('./experiment-logs/NextBatchAttacks.txt') as fin:
    for line in fin:
        if "Non-adversarial prior obtained at iteration" in line:
            x = line.replace("Non-adversarial prior obtained at iteration ", '')
            x = x.replace(" top", '')
            x = x.replace(" /5", '')
            required_lines.append(x)
        elif "Summary of attack " in line:
            required_lines.append(line)

with open('newbatch_parsed.txt', 'w') as f:
    for item in required_lines:
        f.write("%s" % item)