required_lines = []
files_to_read = ['./experiment-logs/Fixed-Term-Batch-1.txt', './experiment-logs/Fixed-Term-Batch-2.txt', './experiment-logs/Fixed-Term-Batch-3.txt']

for file in files_to_read:
    with open(file) as fin:
        for line in fin:
            if "Begin attack" in  line:
                required_lines.append(line)
            elif "Iteration 0" in line:
                required_lines.append(line)
            elif "Non-adversarial prior obtained at iteration" in line:
                required_lines.append(line)

with open('fixed_term_processed.txt', 'w') as f:
    for item in required_lines:
        f.write("%s" % item)

# 1. "Begin attack" - Attack name
# 2. "Iteration 0" - Iteration number , Loss
# 3. "Non-adversarial prior obtained at iteration" - rank