required_lines = []
files_to_read = ['./experiment-logs/Fixed-Term-Batch-1.txt', './experiment-logs/Fixed-Term-Batch-2.txt', './experiment-logs/Fixed-Term-Batch-3.txt']

for file in files_to_read:
    with open(file) as fin:
        for line in fin:
            if "Non-adversarial prior obtained at iteration" in line:
                x = line.replace("Non-adversarial prior obtained at iteration ", '')
                x = x.replace(" top", '')
                x = x.replace(" /5", '')
                required_lines.append(x)
            elif "Summary of attack " in line:
                required_lines.append(line)

with open('fixed_term_parsed.txt', 'w') as f:
    for item in required_lines:
        f.write("%s" % item)

# files_to_read = ['./experiment-logs/output_clean.txt']
# for file in files_to_read:
#     with open(file) as fin:
#         for line in fin:
#             if "Non-adversarial prior obtained at iteration 1700" in line:
#                 x = line.replace("Non-adversarial prior obtained at iteration 1700", '')
#                 x = x.replace(" top", '')
#                 x = x.replace(" /5", '')
#                 required_lines.append(x)
#             # elif "Summary of attack " in line:
#             #     required_lines.append(line)

# with open('clean_term_parsed.txt', 'w') as f:
#     for item in required_lines:
#         f.write("%s" % item)