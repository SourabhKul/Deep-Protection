import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
means, std, variance = [], [], []
iterations = []
ranks = []

# attacks = ["Gradient Sign", "Additive Gaussian Noise", "Blended Uniform Noise", "Gaussian Blur", "NewtonFool", \
#      "Gradient Sign", "DeepFoolAttack", "CarliniWagnerL2Attack", "SaltAndPepperNoiseAttack"]
with open('./newbatch_parsed.txt') as fin:
    for line in fin:
        if "Summary of attack" in line:
            # plt.hist(ranks, bins=5)
            # plt.show()
            iterations = np.array(iterations)
            means.append(np.average(iterations))
            std.append(np.std(iterations))
            variance.append(np.var(iterations))
            iterations = []
            ranks = []
        else:
            iterations.append(int(line.split(" ")[0]))
            ranks.append(int(line.split(" ")[1]))

print (means)
print (std)
print (variance)

# lines = []
# with open('./fixed_term_processed_1700.txt') as fin:
#     for line in fin:
#         if "Begin attack" in line:
#             # plt.hist(ranks, bins=5)
#             # plt.show()
#             c = Counter(lines)
#             print (c[1], len(lines), line)
#             lines = []
#         else:
#             lines.append(int(line))
# c = Counter(lines)
# print (c[1], len(lines))

# f = open("./clean_term_parsed.txt")
# lines = f.readlines()
# lines = [int(i) for i in lines]
# c = Counter(lines)
# print (c[1], len(lines), "Clean")
