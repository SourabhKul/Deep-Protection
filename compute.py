import numpy as np 
import matplotlib.pyplot as plt

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
