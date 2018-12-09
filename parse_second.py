import numpy as np 
from matplotlib import pyplot as plt
img_id, attack = 0, ""
iteration  = 0
attacks_dict = {}
loss = 0.0
with open('./fixed_term_processed.txt') as fin:
    for line in fin:
        if "Begin attack" in line:
            # Reset attack and image id
            attack = line.replace("Begin attack ", "")
            attack = attack.rstrip()
            img_id = 0
            attacks_dict[attack] = []
        elif "Iteration 0" in line:
            n = line.replace("Iteration ", '')
            n = n.replace("    Loss", '')
            iteration = int(n.split(" ")[0])
            loss = float(n.split(" ")[1])
            attacks_dict[attack].append([attack, img_id, iteration, loss, 0])
            if iteration == 1700:
                img_id += 1
        elif "Non-adversarial prior obtained at iteration" in line:
            x = line.replace("Non-adversarial prior obtained at iteration ", '')
            x = x.replace(" top", '')
            x = x.replace(" /5", '')
            attacks_dict[attack][-1][-1] = int(x.split(" ")[1])


# Order
# attack, image_id, iteration, loss, rank

# Attack names
keys = attacks_dict.keys()
iter_correct = []
# Accuracy at different Iterations
specific_iterations = [5, 10, 15, 20, 25, 30, 34]
for key in keys:
    print ("Attack - ", key)
    if key == "GaussianBlurAttack":
        continue
    for ite in specific_iterations:
        le = len(attacks_dict[key])
        correct = 0
        total = 0
        i = ite
        while i < le:
            total += 1
            rank = attacks_dict[key][i][-1]
            if rank > 0 and rank < 5:
                correct += 1
            i += 35
        print ("Iteration ", ite * 50, " Correct ", correct, " out of ", total)
        iter_correct.append(correct)
    plot = plt
    # plot.plot(np.arange(iter_correct),iter_correct)
    # plot.show()

# Underfit/Fail and Overfit

for key in keys:
    overfit = 0
    underfit = 0

    print ("Attack - ", key)
    if key == "GaussianBlurAttack":
        continue
    i = 0
    le = len(attacks_dict[key])
    
    total = le/35
    while i < (le/35)-1:
        img_ranks = np.array(attacks_dict[key][i*35:(i+1)*35])[:,4]
        img_ranks = img_ranks.astype(np.int)
        loss_ranks = np.array(attacks_dict[key][i*35:(i+1)*35])[:,3]
        loss_ranks = loss_ranks.astype(np.float)
        i += 1
        count = np.count_nonzero(img_ranks)
        if count > 0 and img_ranks[-1] == 0:
            overfit += 1
        elif count == 0 and img_ranks[-1] == 0:
            underfit += 1
    print ("Overfit ", overfit," and Underfit/Fail ", underfit, " out of ", total)