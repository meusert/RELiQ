import os
import numpy as np
import math

all_fidelities = []
fidelity_steps = 100
error_steps = 20
save_interval = 1

for f1_int in range(0, math.ceil(.5 * fidelity_steps) + 1):
    all_fidelities.append([])
    for f2_int in range(0, math.ceil(.5 * fidelity_steps) + 1):
        all_fidelities[f1_int].append([])
        for error_int in range(0, math.ceil(1 * error_steps) + 1):
            all_fidelities[f1_int][f2_int].append([])


for file in os.listdir("."):
    if file.startswith("fidelities_diagonal") and file.endswith(".npy"):
        fidelities = np.load(file)
        print(fidelities.shape)

        for f1_int in range(0, math.ceil(.5 * fidelity_steps) + 1):
            f1 = int(.5 * fidelity_steps + f1_int) / fidelity_steps
            f2_int = f1_int
            f2 = f1

            for error_int in range(0, math.ceil(1 * error_steps) + 1):
                error = error_int / error_steps

                for fidelity in fidelities[f1_int][error_int]:
                    all_fidelities[f1_int][f2_int][error_int].append(fidelity)

    elif file.startswith("fidelities") and file.endswith(".npy"):
        fidelities = np.load(file)
        print(fidelities.shape)

        for f1_int in range(0, math.ceil(.5 * fidelity_steps) + 1):
            f1 = int(.5 * fidelity_steps + f1_int) / fidelity_steps

            for f2_int in range(0, math.ceil(.5 * fidelity_steps) + 1):
                f2 = int(.5 * fidelity_steps + f2_int) / fidelity_steps

                for error_int in range(0, math.ceil(1 * error_steps) + 1):
                    error = error_int / error_steps

                    for fidelity in fidelities[f1_int][f2_int][error_int]:
                        all_fidelities[f1_int][f2_int][error_int].append(fidelity)
                        if f1_int != f2_int:
                            all_fidelities[f2_int][f1_int][error_int].append(fidelity)


print("Total: " + str(np.array(all_fidelities).shape))

np.save("data/all_fidelities.npy", np.array(all_fidelities))