import argparse
import numpy as np
import os
import math

from qiskit.quantum_info import gate_error

from swap_function import qiskit_simulated_swap_fidelity_v2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate fidelity of quantum circuits."
    )

    parser.add_argument(
        "--index",
        type=int,
        help="Index",
        default=0,
    )

    parser.add_argument(
        "--diagonal",
        help="Only calculate diagnoal.",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.diagonal:
        filename = "fidelities" + str(args.index) + ".npy"
    else:
        filename = "fidelities_diagonal" + str(args.index) + ".npy"

    if os.path.isfile(filename):
        fidelities = np.load(filename)
        swap_operations_results = fidelities.tolist()
    else:
        swap_operations_results = []
    fidelity_steps = 100
    error_steps = 100
    save_interval = 1

    fidelity_min = .26

    if len(swap_operations_results) > 0 and len(swap_operations_results[0]) > 0 and len(swap_operations_results[0][0]) > 0:
        count = len(swap_operations_results[0][0][0]) + 1
    else:
        count = 1

    while count <= 1:
        print("Step " + str(count))
        for f1_int in range(0, math.ceil((1 - fidelity_min) * fidelity_steps) + 1):
            f1 = int(fidelity_min * fidelity_steps + f1_int) / fidelity_steps
            if len(swap_operations_results) <= f1_int:
                swap_operations_results.append([])

            for f2_int in range(0, math.ceil((1 - fidelity_min) * fidelity_steps) + 1):
                f2 = int(fidelity_min * fidelity_steps + f2_int) / fidelity_steps

                print("Step " + str(count) + ": Calculating for " + str(f1) + " - " + str(f2))

                if len(swap_operations_results[f1_int]) <= f2_int:
                    swap_operations_results[f1_int].append([])

                for error_int in range(0, math.ceil(1 * error_steps) + 1):
                    error = error_int / error_steps
                    if len(swap_operations_results[f1_int][f2_int]) <= error_int:
                        swap_operations_results[f1_int][f2_int].append([])

                    fidelity_value, _ = qiskit_simulated_swap_fidelity_v2(f1, f2, 1 - error, return_circuit=False)

                    if error_int == 0:
                        print("Step " + str(count) + ": Calculating for " + str(f1) + " - " + str(f2) + ": " + str(fidelity_value))

                    swap_operations_results[f1_int][f2_int][error_int].append(fidelity_value)

        if count % save_interval == 0:
            swap_operations_results_np = np.array(swap_operations_results)
            np.save(filename, swap_operations_results_np)

        count += 1

