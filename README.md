# RELiQ: Scalable Entanglement Routing via Reinforcement Learning in Quantum Networks

RELiQ is a reinforcement learning framework for scalable entanglement routing in quantum networks. This repository contains the research code for the paper:

**Meuser, T., Weil, J., Lahiri, A., & Paraschiv, M. (2026). RELiQ: Scalable Entanglement Routing via Reinforcement Learning in Quantum Networks. *IEEE Transactions on Communications*, 74, 1860-1875. DOI: [10.1109/TCOMM.2025.3640083](https://doi.org/10.1109/TCOMM.2025.3640083)**

**Note:** This repository is based on the [graph-marl](https://github.com/jw3il/graph-marl) repository, which provides the foundation for multi-agent reinforcement learning in graphs.

## Overview

RELiQ uses deep reinforcement learning (specifically DQN) combined with graph neural networks (GNNs) to learn efficient entanglement routing policies in quantum networks. The framework models quantum networks as graphs where nodes represent quantum repeaters and edges represent quantum links with associated fidelities. The agent learns to route entanglement requests while considering link quality, resource availability, and network topology.

## Training

To train a RELiQ model, use the following command:

```bash
python -u src/main.py \
  --use-future-rewards \
  --netmon-agg-type=sage \
  --no-idle-action \
  --request-based-observation \
  --fixed-requests \
  --action-mask \
  --disable-progressbar \
  --total-steps=15_000_000 \
  --step-between-train=200 \
  --step-before-train=100_000 \
  --netmon \
  --model=dqn \
  --device=cuda \
  --capacity=100_000 \
  --min-path-length=1 \
  --output-dir=runs_quantum \
  --comment=RELiQ
```

**Note:** The `--device=cuda` flag should only be used if a GPU is available. If you don't have a GPU, omit this flag (the code will default to CPU) or use `--device=cpu`.

## Evaluation

To evaluate a trained model, use:

```bash
python -u src/main.py \
  --model-load-path=models/RELiQ.pt \
  --request-based-observation \
  --no-idle-action \
  --use-realistic-decay \
  --action-mask \
  --fixed-requests \
  --eval \
  --eval-output-dir=eval/reliq \
  --detailed-eval-logs \
  --disable-progressbar
```

## Baselines

RELiQ includes several baseline routing algorithms for comparison. To run a baseline, use the `--policy` flag along with the evaluation flags:

- **QPath**: 
  ```bash
  python -u src/main.py --policy=qpath --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/qpath --detailed-eval-logs --disable-progressbar
  ```

- **QLeap**: 
  ```bash
  python -u src/main.py --policy=qleap --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/qleap --detailed-eval-logs --disable-progressbar
  ```

- **Greedy Entanglement Routing (GER)**: 
  ```bash
  python -u src/main.py --policy=ger --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/ger --detailed-eval-logs --disable-progressbar
  ```

- **Modified Greedy Entanglement Routing (MGER)**: 
  ```bash
  python -u src/main.py --policy=mger --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/mger --detailed-eval-logs --disable-progressbar
  ```

- **Local Best Effort Routing (LBER)**: 
  ```bash
  python -u src/main.py --policy=lber --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/lber --detailed-eval-logs --disable-progressbar
  ```

- **NoN Local Best Effort Routing (NLBER)**: 
  ```bash
  python -u src/main.py --policy=nlber --use-realistic-decay --fixed-requests --eval --eval-output-dir=eval/nlber --detailed-eval-logs --disable-progressbar
  ```

## Requirements

The codebase requires Python 3.x and the following key dependencies:
- PyTorch
- NetworkX
- NumPy
- PyTorch Geometric (for graph neural network operations)

All dependencies can be installed using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Note:** Make sure to install the correct PyTorch version for your system. If you plan to use GPU acceleration, install the CUDA-enabled version of PyTorch from the [official PyTorch website](https://pytorch.org/get-started/locally/). For CPU-only usage, install the CPU version.

## Citation

If you use RELiQ in your research, please cite:

```bibtex
@ARTICLE{11275902,
  author={Meuser, Tobias and Weil, Jannis and Lahiri, Aninda and Paraschiv, Marius},
  journal={IEEE Transactions on Communications}, 
  title={RELiQ: Scalable Entanglement Routing via Reinforcement Learning in Quantum Networks}, 
  year={2026},
  volume={74},
  number={},
  pages={1860-1875},
  keywords={Qubit;Routing;Network topology;Quantum computing;Topology;Quantum entanglement;Quantum mechanics;Repeaters;Reinforcement learning;Graph neural networks;Quantum networks;deep reinforcement learning;graph neural networks},
  doi={10.1109/TCOMM.2025.3640083}
}
```

## Acknowledgement

The authors acknowledge the financial support by the Federal Ministry of Research, Technology and Space (BMFTR) of Germany in the project "Open6GHub" (grant number: 16KISK014).

---

*Note: This README was generated by an AI assistant based on the codebase structure and user-provided information.*
