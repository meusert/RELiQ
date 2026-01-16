import copy
import os.path
from collections import defaultdict
import json
import lzma
from pathlib import Path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Union
from matplotlib import pyplot as plt
import networkx as nx

import numpy as np
from tqdm import tqdm

from env.constants import EVAL_SEEDS
from env.entanglementenv import EntanglementEnv


class StepStats(NamedTuple):
    n: int
    obs: Optional[np.ndarray]
    adj: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: dict
    node_state: Optional[np.ndarray]
    node_aux: Optional[np.ndarray]


class EpisodeStats(NamedTuple):
    steps: List[StepStats]
    aux: Dict[str, Any]


def evaluate(
    env,
    policy,
    episodes,
    steps_per_episode,
    disable_progressbar=False,
    output_dir: Optional[Union[Path, str]] = None,
    output_detailed=False,
    output_node_state_aux=False,
    envs = None,
    policies = None,
    detailed_eval: bool = True,
    load_metrics: bool = False,
):
    previous_metrics = None
    if load_metrics:
        with open(output_dir + "/metrics_temp.json", "r") as f:
            previous_metrics = json.load(f)

    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    episode_stats = []

    if envs is not None:
        envs.insert(0, env)
        policies.insert(0, policy)
    else:
        envs = [env]
        policies = [policy]

    for policy in policies:
        if hasattr(policy, "eval"):
            policy.eval()

    for env in envs:
        if hasattr(env, "set_eval_info"):
            env.set_eval_info(True)

        if isinstance(env.get(), EntanglementEnv) and output_dir is not None:
            if env.get().render_episode >= 0:
                env.get().figure_path = os.path.join(output_dir, "video")
            Path(env.get().figure_path).mkdir(exist_ok=True, parents=True)

            env.get().network.random_neighbors = False
            env.get().network.eval = True

            env.get().network.provided_seeds = EVAL_SEEDS
            env.get().network.seeds = EVAL_SEEDS
            env.get().network.exclude_seeds = []


    # perform evaluation
    # print("Performing Evaluation")
    for ep in tqdm(range(episodes), disable=disable_progressbar):
        for env in envs:
            if isinstance(env.get(), EntanglementEnv):
                env.get().episode = ep
                env.get().figure_index = 0

        if disable_progressbar:
            print("Current episode:" + str(ep))

        step_stats = []
        aux_stats = {}

        obss = []
        adjs = []

        for env in envs:
            obs, adj = env.reset()
            obss.append(obs)
            adjs.append(adj)

        # print(f"Episode {ep} seed was {env.current_topology_seed}")

        for policy in policies:
            # reset all agents
            if hasattr(policy, "reset"):
                policy.reset(1)

            if hasattr(policy, "reset_episode"):
                policy.reset_episode()

        if envs[0].get().render_episode >= 0 and envs[0].get().render_episode != ep:
            continue

        if previous_metrics is not None and len(previous_metrics["average_episode_packet_distance"]) > ep:
            continue

        for step in range(steps_per_episode):
            env_done = False
            infos = {}

            for env_id in range(len(envs)):
                env = envs[env_id]
                policy = policies[env_id]
                obs = obss[env_id]
                adj = adjs[env_id]

                env.pre_step()

                if hasattr(env, "netmon"):
                    node_state = env.netmon.state.detach().cpu().squeeze(0).numpy()
                    if node_state.size == 0:
                        node_state = np.zeros((obs.shape[0], 1))
                else:
                    node_state = np.zeros((obs.shape[0], 1))

                if output_node_state_aux:
                    node_aux = env.get_node_aux()
                else:
                    node_aux = None

                actions = policy(obs, adj)
                next_obs, next_adj, reward, done, info = env.step(actions)

                for key in info:
                    if len(envs) > 1:
                        if key not in infos:
                            infos[key] = 0
                        infos[key] += info[key]
                    else:
                        infos[key] = info[key]

                env_done |= isinstance(env.get(), EntanglementEnv) and env.is_done()

                if env_id == 0:
                    step_stats.append(
                        StepStats(
                            step, obs if detailed_eval else None, adj, actions, reward, done, infos, node_state if detailed_eval else None, node_aux if detailed_eval else None
                        )
                    )

                for policy in policies:
                    # reset done agents
                    if hasattr(policy, "reset"):
                        policy.reset(done)

                obss[env_id] = next_obs
                adjs[env_id] = next_adj

            if step + 1 == steps_per_episode or env_done:
                for env_id, env in enumerate(envs):
                    final_info = {}
                    final_info = env.get_final_info(final_info)

                    for key in final_info:
                        if len(envs) > 1:
                            if key not in infos:
                                if isinstance(final_info[key], list):
                                    infos[key] = []
                                elif isinstance(final_info[key], str):
                                    infos[key] = None
                                else:
                                    infos[key] = 0
                            if isinstance(final_info[key], str):
                                infos[key] = final_info[key]
                            else:
                                infos[key] += final_info[key]
                        else:
                            infos[key] = final_info[key]

        episode_stats.append(EpisodeStats(step_stats, aux_stats))

        if (ep + 1) % 5 == 0:
            eval_metrics = get_eval_metrics(episode_stats, previous_metrics, True)
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True, parents=True)
                with open(output_dir / "metrics_temp.json", "w+") as f:
                    json.dump(eval_metrics, f, indent=4, sort_keys=True, default=str)

    if hasattr(env, "set_eval_info"):
        env.set_eval_info(False)

    eval_metrics = get_eval_metrics(episode_stats, previous_metrics, True)

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "metrics.json", "w+") as f:
            json.dump(eval_metrics, f, indent=4, sort_keys=True, default=str)

    return get_eval_metrics(episode_stats, previous_metrics)


def get_eval_metrics(episode_stats: List[EpisodeStats], previous_metrics, include_lists = False):
    stats_lists = defaultdict(list)

    if previous_metrics is not None:
        for metric in previous_metrics:
            if isinstance(previous_metrics[metric], list):
                stats_lists[metric] = copy.deepcopy(previous_metrics[metric])

    # join stats for each step in each episode
    for episode in episode_stats:
        for step in episode.steps:
            for k, v in step.info.items():
                if isinstance(v, list):
                    stats_lists[k] += v
                else:
                    stats_lists[k].append(v)

            '''
            for r in step.reward:
                stats_lists["reward"].append(r)
            '''

    # calculate mean
    metrics = dict()
    for k, v in stats_lists.items():
        if len(stats_lists[k]) > 0:
            v_arr = np.array(v)
            if isinstance(stats_lists[k][0], (int, float, complex)) and not isinstance(stats_lists[k][0], bool):
                metrics[k + "_mean"] = v_arr.mean()
        else:
            v_arr = np.array([])
            if isinstance(stats_lists[k], (int, float, complex)) and not isinstance(stats_lists[k], bool):
                metrics[k + "_mean"] = float("inf")
        if include_lists:
            metrics[k] = v_arr.tolist()

    return metrics


def save_distance_map_plot(distance_map, filename):
    if len(distance_map) == 0:
        return

    X = np.sort(list(distance_map.keys()))
    Y = np.zeros_like(X, dtype=float)
    Y_err = np.zeros_like(X)
    for i, x in enumerate(X):
        Y_arr = np.array(distance_map[x])
        Y[i] = Y_arr.mean()
        Y_err[i] = Y_arr.std()

    plt.clf()
    plt.plot(X, Y, label="Agent")
    plt.plot(X, X, label="Lower Bound")
    plt.xlabel("Shortest path [steps]")
    plt.ylabel("Agent path [steps]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


def save_packet_location_graph(
    G,
    sum_packets_per_node,
    sum_packets_per_edge,
    num_steps,
    filename,
):
    plt.clf()
    pos = nx.drawing.spring_layout(G, seed=1337)
    # pos = nx.get_node_attributes(G, "pos")
    edge_weight = np.array([data["weight"] for n1, n2, data in G.edges(data=True)])
    nx_edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        width=4,
        edge_color=sum_packets_per_edge / (np.sum(sum_packets_per_edge) * edge_weight),
        edge_cmap=plt.get_cmap("viridis"),
    )
    plt.colorbar(nx_edges, label="Normalized edge utilization")
    nx_nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=sum_packets_per_node / np.sum(sum_packets_per_node),
        cmap=plt.get_cmap("viridis"),
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=dict([(i, i) for i in range(G.order())]),
    )
    plt.colorbar(nx_nodes, label="Normalized node utilization")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.0, edgecolor="white"),
    )
    # remove border around network
    plt.gca().axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


def ddlist():
    """
    Default dict of lists, required for pickle.

    :return: a defaultdict of lists.
    """
    return defaultdict(list)