import copy
import json
import math
import os.path
import textwrap
from enum import Enum
from pathlib import Path

import pickle
import base64

import networkx as nx
import numpy as np
from collections import defaultdict

from gymnasium.spaces import Discrete

from env.environment import EnvironmentVariant, NetworkEnv

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from env.quantum_network import QuantumNetwork, QuantumLink, QuantumRepeater, Entanglement, LinkReservation
from util import one_hot_list

class Metrics:
    def __init__(self):
        self.total_success_resources = 0
        self.total_reached_satellites = 0
        self.total_reached_ground_stations = 0
        self.total_resources = 0
        self.total_success_packets = 0
        self.total_target_packets = 0
        self.total_packets = 0
        self.total_success_fidelity = 0
        self.total_success_path_length = 0
        self.last_successful_link = 0
        self.max_success_path_length = 0
        self.reward = 0
        self.fidelities_arrived = []
        self.gnn_duration = 0
        self.rl_duration = 0


class LinkRequest:
    """
    A data packet.
    """

    def __init__(
        self, id
    ):  # Add a default fidelity_threshold value
        self.creation_step = 0
        self.id = id
        self.now = None
        self.target = None
        self.start = None
        self.time = 0
        self.edge = -1
        self.neigh = None
        self.ttl = None
        self.link = None
        self.shortest_path_weight = None
        self.visited_nodes = None
        self.path = None
        self.breakpoints = [],
        self.fidelity_threshold = QuantumLink.FIDELITY_THRESHOLD
        self.distillations = []

    def reset(
        self,
        start,
        target,
        size,
        ttl,
        shortest_path_weight,
        creation_step,
        fidelity_threshold=QuantumLink.FIDELITY_THRESHOLD,
    ):
        self.creation_step = creation_step
        self.now = start
        self.target = target
        self.start = start
        self.time = 0
        self.edge = -1
        self.neigh = [self.id]
        self.ttl = ttl
        self.link = None
        self.shortest_path_weight = shortest_path_weight
        self.visited_nodes = set([start])
        self.path = [start]
        self.fidelity_threshold = fidelity_threshold  # same as above
        self.breakpoints = []

class RewardType(Enum):
    FIDELITY_WITH_SHORTEST_PATH = 0,
    SHORTEST_PATH = 1,
    FIDELITY_WITH_SHORTEST_PATH_NO_PARTIAL = 2,
    FIDELITY = 3,
    FIDELITY_NO_PARTIAL = 4,

class EntanglementEnv(NetworkEnv):
    """
    Entanglement environment based on the environment by
    Jiang et al. https://github.com/PKU-RL/DGN/blob/master/Routing/routers.py
    used for their DGN paper https://arxiv.org/abs/1810.09202.

    The task is to route packets from random source to random destination nodes in a
    given network. Each agent controls a single packet. When a packet reaches its
    destination, a new packet is instantly created at a random location with a new
    random target.
    """

    def __init__(
        self,
        network: QuantumNetwork,
        n_data,
        infinite_quantum_links=False,
        k=3,
        enable_congestion=True,
        enable_action_mask=False,
        ttl=0,
        no_idle_action: bool = False,
        fixed_request_delay: bool = False,
        fixed_requests: bool = True,
        request_based_observation: bool = False,
        render = -1,
        figure_path = None,
        reward_idle_punishment: float = .75,
        fidelity_choices: int = 1,
        fixed_path_length: int = -1,
        node_observation_ignore_value: int = 0b0000000,
        disable_agent_observation: bool = False,
        fairness: bool = False,
        end_swap_only: bool = True,
        eval_max_entanglements: int = 10,
        min_path_length: int = 1,
        max_path_length: int = 12,
        failure_frequency: int = -1,
        n_pseudo_targets: int = None,
        limit_flooding: bool = False,
        detailed_eval: bool = False,
    ):
        """
        Initialize the environment.

        :param network: a network
        :param n_data: the number of data packets
        :param k: include k neighbors in local observation (only for environment variant WITH_K_NEIGHBORS), defaults to 3
        :param enable_congestion: whether to respect link capacities, defaults to True
        :param enable_action_mask: whether to generate an action mask for agents that does not allow visiting nodes twice, defaults to False
        :param ttl: time to live before packets are discarded, defaults to 0
        """
        super(EntanglementEnv, self).__init__()

        self.detailed_eval = detailed_eval

        self.env_id = 0
        self.n_envs = 1

        self.episode = 0

        self.current_path_lengths = {}

        self.limit_flooding = limit_flooding
        self.flooding_area = []

        self.number_of_paths = 32

        self.policy = None

        self.network = network
        assert isinstance(self.network, QuantumNetwork)

        self.eval_max_entanglements = eval_max_entanglements
        
        self.end_swap_only = end_swap_only
        self.encode_path_information = False

        self.fixed_path_length = fixed_path_length

        self.reward_idle_punishment = reward_idle_punishment
        self.infinite_quantum_links = infinite_quantum_links

        self.min_path_length = min_path_length
        self.max_path_length = max_path_length

        self.fairness = fairness
        self.force_fairness = False

        self.disable_agent_observation = disable_agent_observation

        self.figure_path = None
        self.set_figure_path(figure_path)
        self.figure_index = 0
        self.render_episode = render

        self.n_requests = n_data
        if n_pseudo_targets is None:
            self.n_pseudo_targets = 0
        else:
            self.n_pseudo_targets = max(n_pseudo_targets - self.n_requests, 0)
        self.requests = []
        self.targets = []

        self.no_idle_action = no_idle_action
        self.fixed_request_delay = fixed_request_delay

        self.fixed_requests = fixed_requests
        self.request_based_observation = request_based_observation
        if not self.fixed_requests and self.request_based_observation:
            print(
                "--request-based-observation is only allowed if --fixed-requests is set"
            )
            exit(1)
        self.network.random_neighbors = True

        # optionally include k neighbors in local observation
        self.k = k

        # log information
        self.agent_steps = np.zeros(self.n_requests)
        self.agent_resources = np.zeros(self.n_requests)

        # whether to use random targets or target == 0 for all packets
        self.num_random_targets = self.network.n_nodes
        assert self.num_random_targets >= 0

        self.enable_ttl = ttl > 0
        self.enable_congestion = enable_congestion
        self.ttl = ttl
        self.sum_packets_per_node = None
        self.sum_packets_per_edge = None

        self.done_metrics = Metrics()

        self.disable_neighbor_state = False

        self.encoding = {
            "own": {
                "ignore_own_swap_prob": (node_observation_ignore_value & 0b1000000) > 0,
                "ignore_target_info": (node_observation_ignore_value & 0b0100000) > 0,
                "ignore_cluster_info": (node_observation_ignore_value & 0b0010000) > 0,
            },
            "other": {
                "ignore_target_info": (node_observation_ignore_value & 0b0001000) > 0,
                "ignore_cluster_info": (node_observation_ignore_value & 0b0000100) > 0,
            },
            "edge": {
                "ignore_number_of_links": (node_observation_ignore_value & 0b0000010) > 0,
                "ignore_fidelity": (node_observation_ignore_value & 0b0000001) > 0,
            }
        }

        self.enable_action_mask = True
        self.enable_fine_action_mask = enable_action_mask

        self.action_space = Discrete(
            self.network.neighbor_count + 1, start=0
        )  # {0, 1, 2, 3} using gym action space
        self.action_mask = np.zeros(
            (n_data, self.network.neighbor_count + 1), dtype=bool
        )

        self.eval_info_enabled = False
        self.debug_info_enabled = False

    def get_edge_cost(self, start, end):
        """
        Retrieve the cost of the edge between the start and end nodes.
        """
        edge = self.network.get_edge(start, end)
        # Assuming the cost is the sum of the costs of the links within the edge

        return 1

    def set_figure_path(self, figure_path):
        self.figure_path = figure_path
        if self.figure_path is not None:
            Path(self.figure_path).mkdir(exist_ok=True, parents=True)

    def set_eval_info(self, val):
        """
        Whether the step function should return additional info for evaluation.

        :param val: the step function returns additional info if true
        """
        self.eval_info_enabled = val

    def _calculate_action_mask(self, request, index):
        self.action_mask[index] = 0

        offset = 1
        if self.enable_fine_action_mask:
            if request.now == request.target and self.force_fairness:
                self.action_mask[index] = 1
                self.action_mask[index, 0] = 0
            elif request.now == request.start:
                if self.no_idle_action:
                    self.action_mask[index, 0] = 1
                else:
                    self.action_mask[index, 0] = 0
            else:
                self.action_mask[index, 0] = 1
        else:
            self.action_mask[index, 0] = 1

        if request.now != request.target or not self.force_fairness:
            for edge_i in range(self.network.neighbor_count):
                if edge_i < len(self.network.get_nodes(self.env_id)[request.now].edges):
                    e = self.network.get_nodes(self.env_id)[request.now].edges[edge_i]
                    if self.enable_fine_action_mask:
                        self.action_mask[index, offset + edge_i : offset + (edge_i + 1)] = (
                            self.network.edges[e].get_other_node(request.now)
                            in request.visited_nodes
                        ) or len(self.network.edges[e].links) - self.network.edges[e].get_total_reservations() <= 0
                    else:
                        self.action_mask[index, offset + edge_i : offset + (edge_i + 1)] = 0
                else:
                    self.action_mask[index, offset + edge_i : offset + (edge_i + 1)] = 1

    def reset_request(self, packet: LinkRequest, index: int):
        """
        Resets the given data packet using the settings of this environment.

        :param packet: a data packet that will be reset *in-place*
        """

        # free resources on used edge
        if packet.edge != -1:
            self.network.edges[packet.edge].load -= 1

        if (packet.start is None and packet.target is None) or not self.fixed_requests:
            # reset packet in place
            while True:
                path_length = self.fixed_path_length
                if path_length <= 0:
                    path_length = self.network.agent_generator.integers(low=self.min_path_length, high=min(self.ttl, self.network.diameter, self.max_path_length) + 1)
                self.current_path_lengths[packet.id] = path_length

                valid_choices = []

                start_set = False
                taken_destinations = set([])
                if self.fairness:
                    for request in self.requests:
                        if request.start is not None:
                            start_set = True
                            taken_destinations.add(request.target)
                if start_set:
                    valid_starts = [self.requests[0].start]
                else:
                    valid_starts = self.network.shortest_paths_weights.keys()

                for start in valid_starts:
                    if self.network.nodes[start].source_destination_valid:
                        for end in self.network.shortest_paths_weights[start]:
                            if end not in taken_destinations and self.network.nodes[end].source_destination_valid:
                                if self.network.shortest_paths_weights[start][end] == path_length:
                                    if self.network.topohub_topology is not None:
                                        number_of_paths = 0
                                        for _ in nx.all_simple_paths(self.network.G, start, end, cutoff=10):
                                            number_of_paths += 1
                                            if number_of_paths > self.number_of_paths:
                                                break
                                    else:
                                        number_of_paths = -1
                                    if number_of_paths < 0 or number_of_paths > self.number_of_paths:
                                        valid_choices.append((start, end))

                if len(valid_choices) > 0:
                    pair = valid_choices[self.network.agent_generator.integers(len(valid_choices) - 1)]
                    start = pair[0]
                    target = pair[1]
                    break
                else:
                    print("Error in finding appropriate path")
                    self.number_of_paths /= 2
        else:
            start = packet.start
            target = packet.target

        # to enable the delayed creation of packets
        creation_step = self.network.env_steps
        if self.ttl > 0 and self.network.env_steps > 0 and self.fixed_request_delay:
            creation_step = max(creation_step, packet.creation_step + self.ttl)

        packet.reset(
            start=start,
            target=target,
            size=1,
            ttl=self.ttl,#min(self.ttl, self.fixed_path_length) if (not self.network.eval and self.fixed_path_length >= 0) else self.ttl,
            shortest_path_weight=1,
            creation_step=creation_step,
        )

        if self.policy is not None:
            if hasattr(self.policy, "first_step"):
                self.policy.first_step = True

        if self.enable_action_mask:
            self._calculate_action_mask(packet, index)


    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            Quantum entanglement environment with parameters
            > Network: {self.network.n_nodes} nodes
            > Number of requests: {self.n_requests}
            > Congestion: {self.enable_congestion}
            > Action mask: {self.enable_action_mask}
            > TTL: {self.ttl if self.enable_ttl else "disabled"}\
            """
        )

    def is_done(self):
        return self.network.is_done()

    def determine_flooding_area(self):
        flooding_area = set([])

        x = 1

        for request in self.requests:
            if x == 0:
                node_distances_start = nx.shortest_path_length(self.network.G, request.start)
                node_distances_end = nx.shortest_path_length(self.network.G, request.target)

                for node in self.network.nodes:
                    if (node_distances_start[node.id] if node.id in node_distances_start else float("inf")) + (node_distances_end[node.id] if node.id in node_distances_end else float("inf")) <= self.ttl:
                        flooding_area.add(node.id)
            elif x == 1:
                number_of_paths = 0
                max_number_of_paths = 20
                length = nx.shortest_path_length(self.network.G, request.start, request.target)
                while number_of_paths < max_number_of_paths:
                    for path in nx.all_simple_paths(self.network.G, request.start, request.target, cutoff=length):
                        for node in path:
                            flooding_area.add(node)
                        number_of_paths += 1
                        if number_of_paths >= max_number_of_paths:
                            break
                    length += 1
            else:
                return None

        return flooding_area

    def reset(self, perform_reset=True):
        if perform_reset:
            self.number_of_paths = 32

            self.agent_steps = np.zeros(self.n_requests)
            self.agent_resources = np.zeros(self.n_requests)

            if self.env_id == 0:
                self.network.reset()

                for edge in self.network.edges:
                    # add new load attribute to edges
                    edge.load = 0

            if self.eval_info_enabled:
                self.sum_packets_per_node = np.zeros(self.network.n_nodes)
                self.sum_packets_per_edge = np.zeros(len(self.network.edges))

            self.done_metrics = Metrics()

            # generate random data packets
            self.requests = []
            self.targets = []

            ids = list(range(self.n_requests + self.n_pseudo_targets))
            requests_by_id = {}
            for i in range(self.n_requests):
                if self.n_pseudo_targets > 0:
                    id = ids.pop(self.network.agent_generator.integers(len(ids)))
                else:
                    id = i
                new_request = LinkRequest(id)
                self.reset_request(new_request, i)
                self.requests.append(new_request)
                requests_by_id[new_request.id] = new_request

                '''
                for j in range(self.multi_path - 1):
                    copied_request = copy.deepcopy(new_request)
                    self.reset_request(copied_request, i + j)
                    # copied_request.creation_step = int(self.network.ttl / self.multi_path * j)
                    self.requests.append(copied_request)
                '''

            for i in range(self.n_requests + self.n_pseudo_targets):
                if i in requests_by_id:
                    self.targets.append(requests_by_id[i].target)
                else:
                    while True:
                        target = self.network.topology_generator.integers(len(self.network.nodes))
                        if not target in self.targets:
                            self.targets.append(target)
                            break

            if self.limit_flooding:
                self.network.set_virtual_node_group(self.env_id, self.determine_flooding_area())

            if self.figure_path is not None and self.render_episode >= 0:
                self.render(figure_path=os.path.join(self.figure_path, "episode_start_" + str(self.episode) + ".pdf"), fancy=False, show_packet=False)

        return self._get_observation(), self._get_data_adjacency()

    def render(self, done=None, success=None, figure_path=None, fancy=False, show_packet=True):
        packet_radius = 10

        max_x = -float("inf")
        max_y = -float("inf")
        min_x = float("inf")
        min_y = float("inf")
        for node in self.network.nodes:
            max_x = max(max_x, node.x)
            min_x = min(min_x, node.x)
            max_y = max(max_y, node.y)
            min_y = min(min_y, node.y)

        fig = plt.figure(figsize=(24, 24 / float(max_x - min_x) * (max_y - min_y)))
        ax = plt.gca()

        starts = []
        targets = []
        for i in range(self.n_requests):
            request = self.requests[i]

            starts.append(request.start)
            targets.append(request.target)

        self.network.render(show_plot=False, fig=fig, ax=ax, fancy=fancy, starts=starts, targets=targets)
        
        pos = nx.get_node_attributes(self.network.G, "pos")

        for i in range(self.n_requests):
            request = self.requests[i]

            if request.edge == -1:
                node = self.network.nodes[request.now]
                if request.now in pos:
                    position = pos[request.now]
                else:
                    position = (node.x, node.y)
            else:
                edge = self.network.edges[request.edge]
                start_node = self.network.nodes[edge.start]
                end_node = self.network.nodes[edge.end]
                if edge.start in pos:
                    start_position = pos[edge.start]
                else:
                    start_position = (start_node.x, start_node.y)
                if edge.end in pos:
                    end_position = pos[edge.end]
                else:
                    end_position = (end_node.x, end_node.y)
                position = (
                    (start_position[0] + end_position[0]) / 2,
                    (start_position[1] + end_position[1]) / 2,
                )

            face_color = "red"
            if done is not None and done[i]:
                if success is None or not success[i]:
                    face_color = "fuchsia"

            if show_packet:
                circle = Circle(
                    position, radius=packet_radius, facecolor=face_color, zorder=10
                )
                ax.add_patch(circle)

        ax.axis('off')

        if figure_path is not None:
            plt.box(False)
            if fancy:
                plt.savefig(figure_path, dpi=300,  bbox_inches='tight')
            else:
                plt.savefig(figure_path,  bbox_inches='tight')
        elif self.figure_path is not None:
            plt.savefig(
                os.path.join(
                    self.figure_path,
                    "entanglement_" + "{:02d}_{:08d}".format(self.episode, self.figure_index) + ".png",
                ),  bbox_inches='tight'
            )
            self.figure_index += 1
        else:
            plt.show()
        plt.close()

    def get_nodes_adjacency(self):
        return self.network.get_nodes_adjacency_for_env(self.env_id)

    def get_node_observation(self):
        """
        Get the node observation for each node in the network.

        :return: node observations of shape (num_nodes, node_observation_size)
        """
        obs = []
        requests = self.requests

        for j in range(self.network.n_nodes):
            if not self.request_based_observation:
                ob = []

                # encode target information here
                ob += one_hot_list(j, self.network.n_nodes)

                # edge info
                for k in range(min(len(self.network.get_nodes(self.env_id)[j].edges), self.network.neighbor_count)):
                    edge = self.network.get_nodes(self.env_id)[j].edges[k]

                    other_node = self.network.edges[edge].get_other_node(j)
                    # ob.append(other_node)
                    ob += one_hot_list(other_node, self.network.n_nodes)
                    ob.append(self.network.get_nodes(self.env_id)[other_node].swap_prob)

                    for _ in range(
                        self.network.n_quantum_links - len(self.network.edges[edge].links)
                    ):
                        ob.append(0)
                    for link in self.network.edges[edge].links:
                        ob.append(link.fidelity)

                for _ in range(self.network.neighbor_count - min(len(self.network.get_nodes(self.env_id)[j].edges), self.network.neighbor_count)):
                    ob += one_hot_list(-1, self.network.n_nodes)
                    ob.append(0)

                    for _ in range(self.network.n_quantum_links):
                        ob.append(0)

                obs.append(ob)
            else:
                ob = []

                if not self.encoding["own"]["ignore_own_swap_prob"]:
                    ob.append(self.network.get_nodes(self.env_id)[j].swap_prob)
                else:
                    ob.append(0)

                ob.append(len(self.network.get_nodes(self.env_id)[j].entanglements))

                for pseudo_target in self.targets:
                    if pseudo_target == j:
                        state = 1
                    else:
                        state = 0

                    if not self.encoding["own"]["ignore_target_info"]:
                        ob += [state]
                        # ob += one_hot_list(state, 2)
                    else:
                        ob += [0]

                # edge info
                for k in range(min(len(self.network.get_nodes(self.env_id)[j].edges), self.network.neighbor_count)):
                    edge = self.network.get_nodes(self.env_id)[j].edges[k]
                    other_node = self.network.edges[edge].get_other_node(j)

                    if not self.disable_neighbor_state:
                        # encode target information here

                        for target in self.targets:

                            state = 0
                            if other_node == target:
                                state = 1

                            if not self.encoding["other"]["ignore_target_info"]:
                                ob += [state]
                            else:
                                ob += [0]


                    if not self.encoding["edge"]["ignore_number_of_links"]:
                        ob.append(max(len(self.network.edges[edge].links) - self.network.edges[edge].get_total_reservations(), 0))
                    else:
                        ob.append(0)

                    if len(self.network.edges[edge].links) - self.network.edges[edge].get_total_reservations() > 0:
                        if not self.encoding["edge"]["ignore_fidelity"]:
                            ob.append(
                                self.network.edges[edge].links[len(self.network.edges[edge].links) - 1].fidelity)
                        else:
                            ob.append(0)
                    else:
                        ob.append(0)

                # Add empty observation for non-available edges
                for _ in range(self.network.neighbor_count - min(len(self.network.get_nodes(self.env_id)[j].edges), self.network.neighbor_count)):
                    if not self.disable_neighbor_state:
                        for _ in self.targets:
                            ob += [0]

                    ob.append(0)
                    ob.append(0)

                obs.append(ob)
        return np.array(obs, dtype=np.float32)

    def get_node_aux(self):
        """
        Auxiliary targets for each node in the network.

        :return: Auxiliary targets of shape (num_nodes, node_aux_target_size)
        """
        aux = []
        for j in range(self.network.n_nodes):
            aux_j = []

            if j in self.network.shortest_paths_weights:
                # for routing, it is essential for a node to estimate the distance to
                # other nodes -> auxiliary target is length of shortest paths to all nodes
                for k in range(self.network.n_nodes):
                    if k in self.network.shortest_paths_weights[j]:
                        aux_j.append(self.network.shortest_paths_weights[j][k])
                    else:
                        aux_j.append(float("inf"))
            else:
                for _ in range(self.network.n_nodes):
                    aux_j.append(float("inf"))

            aux.append(aux_j)

        return np.array(aux, dtype=np.float32)

    def get_node_agent_matrix(self):
        """
        Gets a matrix that indicates where agents are located,
        matrix[n, a] = 1 iff agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        node_agent = np.zeros((self.network.n_nodes, self.n_requests), dtype=np.int8)
        for a in range(len(self.requests)):
            node_agent[self.requests[a].now, a] = 1

        return node_agent

    def _get_observation(self):
        obs = []

        requests = self.requests

        if not self.request_based_observation:
            for i in range(self.n_requests):
                ob = []

                if not self.disable_agent_observation:
                    # packet information
                    ob += one_hot_list(requests[i].now, self.network.n_nodes)
                    ob += one_hot_list(requests[i].target, self.network.n_nodes)

                    # packets should know where they are coming from when traveling on an edge
                    ob.append(int(requests[i].edge != -1))
                    if requests[i].edge != -1:
                        other_node = self.network.edges[
                            requests[i].edge
                        ].get_other_node(requests[i].now)
                    else:
                        other_node = -1
                    ob += one_hot_list(other_node, self.network.n_nodes)

                    ob.append(requests[i].time)
                    ob.append(requests[i].id)

                    # edge information
                    for j in range(min(len(self.network.get_nodes(self.env_id)[requests[i].now].edges), self.network.neighbor_count)):
                        edge = self.network.get_nodes(self.env_id)[requests[i].now].edges[j]
                        other_node = self.network.edges[edge].get_other_node(
                            requests[i].now
                        )
                        ob += one_hot_list(other_node, self.network.n_nodes)
                        ob.append(self.network.get_nodes(self.env_id)[other_node].swap_prob)

                        for _ in range(
                            self.network.n_quantum_links - len(self.network.edges[edge].links)
                        ):
                            ob.append(0)
                        for link in self.network.edges[edge].links:
                            ob.append(link.fidelity)

                    # Add empty observation for non-available edges
                    for _ in range(self.network.neighbor_count - min(len(self.network.get_nodes(self.env_id)[requests[i].now].edges), self.network.neighbor_count)):
                        ob += one_hot_list(-1, self.network.n_nodes)
                        ob.append(0)

                        for _ in range(self.network.n_quantum_links):
                            ob.append(0)

                ob_numpy = np.array(ob)

                obs.append(ob_numpy)
        else:
            for i in range(len(requests)):
                request = requests[i]

                ob = []

                # packet information
                ob += one_hot_list(requests[i].id, len(self.targets))
                if self.encode_path_information and self.end_swap_only:
                    links_used = len(requests[i].path) - 1
                    for hop_index in range(links_used):
                        last_node_id = requests[i].path[hop_index]
                        next_node_id = requests[i].path[hop_index + 1]
                        ob.append(self.network.get_nodes(self.env_id)[next_node_id].swap_prob)
                        edge = self.network.edge_node_association[(min(last_node_id, next_node_id), max(last_node_id, next_node_id))]
                        if not edge.dead and len(edge.links) - edge.get_total_reservations() > 0:
                            ob.append(edge.links[len(edge.links) - 1].fidelity)
                        else:
                            ob.append(0)
                    for _ in range(self.ttl - links_used):
                        ob.append(1)
                        ob.append(1)
                else:
                    if requests[i].link is not None:
                        ob.append(requests[i].link.fidelity)
                    else:
                        ob.append(1)


                ob.append(min(len(request.path) - (request.breakpoints[len(request.breakpoints) - 1] if len(request.breakpoints) > 0 else 0) - 1, self.ttl))
                ob.append(len(self.network.get_nodes(self.env_id)[request.target].entanglements))

                if not self.disable_agent_observation:
                    # edge information
                    for j in range(min(len(self.network.get_nodes(self.env_id)[requests[i].now].edges), self.network.neighbor_count)):
                        edge = self.network.get_nodes(self.env_id)[requests[i].now].edges[j]
                        other_node = self.network.edges[edge].get_other_node(requests[i].now)
                        ob.append(self.network.get_nodes(self.env_id)[other_node].swap_prob)

                        ob.append(max(len(self.network.edges[edge].links) - self.network.edges[edge].get_total_reservations(), 0))

                        if len(self.network.edges[edge].links) - self.network.edges[edge].get_total_reservations() > 0:
                            ob.append(self.network.edges[edge].links[len(self.network.edges[edge].links) - 1].fidelity)
                        else:
                            ob.append(0)

                        dest_state = 2
                        if other_node in requests[i].visited_nodes:
                            if other_node == requests[i].start:
                                dest_state = 0
                            else:
                                dest_state = 1
                        elif other_node == requests[i].target:
                            dest_state = 3
                        ob += one_hot_list(dest_state, 6)

                    # Add empty observation for non-available edges
                    for _ in range(self.network.neighbor_count - min(len(self.network.get_nodes(self.env_id)[requests[i].now].edges), self.network.neighbor_count)):
                        ob.append(0)
                        ob.append(0)
                        ob.append(0)
                        ob += one_hot_list(1, 6)

                ob_numpy = np.array(ob)

                obs.append(ob_numpy)

        return np.array(obs, dtype=np.float32)

    def step(self, act):
        if self.network.eval and self.env_id == 0:
            print("Episode step: " + str(self.network.env_steps))

        reward = np.zeros(self.n_requests, dtype=np.float32)
        looped = np.zeros(self.n_requests, dtype=np.float32)
        done = np.zeros(self.n_requests, dtype=bool)
        drop_packet = np.zeros(self.n_requests, dtype=bool)
        success = np.zeros(self.n_requests, dtype=bool)
        target = np.zeros(self.n_requests, dtype=bool)
        blocked = 0
        quantum_looped = 0

        if self.render_episode == self.episode:
            self.render()

        fidelities = []

        delays_arrived = []
        self.agent_steps += 1

        for i in range(self.n_requests):
            request = self.requests[i]

            if request.creation_step > self.network.env_steps:
                self.agent_steps[i] -= 1
                continue

            if request.link is not None:
                request.link.fidelity *= self.network.calculate_decay(request.path[0], request.path[-1])

            if self.eval_info_enabled:
                if request.edge == -1:
                    self.sum_packets_per_node[request.now] += 1

            # select outgoing edge (act == 0 is idle)
            if request.edge == -1 and act[i] > 0:
                chosen_edge = (act[i] - 1)

                if chosen_edge < len(self.network.get_nodes(self.env_id)[request.now].edges):
                    t = self.network.get_nodes(self.env_id)[request.now].edges[chosen_edge]

                    edge = self.network.edges[t]

                    quantum_link_available = False
                    quantum_loop = False
                    selected_quantum_link = None
                    if len(edge.links) > 0:
                        quantum_link_available = True
                        if not self.end_swap_only:
                            selected_link_index = len(edge.links) - 1

                            selected_quantum_link = edge.links[selected_link_index]
                            if request.link is None or not (
                                selected_quantum_link.start == request.link.start
                                and selected_quantum_link.end == request.link.end
                            ):
                                if not self.infinite_quantum_links:
                                    edge.links.remove(selected_quantum_link)
                                    self.network.total_active_quantum_links -= 1
                    if self.end_swap_only:
                        if not (self.env_id, i) in edge.reserved_links:
                            edge.reserved_links[(self.env_id, i)] = LinkReservation(1)

                    self.network.min_active_quantum_links = min(self.network.min_active_quantum_links, len(edge.links))

                    if selected_quantum_link is not None or self.end_swap_only:
                        if not self.end_swap_only:
                            if request.link is not None:
                                current_router = self.network.get_nodes(self.env_id)[request.now]
                                new_link = QuantumLink.swap(request.link, selected_quantum_link, current_router.swap_prob, source=request.now, rng_generator=self.network.swap_generator)
                                request.link = new_link
                            else:
                                request.link = selected_quantum_link
                            if request.link.fidelity == 0:
                                drop_packet[i] = True
                    else:
                        drop_packet[i] = True

                    if not quantum_link_available and not self.end_swap_only:
                        blocked += 1
                    else:
                        self.agent_resources[i] += 1

                        # Without action masking, quantum loops can occur. However, we recommend enabling action masking to prevent them
                        if quantum_loop:
                            quantum_looped += 1

                        request.edge = t
                        request.time = 1

                        old_position = request.now

                        # already set the next position
                        request.now = self.network.edges[t].get_other_node(request.now)

                        if request.now in request.visited_nodes:
                            looped[i] = 1

                        request.visited_nodes.add(request.now)
                        request.path.append(request.now)
                else:
                    # reward[i] -= 1
                    act[i] = 0
            elif act[i] == 0:
                if self.fairness:
                    min_amount = float("inf")
                    max_amount = float("-inf")
                    for request_entry in self.requests:
                        min_amount = min(min_amount, len(self.network.nodes[request_entry.target].entanglements))
                        max_amount = max(max_amount, len(self.network.nodes[request_entry.target].entanglements))
                    if len(self.network.nodes[request_entry.target].entanglements) > min_amount + 5:
                        reward[i] += self.reward_idle_punishment
                    else:
                        reward[i] -= self.reward_idle_punishment
                else:
                    if self.action_mask[i].sum() != self.action_mask[i].shape[0]:
                        reward[i] -= self.reward_idle_punishment
            else:
                drop_packet[i] = True

        if self.render_episode == self.episode:
            self.render()

        if self.eval_info_enabled:
            total_fidelity = 0
            total_path_length = 0

            for i in range(self.n_requests):
                request = self.requests[i]
                if request.link is not None:
                    total_fidelity += request.link.fidelity
                    total_path_length += len(request.path) - 1

            packet_distances = list(
                map(
                    lambda p: self.network.shortest_paths[p.now][p.target] if p.now in self.network.shortest_paths and p.target in self.network.shortest_paths[p.now] else float("inf"),
                    self.requests,
                )
            )

        # then simulate in-flight packets (=> effect of actions)
        for i in range(self.n_requests):
            request = self.requests[i]

            if request.creation_step > self.network.env_steps:
                continue

            if act[i] > 0:
                request.ttl -= 1

            if request.edge != -1:
                request.time -= 1
                # the packet arrived at the destination, reduce load from edge
                if request.time <= 0:
                    request.edge = -1

            drop_packet[i] = drop_packet[i] or (self.enable_ttl and request.ttl <= 0)
            # packets that can't do anything are dropped
            if self.action_mask[i].sum() == self.action_mask[i].shape[0]:
                drop_packet[i] = True

            if request.now in self.targets and request.now != request.target and not self.network.eval:
                drop_packet[i] = True

            # the packet has reached the target (this does not necessarily mean that the end-to-end entanglement can be created
            has_reached_target = (
                request.edge == -1
                and request.now == request.target
            )
            if has_reached_target or drop_packet[i]:
                if self.end_swap_only:
                    for j in range(len(request.path) - 1):
                        start = min(request.path[j], request.path[j + 1])
                        end = max(request.path[j], request.path[j + 1])
                        edge = self.network.edge_node_association[(start, end)]
                        if (self.env_id, i) in edge.reserved_links:
                            del edge.reserved_links[(self.env_id, i)]

                if has_reached_target:
                    target[i] = True

                try_count = 0
                if self.end_swap_only:
                    if has_reached_target:
                        self.agent_resources[i] = 0
                        end_to_end_established = True

                        for j in range(len(request.path) - 1):
                            current = request.path[j]
                            next = request.path[j + 1]
                            edge_taken = None
                            for edge_id in self.network.get_nodes(self.env_id)[current].edges:
                                if self.network.edges[edge_id].start == next or self.network.edges[edge_id].end == next:
                                    edge_taken = self.network.edges[edge_id]
                                    break
                            if edge_taken is not None:
                                if len(request.distillations) > 0:
                                    # Perform distillation
                                    distilled_link = None
                                    if request.distillations[j] > 0:
                                        for _ in range(request.distillations[j] + 1):
                                            if len(edge_taken.links) == 0:
                                                break
                                            link_for_distillation = edge_taken.links[-1]
                                            edge_taken.links.remove(link_for_distillation)

                                            if distilled_link is None:
                                                distilled_link = link_for_distillation
                                            else:
                                                distilled_link = QuantumLink.create_distillation(distilled_link, link_for_distillation, self.network.env_steps * self.network.delta_time)
                                        if distilled_link is not None:
                                            edge_taken.links.append(distilled_link)

                                selected_link_index = len(edge_taken.links) - 1

                                if selected_link_index < 0:
                                    request.link = None
                                    drop_packet[i] = True
                                    end_to_end_established = False
                                    continue

                                selected_quantum_link = edge_taken.links[selected_link_index]
                                if request.link is None or not (
                                        selected_quantum_link.start == request.link.start
                                        and selected_quantum_link.end == request.link.end
                                ):
                                    if not self.infinite_quantum_links:
                                        edge_taken.links.remove(selected_quantum_link)
                                        self.network.total_active_quantum_links -= 1

                                self.agent_resources[i] += 1
                                if request.link is not None:
                                    new_link = QuantumLink.swap(request.link, selected_quantum_link, self.network.nodes[current].swap_prob, source=current, rng_generator=self.network.swap_generator)
                                    request.link = new_link
                                else:
                                    request.link = selected_quantum_link
                                if request.link is None or request.link.fidelity == 0:
                                    if request.link is None:
                                        drop_packet[i] = True
                                    request.link = None
                                    break
                        if not end_to_end_established:
                            request.link = None
                    else:
                        self.agent_resources[i] = 0

                has_reached_target &= request.link is not None and request.link.fidelity > QuantumLink.FIDELITY_THRESHOLD

                if has_reached_target:
                    # Add entanglement to target and source
                    entanglement = Entanglement(request.start, request.target, request.link.fidelity)
                    self.network.nodes[request.start].entanglements.append(entanglement)
                    self.network.nodes[request.target].entanglements.append(entanglement)


                if request.now == request.target:
                    reward[i] = max(request.link.fidelity if request.link is not None else 0, QuantumLink.FIDELITY_THRESHOLD)
                else:
                    reward[i] = 0

                self.done_metrics.reward += reward[i]

                done[i] = True
                success[i] = has_reached_target


                if target[i]:
                    self.done_metrics.total_target_packets += 1

                # insert delays before resetting packets
                if success[i]:
                    self.done_metrics.total_success_packets += 1

                    if self.network.eval and 0 < self.eval_max_entanglements <= self.done_metrics.total_success_packets:
                        self.network.set_done()

                    self.done_metrics.total_success_fidelity += request.link.fidelity
                    self.done_metrics.total_success_path_length += self.agent_steps[i]
                    self.done_metrics.max_success_path_length = max(self.agent_steps[i], self.done_metrics.max_success_path_length)
                    self.done_metrics.last_successful_link = self.network.env_steps
                    self.done_metrics.total_success_resources += self.agent_resources[i]

                    delays_arrived.append(self.agent_steps[i])
                    self.done_metrics.fidelities_arrived.append(request.link.fidelity)

                self.done_metrics.total_packets += 1
                self.done_metrics.total_resources += self.agent_resources[i]

                if request.link is not None:
                    fidelities.append(request.link.fidelity)

                self.agent_steps[i] = 0
                self.agent_resources[i] = 0

        if self.render_episode == self.episode:
            self.render(done=done, success=success)

        for i in range(self.n_requests):
            request = self.requests[i]
            if done[i]:
                self.reset_request(request, i)

        if done.sum() > 0 and not self.network.refresh_rate > 0:
            reachable = 0
            distances = []
            for i in range(self.n_requests):
                request = self.requests[i]
                if request.start in self.network.shortest_available_paths_weights \
                    and request.target in self.network.shortest_available_paths_weights[request.start] \
                    and self.network.shortest_available_paths_weights[request.start][request.target] <= self.ttl:
                    reachable += 1
                    distances.append(self.network.shortest_available_paths_weights[request.start][request.target])
                else:
                    distances.append(-1)

            all_below_fixed_path = True
            for i in range(self.n_requests):
                if 0 <= self.fixed_path_length < distances[i]:
                    all_below_fixed_path = False

            if (reachable == 0 or not all_below_fixed_path) and not self.network.eval:
                self.network.set_done()

        if success.sum() > 0:
            self.network.next_max_env_steps = max(
                self.network.min_env_steps, self.network.env_steps + self.ttl
            )

        obs = self._get_observation()
        adj = self._get_data_adjacency()

        if self.debug_info_enabled:
            info = {
                "fidelity": fidelities,
                "fidelity_arrived": self.done_metrics.fidelities_arrived,
                "delays_arrived": delays_arrived,
                # shortest path ratio in [1, inf) where 1 is optimal
                "looped": looped.sum(),
                "throughput": success.sum(),
                "dropped": (done & ~success).sum(),
                "blocked": blocked,
                "quantum_looped": quantum_looped,
            }
            if self.eval_info_enabled:
                info.update(
                    {
                        "total_fidelity": total_fidelity,
                        "total_path_length": total_path_length,
                        "packet_distances": packet_distances,
                    }
                )
        else:
            if self.detailed_eval:
                info = {
                    "success": success.sum(),
                }
            else:
                info = {}

        return obs, adj, reward, done, info

    def _get_data_adjacency(self):
        """
        Get an adjacency matrix for data packets (agents) of shape (n_agents, n_agents)
        where the second dimension contains the neighbors of the agents in the first
        dimension, i.e. the matrix is of form (agent, neighbors).

        :param data: current data list
        :param n_data: number of data packets
        :return: adjacency matrix
        """
        # eye because self is also part of the neighborhood
        adj = np.eye(self.n_requests, self.n_requests, dtype=np.int8)
        for i in range(self.n_requests):
            adj[i, i] = 1

            '''
            for n in self.requests[i].neigh:
                if n != -1:
                    # n is (currently) a neighbor of i
                    adj[i, n] = 1
            '''
        return adj

    def pre_step(self):
        if self.env_id == 0:
            self.network.pre_step()

            self.network.step()

        for i in range(self.n_requests):
            self._calculate_action_mask(self.requests[i], i)

    def get_final_info(self, info: dict):
        packet_distance_avg = 0

        for request in self.requests:
            if request.start in self.network.shortest_paths_weights and request.target in self.network.shortest_paths_weights[request.start]:
                packet_distance_avg += self.network.shortest_paths_weights[request.start][request.target]

        packet_distance_avg /= self.n_requests
        info["average_episode_packet_distance"] = packet_distance_avg / (self.n_envs if self.network.eval else 1)
        info["average_episode_packets"] = self.done_metrics.total_success_packets
        info["average_episode_packets_target"] = self.done_metrics.total_target_packets

        if self.eval_info_enabled and self.env_id == 0:
            info["transmitted_messages"] = self.network.transmitted_messages / self.network.env_steps

            if self.detailed_eval:
                transmitted_message_load_edges = np.array(list(self.network.transmitted_message_load_edges.values()))
                transmitted_message_load_nodes = np.array(list(self.network.transmitted_message_load_nodes.values()))

                for i in range(0, 41):
                    if len(transmitted_message_load_edges) > 0:
                        info["transmitted_messages_load_edge_p" + f"{i * 2.5:0>5.1f}"] = np.quantile(transmitted_message_load_edges, i * 0.025) / self.network.env_steps
                    if len(transmitted_message_load_nodes) > 0:
                        info["transmitted_messages_load_nodes_p" + f"{i * 2.5:0>5.1f}"] = np.quantile(transmitted_message_load_nodes, i * 0.025) / self.network.env_steps

            if self.detailed_eval:
                G = self.network.G.copy()
                pos = {}
                for node in self.network.nodes:
                    pos[node.id] = (node.x, node.y)
                nx.set_node_attributes(G, pos, "pos")

                nx.set_edge_attributes(G, self.network.transmitted_message_load_edges, "transmitted_messages_load_edges")
                nx.set_node_attributes(G, self.network.transmitted_message_load_nodes, "transmitted_messages_load_nodes")

                info["topology"] = base64.b64encode(pickle.dumps(G)).decode("utf-8")

        for request in self.requests:
            info["average_episode_packets_" + str(self.env_id) + "_" + str(request.id)] = len(self.network.nodes[request.target].entanglements)

        if self.done_metrics.total_success_packets > 0:
            info["resources_for_and_per_success"] = self.done_metrics.total_success_resources / self.done_metrics.total_success_packets / (self.n_envs if self.network.eval else 1)
            info["resources_per_success"] = self.done_metrics.total_resources / self.done_metrics.total_success_packets / (self.n_envs if self.network.eval else 1)
        info["max_episode_path_length_" + str(self.env_id)] = self.done_metrics.max_success_path_length
        info["max_episode_path_length_in_diameter_" + str(self.env_id)] = (
            self.done_metrics.max_success_path_length / self.network.get_diameter()
        )
        info["last_episode_success_" + str(self.env_id)] = self.done_metrics.last_successful_link
        if self.done_metrics.total_success_packets > 0:
            info["average_episode_path_length_" + str(self.env_id)] = (
                self.done_metrics.total_success_path_length / self.done_metrics.total_success_packets
            )
            info["average_episode_path_length_in_diameter_" + str(self.env_id)] = (
                self.done_metrics.total_success_path_length
                / self.done_metrics.total_success_packets
                / self.network.get_diameter()
            )
            info["total_episode_fidelity"] = (
                self.done_metrics.total_success_fidelity / self.done_metrics.total_success_packets / (self.n_envs if self.network.eval else 1)
            )
        if self.done_metrics.total_packets > 0:
            info["episode_success_rate"] = (
                self.done_metrics.total_success_packets / self.done_metrics.total_packets / (self.n_envs if self.network.eval else 1)
            )
            info["resources_per_packet"] = self.done_metrics.total_resources / self.done_metrics.total_packets / (self.n_envs if self.network.eval else 1)
            info["average_reward"] = self.done_metrics.reward / self.done_metrics.total_packets / (self.n_envs if self.network.eval else 1)

        if self.eval_info_enabled:
            all_fidelities = np.array(self.done_metrics.fidelities_arrived)
            all_fidelities = -np.sort(-all_fidelities)
            for i in range(0, 41):
                if len(all_fidelities) > 0:
                    info["fidelities_" + str(self.env_id) + "_p" + f"{i * 2.5:0>5.1f}"] = np.quantile(all_fidelities, i * 0.025)
                else:
                    info["fidelities_" + str(self.env_id) + "_p" + f"{i * 2.5:0>5.1f}"] = -1
            for i in range(0, 151, 1):
                info["fidelities_" + str(self.env_id) + "_" + f"{i:03d}"] = all_fidelities[i] if len(all_fidelities) > i else -1
            info["complexity_path_finding"] = self.done_metrics.rl_duration
            info["complexity_monitoring"] = self.done_metrics.gnn_duration

        return info

    def get_num_agents(self):
        return self.n_requests

    def get_num_nodes(self):
        return self.network.n_nodes
