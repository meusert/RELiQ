import math
import time

import numpy as np
import heapq
import copy
import networkx as nx
import torch
from gymnasium.spaces import Discrete

from env.entanglementenv import EntanglementEnv
from env.quantum_network import QuantumNetwork, Edge, QuantumLink, LinkReservation


class EpsilonGreedy:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._enable_action_mask = (
            hasattr(self._env, "enable_action_mask") and self._env.enable_action_mask
        )
        self._model = model
        self._action_space = action_space
        self._args = args
        self._epsilon = args.epsilon
        self._step = 0
        self._epsilon_tmp = None

        self.epsilon_based_refresh_rate = self._env.get().network.refresh_rate < 0 if hasattr(self._env.get(), "network") and hasattr(self._env.get().network, "refresh_rate") else False


    def __call__(self, obs, adj):
        self._step += 1
        # first dimension is number of agents
        actions = np.zeros(obs.shape[0], dtype=np.int32)

        if self._env.eval_info_enabled:
            start = time.time()

        with torch.no_grad():
            device = next(self._model.parameters()).device
            obs = (
                torch.tensor(obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            adj = (
                torch.tensor(adj, dtype=torch.float32)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            # run our model
            q_values = self._model(obs, adj)
            # squeeze batch dimension, our batch size is 1
            q_values = q_values.cpu().squeeze(0).detach().numpy()

            if self._enable_action_mask:
                q_values[self._env.action_mask.nonzero()] = float("-inf")

        # epsilon-greedy action selection
        random_actions = np.random.rand(actions.shape[0], self._action_space)
        if self._enable_action_mask:
            random_actions[self._env.action_mask.nonzero()] = float("-inf")
        random_actions = np.argmax(random_actions, axis=1)
        random_filter = np.random.rand(actions.shape[0]) < self._epsilon
        actions = (
            np.argmax(q_values, axis=-1) * ~random_filter
            + random_filter * random_actions
        )

        # instead of having step as parameter we should log the steps to track
        # decay from https://github.com/PKU-RL/DGN/blob/master/Routing/routers_regularization.py
        if (
            self._epsilon > 0
            and self._step > self._args.step_before_train
            and self._step % self._args.epsilon_update_freq == 0
        ):
            self._epsilon *= self._args.epsilon_decay
            if self._epsilon < 0.01:
                self._epsilon = 0.01

            if self.epsilon_based_refresh_rate:
                self._env.get().network.refresh_rate = 1 - self._epsilon

        if self._env.eval_info_enabled:
            end = time.time()
            self._env.done_metrics.rl_duration += end - start

        return actions

    def eval(self):
        # remember epsilon and switch to greedy policy
        self._eps_tmp = self._epsilon
        self._epsilon = 0

    def reset(self, agents_to_reset):
        """
        Resets agents according to the given boolean tensor.

        :param agents_to_reset: agents to reset of shape (batch_size, n_agents). A value
                                of 1 indicates that the agent's state should be reset.
        """
        if hasattr(self._model, "state") and self._model.state is not None:
            self._model.state = self._model.state * ~torch.tensor(
                agents_to_reset, dtype=bool, device=self._model.state.device
            ).unsqueeze(-1)

    def train(self):
        # switch back to old epsilon
        if self._epsilon_tmp is not None:
            self._epsilon_tmp = None
            self._epsilon = self._epsilon_tmp


class ShortestPath:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        assert isinstance(env.get(), EntanglementEnv)
        self._n_agents = env.get_num_agents()
        self._model = model
        self._action_space = action_space
        self._args = args
        self.static_shortest_paths = False
        self.network = None

    def reset_episode(self):
        self.network = None

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_requests, dtype=np.int32)

        if self.static_shortest_paths:
            # create shortest paths at the very beginning, then use them
            if self.network is None:

                self.network = copy.deepcopy(self._env.network)

            network = self.network
        else:
            # always use latest shortest paths
            network = self._env.network

        for i in range(self._env.n_requests):
            packet = self._env.requests[i]
            current_node = packet.now
            target_node = packet.target

            if current_node == target_node:
                act[i] = 0
                continue

            # first index is the source and last index is the target
            next_node = network.shortest_paths[current_node][target_node][1]

            for index, j in enumerate(network.nodes[current_node].edges):
                current_edge = network.edges[j]
                # if current edge is the desired one we choose this edge
                # case 1: edge.start is the current node itself and edge.end is the target node
                if current_edge.get_other_node(packet.now) == next_node:
                    act[i] = index + 1
                    break

        return act


#####################################
#                                   #
#     Benchmark Policies            #
#                                   #
#####################################

class PurificationCostTableRow:
    def __init__(self, round, fidelity, improvement):
        self.round = round
        self.fidelity = fidelity
        self.improvement = improvement

class PurificationCostTable:
    def __init__(self):
        self.edges = {}

class QPath:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        assert isinstance(env.get(), EntanglementEnv)
        self._n_agents = env.get_num_agents()
        self._model = model
        self._action_space = action_space
        self._args = args
        self.static_paths = False
        self.network = None
        self.requests_without_path = {}
        self.current_paths = {}
        self.coordinator_node = None

        self.target_fidelity = 0.7

        self._env.network.auto_distillation_threshold = 0

    @staticmethod
    def verify_resource_availability(network, path_information):
        total_resource_consumption = {}
        for single_path_information in path_information:
            path, distillations = single_path_information
            if path is not None:
                for i in range(len(path) - 1):
                    edge_tuple = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    if edge_tuple not in total_resource_consumption:
                        total_resource_consumption[edge_tuple] = 0
                    total_resource_consumption[edge_tuple] += distillations[i] + 1
        links = nx.get_edge_attributes(network.coordinator_G, "fidelity")
        sufficient_resources = True
        for edge_tuple in total_resource_consumption:
            if len(links[edge_tuple]) < total_resource_consumption[edge_tuple]:
                sufficient_resources = False
        return sufficient_resources

    @staticmethod
    def calculate_channel_capacity(network, path):
        links = nx.get_edge_attributes(network.coordinator_G, "fidelity")
        capacity = 0

        for i in range(len(path) - 1):
            edge_tuple = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            capacity += len(links[edge_tuple])

        return capacity


    @staticmethod
    def calculate_path_utility(network, path, distillations):
        degree_of_freedom = QPath.calculate_path_degree_of_freedom(network, path)
        resource_consumption = QPath.calculate_path_resource_consumption(distillations)
        channel_capacity = QPath.calculate_channel_capacity(network, path)

        return 0.5 / (2 * len(network.edges)) * degree_of_freedom + 0.5 / (len(network.edges) * channel_capacity) * resource_consumption

    @staticmethod
    def calculate_path_degree_of_freedom(network, path):
        degree_of_freedom = 0
        for i in range(1, len(path) - 1):
            node = network.nodes[path[i]]
            degree_of_freedom += len(node.neighbors)
        return degree_of_freedom

    @staticmethod
    def calculate_path_resource_consumption(distillations):
        resources = 0
        for distillation in distillations:
            resources += distillation + 1
        return resources

    @staticmethod
    def get_remaining_links(edge, links, reservations):
        reserved = 0
        if edge in reservations:
            reserved = reservations[edge]
        return links[edge][: len(links[edge]) - reserved]

    @staticmethod
    def calculate_purification_cost_table(network, reservations):
        coordinator_G = network.coordinator_G
        links = nx.get_edge_attributes(coordinator_G, "fidelity")
        purification_cost_table = {}
        for edge in coordinator_G.edges():
            edge_links = QPath.get_remaining_links(edge, links, reservations)

            if not edge_links or len(edge_links) == 0:
                # Handle the case where there are no links
                purification_cost_table_rows = []
            else:
                purification_cost_table_rows = []
                distilled_fidelity = -1
                for i in range(len(edge_links)):
                    edge_link = edge_links[-i - 1]
                    if i == 0:
                        distilled_fidelity = edge_link
                    else:
                        distilled_fidelity = QuantumLink.get_distillation_fidelity(distilled_fidelity, edge_link)
                    purification_cost_table_rows.append(PurificationCostTableRow(i, distilled_fidelity, distilled_fidelity - purification_cost_table_rows[i - 1].fidelity if i > 0 else 0))

            purification_cost_table[min(edge[0], edge[1]), max(edge[0], edge[1])] = purification_cost_table_rows
        return purification_cost_table

    def reset_episode(self):
        self.network = None
        self.requests_without_path = {}

        self.coordinator_node = None

    def __call__(self, obs, adj):
        # print("Calling QPath...")
        act = np.zeros(self._env.n_requests, dtype=np.int32)

        if self.static_paths:
            if self.network is None:
                self.network = copy.deepcopy(self._env.get().network)
            network = self.network
        else:
            network = self._env.get().network

        if self.coordinator_node is None:
            if self._env.get().network.coordination_node is None:
                centrality = nx.closeness_centrality(self._env.network.G)
                max_node = max(centrality, key=centrality.get)
                self.coordinator_node = max_node
                self._env.get().network.set_coordination_node(max_node)
            else:
                self.coordinator_node = self._env.get().network.coordination_node

        path_calculation_necessary = False
        for request in self._env.get().requests:
            if request.now == request.start:
                path_calculation_necessary = True

        if path_calculation_necessary:
            last_failed_requests = -1
            all_paths = [(None, None)] * self._env.get().n_requests
            all_path_utility = np.array([1000000.] * self._env.get().n_requests)
            fixed_paths = [True if request.start != request.now else False for request in self._env.get().requests]

            while sum(fixed_paths) < len(all_paths):
                reservations = {}
                for i, fixed_path in enumerate(fixed_paths):
                    if fixed_path and all_paths[i][0] is not None:
                        current_path = all_paths[i][0]
                        current_distillations = all_paths[i][1]
                        for j in range(len(current_path) - 1):
                            edge_tuple = (min(current_path[j], current_path[j + 1]), max(current_path[j], current_path[j + 1]))
                            if edge_tuple not in reservations:
                                reservations[edge_tuple] = 0
                            reservations[edge_tuple] += current_distillations[j]+ 1

                for i in range(self._env.get().n_requests):
                    request = self._env.get().requests[i]

                    if (last_failed_requests == -1 and not fixed_paths[i]) or (last_failed_requests == i):
                        if self._env.network.eval:
                            start_time = time.time()

                        path, distillations, purification_cost_table = self.q_path_algorithm(network, request, reservations)

                        if self._env.network.eval:
                            end_time = time.time()
                            self._env.done_metrics.rl_duration += end_time - start_time

                        all_paths[i] = (path, distillations)

                        if path is not None:
                            all_path_utility[i] = QPath.calculate_path_utility(network, path, distillations)
                        else:
                            all_path_utility[i] = 1000000
                            fixed_paths[i] = True
                    if fixed_paths[i]:
                        all_path_utility[i] = 1000000

                while sum(fixed_paths) < len(all_paths):
                    path_information = []
                    for i, fixed_path in enumerate(fixed_paths):
                        if fixed_path and all_paths[i][0] is not None:
                            path_information.append(all_paths[i])

                    next_path = np.argmin(all_path_utility)
                    path_information.append(all_paths[next_path])
                    if not QPath.verify_resource_availability(self._env.get().network, path_information):
                        if last_failed_requests == next_path:
                            all_paths[next_path] = (None, None)
                            fixed_paths[next_path] = True
                            all_path_utility[next_path] = 1000000
                        else:
                            last_failed_requests = next_path
                            break
                    else:
                        fixed_paths[next_path] = True
                        all_path_utility[next_path] = 1000000



            for i in range(self._env.get().n_requests):
                request = self._env.get().requests[i]

                if request.start == request.now:
                    path, distillations = all_paths[i]

                    if path is not None:
                        request.distillations = distillations
                        self.current_paths[request.id] = path
                    else:
                        if request.id in self.current_paths:
                            del self.current_paths[request.id]

        for i in range(self._env.get().n_requests):
            request = self._env.get().requests[i]
            current_node = request.now
            target_node = request.target

            if request.id not in self.current_paths:
                act[i] = 0  # or some other default action indicating failure

                if not hasattr(network, "refresh_rate") or network.refresh_rate == 0 and request.now == request.start:
                    self.requests_without_path[request.start] = []
                    if request.target not in self.requests_without_path[request.start]:
                        self.requests_without_path[request.start].append(request.target)

            else:
                found = False
                if request.now in self.current_paths[request.id]:
                    current_index = self.current_paths[request.id].index(request.now)
                    if len(self.current_paths[request.id]) > current_index + 1:
                        next_node = self.current_paths[request.id][current_index + 1]
                    else:
                        next_node = request.now

                    for index, edge_id in enumerate(network.nodes[current_node].edges):
                        if network.edges[edge_id].get_other_node(current_node) == next_node:
                            act[i] = index + 1
                            found = True
                            break

                if not found:
                    del self.current_paths[request.id]
                    act[i] = -1
        return act

    def q_path_algorithm(self, network, request, reservations):
        s_i, d_i = request.start, request.target

        purification_cost_table = QPath.calculate_purification_cost_table(self._env.network, reservations)
        Ga, _, _ = self.construct_auxiliary_graph(purification_cost_table, reservations)

        number_of_paths = 0
        min_path = None
        min_length = float("inf")
        min_path_distillations = None

        shortest_paths = self.find_multiple_shortest_paths(Ga, s_i, d_i)

        costs = []
        distillations = []
        for i in range(len(shortest_paths)):
            costs.append(len(shortest_paths[i]) - 1)
            distillations.append([0] * (len(shortest_paths[i]) - 1))
        number_of_paths += len(shortest_paths)
        to_be_optimized = set(range(len(shortest_paths)))

        while len(to_be_optimized) > 0:
            for i in range(len(shortest_paths)):
                if i not in to_be_optimized:
                    continue
                path = shortest_paths[i]
                current_fidelity = 1
                max_improvement = 0
                max_improvement_index = -1
                for j in range(len(path) - 1):
                    row = purification_cost_table[min(path[j], path[j + 1]), max(path[j], path[j + 1])]
                    entry = row[distillations[i][j]]
                    current_fidelity *= entry.fidelity
                    if len(row) > distillations[i][j] + 1:
                        next_entry = row[distillations[i][j] + 1]
                        if next_entry.improvement > max_improvement:
                            max_improvement = next_entry.improvement
                            max_improvement_index = j

                if current_fidelity >= self.target_fidelity or max_improvement_index < 0 or len(path) - 1 + sum(distillations[i]) >= min_length:
                    to_be_optimized.remove(i)

                    if current_fidelity >= self.target_fidelity:
                        total_costs = len(path) - 1 + sum(distillations[i])
                        if len(path) - 1 + sum(distillations[i]) < min_length:
                            min_path = path
                            min_length = total_costs
                            min_path_distillations = distillations[i]
                else:
                    distillations[i][max_improvement_index] += 1

        return min_path, min_path_distillations, purification_cost_table

    def construct_auxiliary_graph(self, purification_cost_table, reservations):
        links = nx.get_edge_attributes(self._env.network.coordinator_G, "fidelity")

        # print("Constructing auxiliary graph...")
        Ga = nx.empty_graph(self._env.network.coordinator_G)
        for edge in self._env.network.coordinator_G.edges:
            edge_links = QPath.get_remaining_links(edge, links, reservations)

            if len(edge_links) >= 1 and purification_cost_table[edge][-1].fidelity >= self.target_fidelity:
                Ga.add_edge(edge[0], edge[1])
        return Ga, Ga.edges, nx.get_edge_attributes(Ga, "cost")

    def find_multiple_shortest_paths(self, Ga, s_i, d_i):
        paths = []
        max_number_of_paths = 100
        try:
            generator = nx.shortest_simple_paths(Ga, source=s_i, target=d_i)
            for path in generator:
                if len(path) > self._env.get().network.ttl + 1:
                    break
                paths.append(path)
                if len(paths) >= max_number_of_paths:
                    break
        except nx.NetworkXNoPath:
            pass
        return paths

class QLeap:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        assert isinstance(env.get(), EntanglementEnv)
        self._n_agents = env.get_num_agents()
        self._model = model
        self._action_space = action_space
        self._args = args
        self.static_paths = False
        self.network = None
        self.requests_without_path = {}
        self.current_paths = {}

        self.target_fidelity = 0.7

        self.last_G = None
        self.last_paths = None
        self.coordinator_node = None

        self.max_distillation = 5

        self._env.network.auto_distillation_threshold = 0

    def reset_episode(self):
        self.network = None
        self.requests_without_path = {}

        self.coordinator_node = None

    def __call__(self, obs, adj):
        # print("Calling QLeap...")
        act = np.zeros(self._env.n_requests, dtype=np.int32)

        if self.coordinator_node is None:
            if self._env.get().network.coordination_node is None:
                centrality = nx.closeness_centrality(self._env.network.G)
                max_node = max(centrality, key=centrality.get)
                self.coordinator_node = max_node
                self._env.get().network.set_coordination_node(max_node)
            else:
                self.coordinator_node = self._env.get().network.coordination_node

        path_calculation_necessary = False
        for request in self._env.get().requests:
            if request.now == request.start:
                path_calculation_necessary = True

        if path_calculation_necessary:
            last_failed_requests = -1
            all_paths = [(None, None)] * self._env.get().n_requests
            all_path_utility = np.array([1000000.] * self._env.get().n_requests)
            fixed_paths = [True if request.start != request.now else False for request in self._env.get().requests]

            coordinator_G = self._env.get().network.coordinator_G
            links = nx.get_edge_attributes(coordinator_G, 'fidelity')

            while sum(fixed_paths) < len(all_paths):
                reservations = {}
                for i, fixed_path in enumerate(fixed_paths):
                    if fixed_path and all_paths[i][0] is not None:
                        current_path = all_paths[i][0]
                        current_distillations = all_paths[i][1]
                        for j in range(len(current_path) - 1):
                            edge_tuple = (min(current_path[j], current_path[j + 1]), max(current_path[j], current_path[j + 1]))
                            if edge_tuple not in reservations:
                                reservations[edge_tuple] = 0
                            reservations[edge_tuple] += current_distillations[j]+ 1

                for i in range(self._env.n_requests):
                    request = self._env.requests[i]

                    if (last_failed_requests == -1 and not fixed_paths[i]) or (last_failed_requests == i):
                        if self._env.network.eval:
                            start_time = time.time()

                        path, fidelity, Ga = self.q_leap_algorithm(request, reservations)

                        distillations = None
                        if path is not None:
                            distillations = []
                            success = True

                            for j in range(len(path) - 1):
                                start = min(path[j], path[j + 1])
                                end = max(path[j], path[j + 1])
                                edge_tuple = (start, end)

                                if len(links[edge_tuple]) == 0:
                                    success = False
                                    break
                                if len(links[edge_tuple]) > 1 and links[edge_tuple][-1] < math.pow(self.target_fidelity, 1 / (len(path) - 1)):
                                    distillation_fidelity = None
                                    k = 0
                                    for k in range(len(links[edge_tuple])):
                                        link_for_distillation = links[edge_tuple][-k - 1]
                                        if distillation_fidelity is None:
                                            distillation_fidelity = link_for_distillation
                                        else:
                                            distillation_fidelity = QuantumLink.get_distillation_fidelity(distillation_fidelity, link_for_distillation)

                                        if distillation_fidelity >= math.pow(self.target_fidelity, 1 / (len(path) - 1)):
                                            break
                                    distillations.append(k)

                                    if distillation_fidelity < math.pow(self.target_fidelity, 1 / (len(path) - 1)):
                                        success = False
                                else:
                                    distillations.append(0)
                                    if links[edge_tuple][-1] < math.pow(self.target_fidelity, 1 / (len(path) - 1)):
                                        success = False
                            if not success:
                                path = None
                                distillations = None
                            all_paths[i] = (path, distillations)

                        if self._env.network.eval:
                            end_time = time.time()
                            self._env.done_metrics.rl_duration += end_time - start_time

                        all_paths[i] = (path, distillations)

                        path, distillations = all_paths[i]
                        if path is not None:
                            all_path_utility[i] = QPath.calculate_path_utility(self._env.get().network, path, distillations)
                        else:
                            all_path_utility[i] = 1000000
                            fixed_paths[i] = True
                    if fixed_paths[i]:
                        all_path_utility[i] = 1000000

                while sum(fixed_paths) < len(all_paths):
                    path_information = []
                    for i, fixed_path in enumerate(fixed_paths):
                        if fixed_path and all_paths[i][0] is not None:
                            path_information.append(all_paths[i])

                    next_path = np.argmin(all_path_utility)
                    path_information.append(all_paths[next_path])
                    if not QPath.verify_resource_availability(self._env.get().network, path_information):
                        if last_failed_requests == next_path:
                            all_paths[next_path] = (None, None)
                            fixed_paths[next_path] = True
                            all_path_utility[next_path] = 1000000
                        else:
                            last_failed_requests = next_path
                            break
                    else:
                        fixed_paths[next_path] = True
                        all_path_utility[next_path] = 1000000

        for i in range(self._env.get().n_requests):
            request = self._env.get().requests[i]

            if request.start == request.now:
                path, distillations = all_paths[i]
                if path is not None:
                    request.distillations = distillations
                    self.current_paths[request.id] = path
                else:
                    if request.id in self.current_paths:
                        del self.current_paths[request.id]

        for i, request in enumerate(self._env.get().requests):
            current_node = request.now
            target_node = request.target

            if request.id not in self.current_paths:
                act[i] = 0

                if not hasattr(self._env.network, "refresh_rate") or self._env.network.refresh_rate == 0 and request.now == request.start:
                    if request.start not in self.requests_without_path:
                        self.requests_without_path[request.start] = []
                    if request.target not in self.requests_without_path[request.start]:
                        self.requests_without_path[request.start].append(request.target)

            else:
                found = False
                if request.now in self.current_paths[request.id]:
                    current_index = self.current_paths[request.id].index(request.now)
                    if len(self.current_paths[request.id]) > current_index + 1:
                        next_node = self.current_paths[request.id][current_index + 1]
                    else:
                        next_node = request.now

                    for index, edge_id in enumerate(self._env.network.nodes[current_node].edges):
                        if self._env.network.edges[edge_id].get_other_node(current_node) == next_node:
                            act[i] = index + 1
                            found = True
                            break

                if not found:
                    del self.current_paths[request.id]
                    act[i] = -1

        return act

    def q_leap_algorithm(self, request, reservations):
        s_i, d_i = request.start, request.target

        purification_cost_table = QPath.calculate_purification_cost_table(self._env.network, reservations)
        Ga = self.construct_auxiliary_graph(purification_cost_table, reservations)

        max_path = None
        fidelity = -1

        try:
            max_path = nx.shortest_path(Ga, s_i, d_i, weight='fidelity_log')
            fidelity = math.pow(10, -nx.shortest_path_length(Ga, s_i, d_i, weight='fidelity_log'))

        except nx.NetworkXNoPath:
            pass

        return max_path, fidelity, Ga

    def construct_auxiliary_graph(self, purification_cost_table, reservations):
        links = nx.get_edge_attributes(self._env.network.coordinator_G, "fidelity")

        Ga = nx.empty_graph(self._env.network.coordinator_G)
        for edge in self._env.network.coordinator_G.edges:
            edge_links = QPath.get_remaining_links(edge, links, reservations)

            if len(purification_cost_table[edge]) > 0 and purification_cost_table[edge][-1].fidelity > self.target_fidelity:
                Ga.add_edge(edge[0], edge[1])
                fidelity_log = 1000
                if len(edge_links) >= 1:
                    fidelity = edge_links[-1]

                    fidelity_log = -math.log10(fidelity)
                nx.set_edge_attributes(Ga, {edge: fidelity_log}, "fidelity_log")
        return Ga

class GreedyEntanglementRouting:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self.requests_without_path = {}

    def reset_episode(self):
        pass

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_requests, dtype=np.int32)
        network = self._env.network

        for i in range(self._env.n_requests):
            request = self._env.requests[i]
            source = request.now
            target = request.target

            if source == target:
                act[i] = 0
                continue

            if request.start in self.requests_without_path and request.target in self.requests_without_path[request.start]:
                continue


            if self._env.network.eval:
                start_time = time.time()
            path = self.path_discovery(source, target, network)
            if self._env.network.eval:
                end_time = time.time()
                self._env.done_metrics.rl_duration += end_time - start_time
            if path is None or len(path) < 2:
                act[i] = 0

                if not hasattr(network, "refresh_rate") or network.refresh_rate == 0 and request.now == request.start:
                    if request.start not in self.requests_without_path:
                        self.requests_without_path[request.start] = []
                    if request.target not in self.requests_without_path[request.start]:
                        self.requests_without_path[request.start].append(request.target)
                continue

            found = False
            for j, edge_id in enumerate(network.nodes[source].edges):
                edge = network.edges[edge_id]
                if edge.get_other_node(source) == path[1] and len(edge.links) > 0:
                    act[i] = j + 1
                    found = True
            if not found:
                act[i] = 0

        return act

    def path_discovery(self, source, target, network):
        try:
            return nx.shortest_path(network.G, source, target, weight=network.get_path_length)
        except nx.NetworkXNoPath:
            return None


class ModifiedGreedyEntanglementRouting:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self.network = None
        self.requests_without_path = {}

    def reset_episode(self):
        self.network = None

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_requests, dtype=np.int32)
        network = self._env.network

        for i in range(self._env.n_requests):
            request = self._env.requests[i]
            source = request.now
            target = request.target

            if source == target:
                act[i] = 0
                continue

            if request.start in self.requests_without_path and request.target in self.requests_without_path[request.start]:
                continue

            if self._env.network.eval:
                start_time = time.time()
            path = self.modified_path_discovery(source, target, network, request.ttl)
            if self._env.network.eval:
                end_time = time.time()
                self._env.done_metrics.rl_duration += end_time - start_time
            if path is None or len(path) < 2:
                act[i] = 0
                continue

            if not hasattr(network, "refresh_rate") or network.refresh_rate == 0 and request.now == request.start:
                if request.start not in self.requests_without_path:
                    self.requests_without_path[request.start] = []
                if request.target not in self.requests_without_path[request.start]:
                    self.requests_without_path[request.start].append(request.target)

            found = False
            for j, edge_id in enumerate(network.nodes[source].edges):
                edge = network.edges[edge_id]
                if edge.get_other_node(source) == path[1]:
                    act[i] = j + 1
                    found = True
            if not found:
                act[i] = 0

        return act

    def modified_path_discovery(self, source, target, network, ttl):
        available_virtual_paths = []
        for edge_id in network.nodes[source].edges:
            edge = network.edges[edge_id]

            if len(edge.links) == 0:
                continue

            neigh = edge.get_other_node(source)
            try:
                path = nx.shortest_path(network.G, neigh, target, weight=network.get_path_length)
                path_length_neigh_target = nx.shortest_path_length(network.G, neigh, target, weight=network.get_path_length)
                path_length_start_neigh = nx.shortest_path_length(network.G, source, neigh, weight=network.get_path_length)
                path_length_start_target = nx.shortest_path_length(network.G, source, target, weight=network.get_path_length)

                if round(path_length_start_target, 3) == round(path_length_start_neigh + path_length_neigh_target, 3):
                    return [source, *path]
                else:
                    available_virtual_paths.append([source, *path])
            except nx.NetworkXNoPath:
                continue

        if len(available_virtual_paths) > 0:
            min_available_path = None
            for available_virtual_path in available_virtual_paths:
                if min_available_path is None or len(min_available_path) < len(available_virtual_path):
                    min_available_path = available_virtual_path
            return min_available_path
        else:
            try:
                return nx.shortest_path(network.G, source, target, weight=network.get_path_length)
            except nx.NetworkXNoPath:
                return None

class LocalBestEffortRouting:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self.network = None

    def reset_episode(self):
        self.network = None

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_requests, dtype=np.int32)
        network = self._env.network

        for i in range(self._env.n_requests):
            request = self._env.requests[i]
            source = request.now
            target = request.target

            if source == target:
                act[i] = 0
                continue

            if self._env.network.eval:
                start_time = time.time()
            path = self.local_best_effort_path_discovery(source, target, network)
            if self._env.network.eval:
                end_time = time.time()
                self._env.done_metrics.rl_duration += end_time - start_time
            if path is None or len(path) < 2:
                act[i] = 0
                continue

            found = False
            for j, edge_id in enumerate(network.nodes[source].edges):
                edge = network.edges[edge_id]
                if edge.get_other_node(source) == path[1]:
                    act[i] = j + 1
                    found = True
            if not found:
                act[i] = 0

        return act

    def local_best_effort_path_discovery(self, source, target, network):
        min_available_path = None
        min_available_path_length = float("inf")
        for edge_id in network.nodes[source].edges:
            edge = network.edges[edge_id]

            if len(edge.links) == 0:
                continue

            neigh = edge.get_other_node(source)
            try:
                path = nx.shortest_path(network.G, neigh, target, weight=network.get_path_length)
                path_length = nx.shortest_path_length(network.G, neigh, target, weight=network.get_path_length)

                if path_length < min_available_path_length:
                    min_available_path = [source, *path]
                    min_available_path_length = path_length
            except nx.NetworkXNoPath:
                continue

        if min_available_path is not None:
            return min_available_path
        else:
            try:
                return nx.shortest_path(network.G, source, target, weight=network.get_path_length)
            except nx.NetworkXNoPath:
                return None

class NoNLocalBestEffortRouting:
    def __init__(self, env, model, action_space, args) -> None:
        self._env = env
        self._model = model
        self._action_space = action_space
        self._args = args
        self.network = None

        self.shortest_paths_topology = None

    def reset_episode(self):
        self.network = None
        self.shortest_paths_topology = None

    def __call__(self, obs, adj):
        act = np.zeros(self._env.n_requests, dtype=np.int32)
        network = self._env.network

        if network.coordination_node is None:
            network.set_coordination_node(-1)

        if self.shortest_paths_topology is None:
            if self._env.network.eval:
                start_time = time.time()
            self.shortest_paths_topology = dict(nx.all_pairs_dijkstra(network.G, weight=network.get_path_length))
            if self._env.network.eval:
                end_time = time.time()
                self._env.done_metrics.rl_duration += end_time - start_time

        for i in range(self._env.n_requests):
            request = self._env.requests[i]

            source = request.now
            target = request.target

            if source == target:
                act[i] = 0
                continue

            if self._env.network.eval:
                start_time = time.time()

            path = self.non_local_best_effort_path_discovery(source, target, network)

            if self._env.network.eval:
                end_time = time.time()
                self._env.done_metrics.rl_duration += end_time - start_time

            if path is None or len(path) < 2:
                act[i] = 0
                continue

            found = False
            for j, edge_id in enumerate(network.nodes[source].edges):
                edge = network.edges[edge_id]
                if edge.get_other_node(source) == path[1]:
                    act[i] = j + 1
                    found = True
            if not found:
                act[i] = 0

        return act

    def non_local_best_effort_path_discovery(self, source, target, network):
        min_available_path = None
        min_available_path_length = float("inf")
        min_path = None
        min_path_length = float("inf")

        links = nx.get_edge_attributes(network.node_Gs[source], "fidelity")
        for edge in network.node_Gs[source].edges(source):
            neigh = edge[1]
            if edge not in links:
                edge = (edge[1], edge[0])

            if neigh == target:
                if len(links[edge]) > 0:
                    min_available_path = [source, neigh]
                    min_available_path_length = 0
                else:
                    min_path = [source, neigh]
                    min_path_length = 0
            else:
                for neigh_edge in network.node_Gs[source].edges(neigh):
                    neigh_of_neigh = neigh_edge[1]
                    if neigh_edge not in links:
                        neigh_edge = (neigh_edge[1], neigh_edge[0])

                    if neigh_of_neigh == source:
                        continue

                    if neigh_of_neigh in self.shortest_paths_topology and target in self.shortest_paths_topology[neigh_of_neigh][1]:
                        path = self.shortest_paths_topology[neigh_of_neigh][1][target]
                        path_length = self.shortest_paths_topology[neigh_of_neigh][0][target]

                        if len(links[edge]) > 0 and len(links[neigh_edge]) > 0:
                            if path_length < min_available_path_length:
                                min_available_path = [source, neigh, *path]
                                min_available_path_length = path_length
                        if path_length < min_path_length:
                            min_path = [source, neigh, *path]
                            min_path_length = path_length
                    else:
                        continue

        if min_available_path is not None:
            return min_available_path
        elif min_path is not None:
            return min_path
        else:
            try:
                return nx.shortest_path(network.G, source, target, weight=network.get_path_length)
            except nx.NetworkXNoPath:
                return None