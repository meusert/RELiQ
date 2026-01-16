import copy
import math
import time
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import topohub

from matplotlib import colormaps
import geopy.distance
from enum import Enum
import scipy
import util

MAX_LINK_LENGTH = 250

MAX_LINK_LENGTH_REAL_TOPOLOGY = 200
MAX_LINK_LENGTH_DIVISOR_REAL_TOPOLOGY = 150

MAX_QUBITS_GENERATION_RATE = 1000000

class Entanglement:

    def __init__(self, source, destination, fidelity):
        self.source = source
        self.destination = destination
        self.fidelity = fidelity

class QuantumRepeater:
    """
    A node in a network.
    """

    NUMBER_OF_SPECIAL_NODE_TYPES = 3
    class Type(Enum):
        NORMAL = 0

    def __init__(self, id, x, y, z=0, movable=False, movement_vector=(0, 0, 0), swap_prob=1, decay=0., virtual=False, type=Type.NORMAL, n_decoupling_pulses=1024, source_destination_valid: bool = True):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.type = type
        self.movable = movable
        self.movement_vector = movement_vector
        self.neighbors = []
        self.edges = []
        self.edge_node_association = {}
        self.swap_prob = swap_prob
        self.decay = decay
        self.virtual = virtual

        self.failure_step = -1
        self.entanglements = []
        self.n_decoupling_pulses = n_decoupling_pulses

        self.source_destination_valid = source_destination_valid

    def get_quantum_storage_utilization(self, all_edges):
        sum = 0
        for edge_id in self.edges:
            edge = all_edges[edge_id]
            sum += len(edge.links)
        return sum

class LinkReservation:
    link_reservation_id = 0

    def __init__(self, number):
        self.number = number
        self.id = LinkReservation.link_reservation_id
        LinkReservation.link_reservation_id += 1

class QuantumLink:
    """
    A quantum link in a network.
    """

    FIDELITY_THRESHOLD = 0.5
    USE_QISKIT = True
    FIDELITIES_QISKIT = np.load("data/all_fidelities.npy")

    swap_random_state = None

    def __init__(self, start, end, fidelity, cost=0, creation=None):
        self.start = start
        self.end = end
        self.initial_fidelity = fidelity
        self.fidelity = fidelity
        self.cost = cost
        self.creation = creation

    def decay(self, decay_factor):
        self.fidelity *= decay_factor
        if self.fidelity < QuantumLink.FIDELITY_THRESHOLD:
            self.fidelity = 0

    def decay_realistic(self, time, n_decoupling_pulses):
        k = 2.2
        alpha = 0.53  # Power-law exponent from the paper
        C = 42.0  # Increased proportionality constant for a slower, more accurate decay
        B = 0.5  # Baseline fidelity

        def get_t2_n(n, C, alpha):
            """Calculates the coherence time for a given n."""
            return C * (n ** alpha)

        def fidelity_decay(t, t2, A, B, k):
            """Calculates the state fidelity at time t."""
            return A * np.exp(-((t / t2) ** k)) + B

        A_n = self.initial_fidelity - B
        t2_n = get_t2_n(n_decoupling_pulses, C, alpha)
        link_age = (time - self.creation) * 1000

        self.fidelity = fidelity_decay(link_age, t2_n, A_n, B, k)

        '''
        if self.fidelity < QuantumLink.FIDELITY_THRESHOLD * 1.05:
            self.fidelity = 0
        '''

    @staticmethod
    def get_distillation_fidelity(fidelity1, fidelity2):
        return fidelity1 * fidelity2 / (fidelity1 * fidelity2 + (1 - fidelity1) * (1 - fidelity2))

    @staticmethod
    def create_distillation(link1, link2, creation_time):
        assert link1.start == link2.start and link1.end == link2.end

        return QuantumLink(link1.start, link1.end, QuantumLink.get_distillation_fidelity(link1.fidelity, link2.fidelity), creation=creation_time)

    @staticmethod
    def get_swap_fidelity(fidelity1, fidelity2, gate_error=0, random_index=None):
        if QuantumLink.USE_QISKIT:
            min_fidelity = .26
            f1_index = math.floor((fidelity1 - min_fidelity) * 100)
            f2_index = math.floor((fidelity2 - min_fidelity) * 100)
            error_index = round(gate_error * 100)
            if f1_index < 0 or f2_index < 0:
                fidelity = 0
            else:
                if random_index is not None:
                    fidelity = QuantumLink.FIDELITIES_QISKIT[f1_index][f2_index][error_index][random_index]
                else:
                    fidelity = np.mean(QuantumLink.FIDELITIES_QISKIT[f1_index][f2_index][error_index])
        else:
            fidelity = max(
                1
                / 4
                * (1 + 1 / 3 * (4 * fidelity1 - 1) * (4 * fidelity2 - 1)),
                0,
            )
        if fidelity < QuantumLink.FIDELITY_THRESHOLD:
            fidelity = 0
        return fidelity

    @staticmethod
    def swap(link1, link2, swap_probability, source, rng_generator, ignore_drop = False):
        if link1.start == link2.start or link1.start == link2.end \
                or link1.end == link2.start or link1.end == link2.end:
            # we have a valid pair of quantum links

            if link1.start == link2.start and link1.end == link2.end:
                # Quantum Loop
                if link1.start != source:
                    src_node = link1.start
                else:
                    src_node = link1.end
                dst_node = src_node
            else:
                # Normal case
                if link1.start != link2.start and link1.start != link2.end:
                    entangled_node1 = link1.start
                else:
                    entangled_node1 = link1.end

                if link2.start != link1.start and link2.start != link1.end:
                    entangled_node2 = link2.start
                else:
                    entangled_node2 = link2.end

                src_node = min(entangled_node1, entangled_node2)
                dst_node = max(entangled_node1, entangled_node2)

            if QuantumLink.USE_QISKIT:
                index = rng_generator.integers(QuantumLink.FIDELITIES_QISKIT.shape[3])
                fidelity = QuantumLink.get_swap_fidelity(link1.fidelity, link2.fidelity, 1 - swap_probability, index)
                return QuantumLink(src_node, dst_node, fidelity)
            else:
                random_value = rng_generator.random()

                if random_value < swap_probability or (
                    ignore_drop and swap_probability > 0
                ):
                    fidelity = QuantumLink.get_swap_fidelity(link1.fidelity, link2.fidelity, 0)
                    return QuantumLink(src_node, dst_node, fidelity)
                else:
                    return QuantumLink(src_node, dst_node, 0)

    def get_other_node(self, node):
        if self.start == node:
            return self.end
        elif self.end == node:
            return self.start
        else:
            raise ValueError(
                f"Is neither start nor end of edge {self.start}-{self.end}: {node}"
            )


class Edge:
    """
    An edge in a network.
    """

    def __init__(self, start, end, links, dead=False):
        self.start = start
        self.end = end
        self.links = links
        self.dead = dead
        self.dead_step = -1
        self.intermediate_routers = []
        self.reserved_links = {}

    def get_total_reservations(self):
        sum = 0
        for reserved in self.reserved_links.values():
            sum += reserved.number
        return sum

    def get_other_node(self, node):
        if self.start == node:
            return self.end
        elif self.end == node:
            return self.start
        else:
            raise ValueError(
                f"Is neither start nor end of edge {self.start}-{self.end}: {node}"
            )


def _is_ground_connection(node1, node2):
    return node1.z == 0 and node2.z == 0


class CoordinationMessage:
    def __init__(self, current_node_index, path, state, time):
        self.current_node_index = current_node_index
        self.path = path
        self.state = state
        self.time = time

    def get_next_edge(self):
        if self.current_node_index < len(self.path) - 1:
            return (min(self.path[self.current_node_index], self.path[self.current_node_index + 1]), max(self.path[self.current_node_index], self.path[self.current_node_index + 1]))
        return None

    def get_current_node(self):
        return self.path[self.current_node_index]

    def move(self, steps = 1):
        if self.current_node_index < len(self.path) - steps:
            self.current_node_index += steps
            return True
        self.current_node_index = len(self.path) - 1
        return False

    def has_arrived(self):
        return self.current_node_index >= len(self.path) - 1

class QuantumNetwork:
    """
    Network class that manages the creation of graphs.
    """

    def __init__(
        self,
        n_nodes=20,
        n_nodes_connected=None,
        random_topology=False,
        n_random_seeds=None,
        sequential_topology_seeds=False,
        topology_init_seed=476,
        excluded_seeds: Optional[List[int]] = None,
        provided_seeds: Optional[List[int]] = None,
        n_quantum_links: int = 20,
        timestep_decay: float = 0.995,
        max_link_age: float = 0,
        entanglement_probability: float = 0.9942404238,
        world_width: int = 1500,
        world_height: int = 1500,
        episode_steps: int = 30,
        eval_episode_steps: int = 30,
        adaptive_episode_length: bool = False,
        swap_probability: float = 1,
        swap_probability_std: float = 0,
        neighbor_count: int = 3,
        node_degree: int = 3,
        ttl: int = 0,
        topohub_topology: str = None,
        refresh_rate: float = 0,
        fixed_path_length: int = -1,
        attenuation_coefficient: float = 2,
        use_realistic_decay: bool = False,
        n_decoupling_pulses_avg: int = 1024,
        n_decoupling_pulses_std: int = 0,
        initial_fidelity: float = 1,
        auto_distillation_threshold: float = .95,
        message_speed: int = 1,
    ):
        """
        Initializes the network and optionally creates list of valid random topology seeds.

        :param n_nodes: Number of nodes in the network, defaults to 20
        :param random_topology: Create a random topology on each reset, defaults to False
        :param n_random_seeds: Number of random topologies, defaults to None
        :param sequential_topology_seeds: Sample topologies sequentially, defaults to False
        :param topology_init_seed: Seed for topology creation, defaults to 476
        :param excluded_seeds: Seeds that must not be used for topology creation, defaults to None
        :param provided_seeds: Use provided seeds and generate no new topologies, defaults to None
        """
        self.antenna_directivity = pow(10, 8.96/10)
        self.speed_of_light_air = 299792458 / 1000.
        self.speed_of_light_glass = self.speed_of_light_air / 1.5
        self.beam_wave_length = self.speed_of_light_air / (1.58 * pow(10, 12))
        self.attenuation_coefficient = attenuation_coefficient

        self.message_speed = message_speed

        self.node_distances = {}

        self.initial_fidelity = initial_fidelity

        self.n_decoupling_pulses_avg = n_decoupling_pulses_avg
        self.n_decoupling_pulses_std = n_decoupling_pulses_std

        self.coordination_node = None
        self.coordinator_distance = None
        self.coordinator_paths = None
        self.coordinator_G = None
        self.node_Gs = None
        self.coordination_messages = []

        self.auto_distillation_threshold = auto_distillation_threshold

        self.transmitted_messages = 0
        self.transmitted_message_load_edges = {}
        self.transmitted_message_load_nodes = {}

        self.use_available_quantum_storage = True

        self.probability_storage = {}

        self.nodes_failed = 0

        self.delta_time = 0.01

        self.n_nodes = n_nodes
        self.n_nodes_connected = n_nodes_connected if n_nodes_connected is not None else n_nodes
        self.refresh_rate = refresh_rate

        self.fixed_path_length = fixed_path_length

        self.eval = False

        self.n_quantum_links = n_quantum_links
        self.max_link_age = max_link_age
        self.timestep_decay = math.pow(timestep_decay, self.delta_time)

        self.min_active_quantum_links = 0
        self.total_active_quantum_links = 0
        self.max_active_quantum_links = 0
        self.average_active_quantum_links = 0

        self.ttl = ttl

        self.done = False

        self.diameter = 1

        self.swap_probability = swap_probability
        self.swap_probability_std = swap_probability_std

        self.env_steps = 0
        self.last_refresh = 0
        self.eval_episode_steps = eval_episode_steps
        self.min_env_steps = episode_steps
        self.current_max_env_steps = self.min_env_steps
        self.next_max_env_steps = self.min_env_steps

        self.neighbor_count = neighbor_count
        self.node_degree = node_degree

        self.topohub_topology = topohub_topology

        self.adaptive_episode_length = adaptive_episode_length

        self.entanglement_probability = entanglement_probability
        self.world_height = world_height
        self.world_width = world_width

        self.use_realistic_decay = use_realistic_decay

        self.nodes = []
        self.edges = []
        self.edge_node_association = {}

        self.G = nx.Graph()
        self.shortest_paths = None
        self.shortest_paths_weights = None

        self.shortest_available_paths = None
        self.shortest_available_paths_weights = None

        self.adj_matrix = None

        self.virtual_node_groups = {}
        self.virtual_subgraphs = {}
        self.virtual_adj_matrix = {}

        # only needed for evaluate the efficiency of creating a correct random topology
        self.repetitions = 0

        self.random_topology = random_topology
        self.current_topology_seed = None
        self.sampled_topology_seeds = []
        self.sequential_topology_seeds = sequential_topology_seeds
        self.sequential_topology_seeds_frozen = False
        self.sequential_topology_index = 0
        self.topology_init_seed = topology_init_seed
        self.exclude_seeds = None if excluded_seeds is None else set(excluded_seeds)
        self.provide_seeds = provided_seeds

        self.quantum_generator = np.random.default_rng(self.topology_init_seed)
        self.topology_generator = np.random.default_rng(self.topology_init_seed)
        self.swap_generator = np.random.default_rng(self.topology_init_seed)
        self.agent_generator = np.random.default_rng(self.topology_init_seed)
        self.node_generator = np.random.default_rng(self.topology_init_seed)

        if provided_seeds is not None and len(provided_seeds) > 0:
            self.seeds = provided_seeds
        else:
            self.seeds = self.build_seed_list(
                random_topology, n_random_seeds, self.exclude_seeds
            )
        if self.exclude_seeds is not None:
            assert all([s not in self.exclude_seeds for s in self.seeds])

        self.reservation_cleanup = {}

    def build_seed_list(self, random_topology, n_random_seeds, exclude_seeds=None):
        if not random_topology:
            return [self.topology_init_seed]

        if n_random_seeds is None or n_random_seeds <= 0:
            return []

        seed_list = []
        # build list of unique num_random_seeds > 0 seeds
        while len(seed_list) < n_random_seeds:
            # build one random network and add topology seed
            new_seed = self._create_valid_network(seeds_exclude=exclude_seeds)
            if new_seed not in seed_list:
                seed_list.append(new_seed)

        return seed_list

    def set_coordination_node(self, coordination_node):
        self.coordination_node = coordination_node
        if coordination_node >= 0:
            self.coordinator_distance = nx.shortest_path_length(self.G, target=self.coordination_node)
            self.coordinator_paths = nx.shortest_path(self.G, target=self.coordination_node)

            coordinator_G = self.G.copy()
            for edge in self.edges:
                fidelity = {(edge.start, edge.end): [link.fidelity for link in edge.links]}
                nx.set_edge_attributes(coordinator_G, fidelity, "fidelity")
            self.coordinator_G = coordinator_G
        else:
            self.node_Gs = []
            node_G = self.G.copy()
            for edge in self.edges:
                fidelity = {(edge.start, edge.end): [link.fidelity for link in edge.links]}
                nx.set_edge_attributes(node_G, fidelity, "fidelity")
            for node in self.nodes:
                if node.virtual:
                    continue
                lengths = nx.shortest_path_length(self.G, source=node.id)
                neighbors = []
                for neighbor in lengths:
                    if lengths[neighbor] <= 2:
                        neighbors.append(neighbor)
                self.node_Gs.append(node_G.subgraph(neighbors))

        if self.coordination_node is not None:
            self.send_update_messages(init=True)

    def send_update_messages(self, init=False):
        for i in range(len(self.coordination_messages)):
            message = self.coordination_messages[i]
            message.move(self.message_speed)

        if self.coordination_node is not None:
            if self.coordination_node >= 0:
                for node in self.nodes:
                    if node.virtual:
                        continue
                    state = {}
                    for edge in node.edges:
                        reservations = self.edges[edge].get_total_reservations()
                        state[(self.edges[edge].start, self.edges[edge].end)] = [link.fidelity for link in self.edges[edge].links[:len(self.edges[edge].links)-reservations]]
                    coordination_message_template = CoordinationMessage(0, self.coordinator_paths[node.id], state, self.env_steps)

                    if init:
                        for i in range(len(coordination_message_template.path)):
                            coordination_message = copy.deepcopy(coordination_message_template)
                            coordination_message.current_node_index = i
                            coordination_message.time = self.env_steps - i
                            self.coordination_messages.append(coordination_message)
                    else:
                        self.coordination_messages.append(coordination_message_template)
            else:
                for node in self.nodes:
                    if node.virtual:
                        continue
                    state = {}
                    for edge in node.edges:
                        state[(self.edges[edge].start, self.edges[edge].end)] = [link.fidelity for link in self.edges[edge].links]
                    for neighbor in node.neighbors:
                        coordination_message_template = CoordinationMessage(0, [node.id, neighbor], state, self.env_steps)
                        if init:
                            for i in range(len(coordination_message_template.path)):
                                coordination_message = copy.deepcopy(coordination_message_template)
                                coordination_message.current_node_index = i
                                coordination_message.time = self.env_steps - i
                                self.coordination_messages.append(coordination_message)
                        else:
                            self.coordination_messages.append(coordination_message_template)

                    self.coordination_messages.append(CoordinationMessage(0, [node.id], state, self.env_steps))

        for i in range(len(self.coordination_messages)):
            message = self.coordination_messages[i]

            if len(message.path) == 1:
                continue

            next_edge = message.get_next_edge()

            if next_edge is not None:
                self.transmitted_messages += 1
                if not next_edge in self.transmitted_message_load_edges:
                    self.transmitted_message_load_edges[next_edge] = 0
                self.transmitted_message_load_edges[next_edge] += 1

            if message.time < self.env_steps:
                node = message.get_current_node()
                if not node in self.transmitted_message_load_nodes:
                    self.transmitted_message_load_nodes[node] = 0
                self.transmitted_message_load_nodes[node] += 1

        original_length = len(self.coordination_messages)
        for i in range(original_length):
            if self.coordination_messages[original_length - i - 1].has_arrived():
                message = self.coordination_messages.pop(original_length - i - 1)

                if self.coordination_node >= 0:
                    nx.set_edge_attributes(self.coordinator_G, message.state, "fidelity")
                else:
                    receiver = message.get_current_node()
                    nx.set_edge_attributes(self.node_Gs[receiver], message.state, "fidelity")

    def fill_missing(self):
        while len(self.nodes) < self.n_nodes:
            self.nodes.append(QuantumRepeater(len(self.nodes), 0, 0, swap_prob=0, virtual=True, n_decoupling_pulses=self.get_n_decoupling_pulses()))

    def get_n_decoupling_pulses(self):
        result = 2048
        while result > 1024:
            result = max(max(round(self.node_generator.normal(self.n_decoupling_pulses_avg, self.n_decoupling_pulses_std) / 8) * 8, 0), 1)
        return result

    def get_link_entanglement_probability(self, start, end, intermediate_routers=[], free_slots = 1, delta_time = None):
        if delta_time is None:
            delta_time = self.delta_time
        if not (start, end, len(intermediate_routers), free_slots, delta_time) in self.probability_storage:
            if self.refresh_rate > 0:
                link_length = self.get_node_distance(self.nodes[start], self.nodes[end]) / (len(intermediate_routers) + 1)
                if _is_ground_connection(self.nodes[start], self.nodes[end]):
                    base_entanglement_probability = self.calculate_base_entanglement_probability_ground(link_length)
                    speed_of_light = self.speed_of_light_glass
                else:
                    base_entanglement_probability = self.calculate_base_entanglement_probability_air(link_length)
                    speed_of_light = self.speed_of_light_air
                link_entanglement_probability = self.calculate_link_entanglement_probability(base_entanglement_probability, speed_of_light, link_length, intermediate_routers, free_slots * (1 if self.refresh_rate <= 1 else round(self.refresh_rate)), delta_time)
                self.probability_storage[(start, end, len(intermediate_routers), free_slots, delta_time)] = link_entanglement_probability
            else:
                link_entanglement_probability = [1]
        else:
            link_entanglement_probability = self.probability_storage[(start, end, len(intermediate_routers), free_slots, delta_time)]
        return link_entanglement_probability

    def _refresh_quantum_links(self):
        if self.refresh_rate > 0 and self.env_steps - self.last_refresh >= (1 / self.refresh_rate):
            for edge in self.edges:
                if edge.dead:
                    continue
                start = edge.start
                end = edge.end

                link_entanglement_probabilities = self.get_link_entanglement_probability(start, end, edge.intermediate_routers, 1 + (self.n_quantum_links - len(edge.links) if self.use_available_quantum_storage else 0))
                probability = 0
                random_value = self.quantum_generator.random()
                for i in range(len(link_entanglement_probabilities)):
                    probability += link_entanglement_probabilities[i]
                    if probability > random_value:
                        for _ in range(i):
                            link_fidelity = self.initial_fidelity #np.power(self.timestep_decay, self.quantum_generator.random() * self.max_link_age)
                            for j in range(len(edge.intermediate_routers)):
                                part_fidelity = self.initial_fidelity #np.power(self.timestep_decay, self.quantum_generator.random() * self.max_link_age)
                                link_fidelity = round(QuantumLink.get_swap_fidelity(link_fidelity, part_fidelity, 1 - edge.intermediate_routers[j]), 9)

                            if link_fidelity >= QuantumLink.FIDELITY_THRESHOLD:
                                while len(edge.links) >= self.n_quantum_links:
                                    edge.links = edge.links[1:]
                                    self.total_active_quantum_links -= 1
                                edge.links.append(QuantumLink(min(start, end), max(start, end), link_fidelity, creation=self.env_steps * self.delta_time))
                                self.total_active_quantum_links += 1
                        break
                self.perform_auto_distillation(edge, self.env_steps * self.delta_time)
                edge.links.sort(key=lambda x: x.fidelity)
            self.last_refresh = self.env_steps

    def _update_edge_node_association(self):
        self.edge_node_association = {}

        for edge in self.edges:
            self.edge_node_association[(edge.start, edge.end)] = edge

    def get_random_decay(self):
        return self.timestep_decay

    def clean_reservations(self):
        if self.env_steps in self.reservation_cleanup:
            cleanup = self.reservation_cleanup[self.env_steps]
            for reservation in cleanup:
                reserved_links = self.edge_node_association[reservation["edge"]].reserved_links
                if reservation["reservation"] in reserved_links and reserved_links[reservation["reservation"]].id == reservation["id"]:
                    del reserved_links[reservation["reservation"]]


    def clean_quantum_links(self, edge):
        while len(edge.links) > 0 and edge.links[0].fidelity < QuantumLink.FIDELITY_THRESHOLD * 1.01:
            edge.links = edge.links[1:]

    def perform_auto_distillation(self, edge, time):
        start_links = len(edge.links)
        low_quality_links = []
        low_quality_created_links = []
        for link in edge.links:
            if link.fidelity < self.auto_distillation_threshold:
                if link.creation == time:
                    low_quality_created_links.append(link)
                else:
                    low_quality_links.append(link)

        current_distilled_link = None
        while len(low_quality_created_links) > 0:
            edge.links.remove(low_quality_created_links[0])
            if current_distilled_link is None:
                current_distilled_link = low_quality_created_links[0]
            else:
                current_distilled_link = QuantumLink.create_distillation(current_distilled_link, low_quality_created_links[0], time)
            start_links -= 1
            low_quality_created_links = low_quality_created_links[1:]
            if current_distilled_link.fidelity > self.auto_distillation_threshold:
                edge.links.append(current_distilled_link)
                start_links += 1
                current_distilled_link = None
        if current_distilled_link is not None:
            while len(low_quality_links) > 0:
                edge.links.remove(low_quality_links[0])
                current_distilled_link = QuantumLink.create_distillation(current_distilled_link, low_quality_links[0], time)
                low_quality_links = low_quality_links[1:]
                start_links -= 1
                if current_distilled_link.fidelity > self.auto_distillation_threshold:
                    break
            edge.links.append(current_distilled_link)
            start_links += 1
            current_distilled_link = None

    def initialize_quantum_links(self):
        if self.eval:
            max_time = 1
            current_time = 0
            timestep_length = .01
            while current_time < max_time:
                for edge in self.edges:
                    if len(edge.links) > 0:
                        for link in edge.links:
                            if self.use_realistic_decay:
                                link.decay_realistic((current_time - max_time), min(self.nodes[link.start].n_decoupling_pulses, self.nodes[link.end].n_decoupling_pulses))
                            else:
                                link.decay(pow(self.timestep_decay, timestep_length))
                        self.clean_quantum_links(edge)


                    link_entanglement_probabilities = self.get_link_entanglement_probability(edge.start, edge.end, edge.intermediate_routers, delta_time=timestep_length, free_slots=1 + (self.n_quantum_links - len(edge.links) if self.use_available_quantum_storage else 0))

                    probability = 0.0
                    random_value = self.quantum_generator.random()

                    for i in range(len(link_entanglement_probabilities)):
                        probability += link_entanglement_probabilities[i]
                        if random_value < probability:
                            for _ in range(i):
                                link_fidelity = self.initial_fidelity #np.power(self.timestep_decay, self.quantum_generator.random() * self.max_link_age)
                                for j in range(len(edge.intermediate_routers)):
                                    part_fidelity = self.initial_fidelity # np.power(self.timestep_decay, self.quantum_generator.random() * self.max_link_age)
                                    link_fidelity = round(QuantumLink.get_swap_fidelity(link_fidelity, part_fidelity, 1 - edge.intermediate_routers[j]), 9)

                                if link_fidelity >= QuantumLink.FIDELITY_THRESHOLD:
                                    while len(edge.links) > 0 and len(edge.links) >= self.n_quantum_links:
                                        edge.links = edge.links[1:]
                                    if len(edge.links) < self.n_quantum_links:
                                        edge.links.append(QuantumLink(edge.start, edge.end, link_fidelity, creation=current_time - max_time))
                            break
                    if self.auto_distillation_threshold > 0:
                        self.perform_auto_distillation(edge, current_time - max_time)
                    edge.links.sort(key=lambda x: x.fidelity)
                current_time += timestep_length
        else:
            for edge in self.edges:
                for index in range(self.n_quantum_links):
                    quantum_link = QuantumLink(edge.start, edge.end, 1, creation=0)
                    quantum_link.decay(self.timestep_decay ** (index * (60 / self.delta_time)))
                    edge.links.append(quantum_link)
                edge.links.sort(key=lambda x: x.fidelity)

    def get_swap_probability(self):
        return max(0, min(1, self.topology_generator.normal(self.swap_probability, self.swap_probability_std)))

    def _create_random_topology_from_topohub(self):
        import geopy
        network = topohub.get(self.topohub_topology)

        self.G = nx.Graph()
        self.nodes = []
        self.edges = []
        cluster = []

        name_id_mapping = {}
        min_lon = 360
        min_lat = 360
        for node in network.get("nodes"):
            position = node.get("pos")
            min_lon = min(min_lon, position[0])
            min_lat = min(min_lat, position[1])

        for i, node in enumerate(network.get("nodes")):
            name_id_mapping[node.get("id")] = i

            # add routers at random locations
            node_swap_probability = self.get_swap_probability()

            position = node.get("pos")
            node_lat = position[1]
            node_lon = position[0]

            pos = (
                geopy.distance.geodesic((min_lat, min_lon), (min_lat, node_lon)).km,
                geopy.distance.geodesic((min_lat, min_lon), (node_lat, min_lon)).km,
            )

            new_router = QuantumRepeater(len(self.nodes), pos[0], pos[1], swap_prob=node_swap_probability, decay=self.get_random_decay(), n_decoupling_pulses=self.get_n_decoupling_pulses())
            self.nodes.append(new_router)
            cluster.append(i)

            self.G.add_node(i, pos=(new_router.x, new_router.y))

        added_links = set()

        for i, edge in enumerate(network.get("links")):
            start = name_id_mapping[edge.get("source")]
            end = name_id_mapping[edge.get("target")]

            if (min(start, end), max(start, end)) not in added_links:
                added_links.add((min(start, end), max(start, end)))
            else:
                continue

            end_node = self.nodes[end]

            previous_node = self.nodes[start]
            if self.get_node_distance(self.nodes[start], self.nodes[end]) > MAX_LINK_LENGTH_REAL_TOPOLOGY:
                number_of_intermediate_routers = int(math.ceil(self.get_node_distance(self.nodes[start], self.nodes[end]) / MAX_LINK_LENGTH_DIVISOR_REAL_TOPOLOGY)) - 1
                x1 = self.nodes[start].x
                x2 = self.nodes[end].x
                y1 = self.nodes[start].y
                y2 = self.nodes[end].y
                for j in range(number_of_intermediate_routers):
                    node_swap_probability = self.get_swap_probability()
                    intermediate_node = QuantumRepeater(len(self.nodes), x1 + (x2 - x1) / (number_of_intermediate_routers + 1.) * (j + 1), y1 + (y2 - y1) / (number_of_intermediate_routers + 1.) * (j + 1), swap_prob=node_swap_probability, decay=self.get_random_decay(), n_decoupling_pulses=self.get_n_decoupling_pulses())
                    intermediate_node.source_destination_valid = False
                    self.nodes.append(intermediate_node)
                    self.G.add_node(intermediate_node.id, pos=(intermediate_node.x, intermediate_node.y))

                    new_edge = Edge(min(previous_node.id, intermediate_node.id), max(previous_node.id, intermediate_node.id), [])

                    edge_id = len(self.edges)
                    self.edges.append(new_edge)
                    previous_node.edges.append(edge_id)
                    intermediate_node.edges.append(edge_id)
                    if intermediate_node.id not in previous_node.neighbors:
                        previous_node.neighbors.append(intermediate_node.id)
                    if previous_node.id not in intermediate_node.neighbors:
                        intermediate_node.neighbors.append(previous_node.id)

                    self.G.add_edge(
                        new_edge.start,
                        new_edge.end,
                        weight=1,
                    )

                    previous_node = intermediate_node

            new_edge = Edge(min(previous_node.id, end_node.id), max(previous_node.id, end_node.id), [])
            edge_id = len(self.edges)
            self.edges.append(new_edge)
            previous_node.edges.append(edge_id)
            end_node.edges.append(edge_id)
            if end_node.id not in previous_node.neighbors:
                previous_node.neighbors.append(end_node.id)
            if previous_node.id not in end_node.neighbors:
                end_node.neighbors.append(previous_node.id)

            self.G.add_edge(
                new_edge.start,
                new_edge.end,
                weight=1,
            )

        for node in self.nodes:
            for neigh in self.nodes:
                if node != neigh and (neigh.id not in node.neighbors) and (not neigh.source_destination_valid) and len(neigh.edges) < self.neighbor_count and (not node.source_destination_valid) and len(node.edges) < self.neighbor_count:
                    if self.get_node_distance(node, neigh) < MAX_LINK_LENGTH_REAL_TOPOLOGY:
                        new_edge = Edge(min(node.id, neigh.id), max(node.id, neigh.id), [])
                        edge_id = len(self.edges)
                        self.edges.append(new_edge)
                        node.edges.append(edge_id)
                        neigh.edges.append(edge_id)
                        if neigh.id not in node.neighbors:
                            node.neighbors.append(neigh.id)
                        if node.id not in neigh.neighbors:
                            neigh.neighbors.append(node.id)

                        self.G.add_edge(
                            new_edge.start,
                            new_edge.end,
                            weight=1,
                        )

        self.initialize_quantum_links()

        if len(self.nodes) > self.n_nodes:
            print("Network cannot be created because it would require too many nodes.")
            exit(1)

        # order router edges by neighbor node id to remove symmetries
        for i in range(len(self.nodes)):
            self.nodes[i].edges = sorted(
                self.nodes[i].edges,
                key=lambda edge_index: self.edges[edge_index].get_other_node(i),
            )

        sum = 0
        count = 0
        for edge in self.edges:
            sum += self.get_node_distance(self.nodes[edge.start], self.nodes[edge.end])
            count += 1

        print("Average node distance: " + str(sum / count))

    def calculate_base_entanglement_probability_ground(self, link_length):
        return pow(10, -self.attenuation_coefficient * link_length / 10)

    def calculate_base_entanglement_probability_air(self, link_length):
        return pow(self.antenna_directivity, 2) *  (self.beam_wave_length / (4 * math.pi / link_length))

    def calculate_link_entanglement_probability(self, base_probability, speed_of_light, link_length, intermediate_routers, free_slots, delta_time=None):
        if delta_time is None:
            delta_time = self.delta_time
        probabilities = []
        n = min(math.ceil(free_slots * speed_of_light * delta_time / (2 * link_length)), MAX_QUBITS_GENERATION_RATE * delta_time)
        for k in range(0, min(n, 5)):
            probabilities.append(scipy.special.comb(n, k) * math.pow(base_probability, k) * math.pow(1 - base_probability, n - k))
        probabilities.append(max(1 - sum(probabilities), 0))

        if len(intermediate_routers) > 0:
            aggregate_probabilities = [0] * len(probabilities)
            probability = 0
            for i in range(len(probabilities)):
                index = len(probabilities) - i - 1
                probability_more_than = 0
                for j in range(index):
                    probability_more_than += probabilities[j]
                probability_more_than = 1 - probability_more_than
                aggregate_probabilities[index] = probability_more_than ** (len(intermediate_routers) + 1) - probability
                probability += aggregate_probabilities[index]
            probabilities = aggregate_probabilities


        return probabilities #1 - pow(1 - base_probability, speed_of_light / (2 * link_length) * delta_time * free_slots)

    def calculate_node_positions(self, G, number_of_nodes):
        scale = min(self.world_width, self.world_height) / 2.25 * math.sqrt(number_of_nodes / 100.)
        distance_optimal = float("inf")
        optimal_segments = -1
        optimal_total_radius = -1
        segments = 1
        while True:
            total_radius = 0
            for j in range(1, segments + 1):
                total_radius += scale / segments * j * math.pi * 2
            distance_per_node_circular = total_radius / (number_of_nodes - 1)
            distance_per_node_radius = scale / segments
            distance = abs(distance_per_node_circular - distance_per_node_radius)
            if distance < distance_optimal:
                distance_optimal = distance
                optimal_segments = segments
                optimal_total_radius = total_radius
            segments += 1

            if distance_per_node_radius < distance_per_node_circular:
                break

        nodes_per_circle = [1]
        radius = scale / optimal_segments
        for j in range(1, optimal_segments + 1):
            nodes_per_circle.append(round(radius * j / optimal_total_radius * number_of_nodes * math.pi * 2))

        difference = number_of_nodes - sum(nodes_per_circle)

        for j in range(1, len(nodes_per_circle)):
            nodes_per_circle[j] += math.floor(difference / (len(nodes_per_circle) - 1))
        difference = number_of_nodes - sum(nodes_per_circle)
        nodes_per_circle[-1] += difference

        positions = {}
        segment = 0
        nodes_per_segment = 0
        circle_angular_offset = 0
        for node in G.nodes():
            angular_offset = self.topology_generator.normal(
                math.pi * 2 * nodes_per_segment / nodes_per_circle[segment] + circle_angular_offset,
                math.pi * 2 / nodes_per_circle[segment] * .1)
            radial_offset = self.topology_generator.normal(radius * segment, radius * .1)
            position = [math.sin(angular_offset) * radial_offset, math.cos(angular_offset) * radial_offset]
            positions[node] = position

            nodes_per_segment += 1
            if nodes_per_segment >= nodes_per_circle[segment]:
                segment += 1
                nodes_per_segment = 0
                circle_angular_offset = self.topology_generator.random() * math.pi * 2

        return positions

    def _create_random_topology_from_template(self):
        """
        Creates a new random topology. This is based on the implementation from networkx.
        """
        # self.G = nx.generators.circular_ladder_graph(int(self.n_nodes / 2))

        self.G = nx.Graph()
        offset = 0
        positions = {}
        number_of_ground_nodes = self.n_nodes_connected

        number_of_nodes = number_of_ground_nodes
        node_ids = range(offset, offset + number_of_nodes)
        G = nx.generators.complete_graph(node_ids)
        positions = {**positions, **self.calculate_node_positions(G, number_of_nodes)}
        self.G = nx.union(self.G, G)
        offset += number_of_nodes

        cluster = []
        self.nodes = []
        self.edges = []
        self.min_active_quantum_links = 0
        self.total_active_quantum_links = 0
        self.max_active_quantum_links = 0
        self.average_active_quantum_links = 0

        nx.set_node_attributes(self.G, positions, "pos")

        for i in range(number_of_ground_nodes):
            # add routers at random locations
            node_swap_probability = self.get_swap_probability()

            pos = positions[i]

            new_router = QuantumRepeater(len(self.nodes), pos[0], pos[1], swap_prob=node_swap_probability, decay=self.get_random_decay(), n_decoupling_pulses=self.get_n_decoupling_pulses())
            self.nodes.append(new_router)
            cluster.append(i)

        G_without_edges = nx.create_empty_copy(self.G)
        for node in G_without_edges.nodes():
            for neighbor in G_without_edges.nodes():
                distance = self.get_node_distance(self.nodes[neighbor], self.nodes[node])
                if node != neighbor and distance <= MAX_LINK_LENGTH:
                    G_without_edges.add_edge(node, neighbor)
        self.G = G_without_edges

        exceeding_nodes = []
        for node in range(number_of_ground_nodes):
            if len(nx.edges(self.G, [node])) > self.neighbor_count:
                exceeding_nodes.append(node)

        distances = {}
        for edge in self.G.edges:
            distance = self.get_node_distance(self.G.nodes[edge[0]], self.G.nodes[edge[1]])
            distances[edge] = distance ** 4
        nx.set_edge_attributes(self.G, distances, "length")

        failures = 0
        if len(exceeding_nodes) > 0:
            while failures < 10:
                current_node_index = self.topology_generator.integers(len(exceeding_nodes))
                if len(nx.edges(self.G, [exceeding_nodes[current_node_index]])) <= self.neighbor_count:
                    exceeding_nodes.remove(exceeding_nodes[current_node_index])
                    continue

                current_node = exceeding_nodes[current_node_index]
                current_node_edges = list(nx.edges(self.G, [current_node]))
                distance_sum = 0
                for edge in current_node_edges:
                    distance_sum += self.G.get_edge_data(edge[0], edge[1])["length"]
                random_value = self.topology_generator.random() * distance_sum
                current_edge = None
                for edge in current_node_edges:
                    random_value -= self.G.get_edge_data(edge[0], edge[1])["length"]
                    if random_value < 0:
                        current_edge = edge
                        break
                assert (current_edge is not None)

                if current_edge[0] >= number_of_ground_nodes or current_edge[1] >= number_of_ground_nodes:
                    continue
                if len(self.G.edges(current_edge[0])) < 2 or len(self.G.edges(current_edge[1])) < 2:
                    failures += 1
                    continue
                failures = 0

                self.G.remove_edge(*current_edge)

                if len(nx.edges(self.G, [current_edge[0]])) <= self.neighbor_count:
                    if current_edge[0] in exceeding_nodes:
                        exceeding_nodes.remove(current_edge[0])
                if len(nx.edges(self.G, [current_edge[1]])) <= self.neighbor_count:
                    if current_edge[1] in exceeding_nodes:
                        exceeding_nodes.remove(current_edge[1])

                if len(exceeding_nodes) == 0:
                    break

        if failures >= 10:
            self.G = None
            return

        nx.set_edge_attributes(self.G, values=1, name = 'weight')

        t_edge = 0

        for u,v,a in self.G.edges(data=True):
            new_edge = Edge(min(u,v), max(u,v), [])
            self.edges.append(new_edge)
            self.nodes[u].edges.append(t_edge)
            self.nodes[v].edges.append(t_edge)
            if v not in self.nodes[u].neighbors:
                self.nodes[u].neighbors.append(v)
            if u not in self.nodes[v].neighbors:
                self.nodes[v].neighbors.append(u)

            t_edge += 1

        distances_debug = []
        for edge in self.edges:
            distances_debug.append(self.get_node_distance(self.nodes[edge.start], self.nodes[edge.end]))
        print("Average distance " + str(np.array(distances_debug).mean()))

        self.initialize_quantum_links()

        # order router edges by neighbor node id to remove symmetries
        for i in range(self.n_nodes_connected):
            self.nodes[i].edges = sorted(
                self.nodes[i].edges,
                key=lambda edge_index: self.edges[edge_index].get_other_node(i),
            )

    def get_edge(self, start, end):
        """
        Retrieve the edge object between the start and end nodes.
        """
        for edge in self.edges:
            if (edge.start == start and edge.end == end) or (
                edge.start == end and edge.end == start
            ):
                return edge
        raise ValueError(f"No edge found between nodes {start} and {end}")

    def get_node_distance(self, node1, node2):
        if isinstance(node1, QuantumRepeater) and isinstance(node2, QuantumRepeater):
            edge = (min(node1.id, node2.id), max(node1.id, node2.id))
            if not edge in self.node_distances:
                self.node_distances[edge] = math.sqrt(self.get_node_distance_squared(node1, node2))
            return self.node_distances[edge]

        return math.sqrt(self.get_node_distance_squared(node1, node2))

    def fail_node(self, node_id):
        node = self.nodes[node_id]
        edges_to_remove = []
        for edge_id in node.edges:
            edges_to_remove.append(edge_id)
            edge = self.edges[edge_id]
            neighbor_id = edge.get_other_node(node_id)
            neighbor = self.nodes[neighbor_id]
            neighbor.neighbors.remove(node_id)
            neighbor.edges.remove(edge_id)
            node.swap_prob = 0
            node.failure_step = self.env_steps

            if self.G.has_edge(node_id, neighbor_id):
                self.G.remove_edge(node_id, neighbor_id)

        edges_to_remove.sort(reverse=True)
        for edge_id in edges_to_remove:
            self.edges[edge_id].dead = True
            self.edges[edge_id].dead_step = self.env_steps
            node.edges.remove(edge_id)

        self._update_nodes_adjacency()
        self._update_shortest_paths()
        self.update_shortest_available_paths()


    def get_node_distance_squared(self, node1, node2):
        if isinstance(node1, QuantumRepeater) and isinstance(node2, QuantumRepeater):
            return (node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2
        else:
            return (node1["pos"][0] - node2["pos"][0]) ** 2 + (node1["pos"][1] - node2["pos"][1]) ** 2

    def _create_random_topology(self, neighbor_count):
        """
        Creates a new random topology. This is based on the implementation
        by Jiang et al. https://github.com/PKU-RL/DGN/blob/master/Routing/routers.py
        used for their DGN paper https://arxiv.org/abs/1810.09202.
        """
        # otherwise create a new topology
        self.G = nx.Graph()
        self.nodes = []
        self.edges = []
        self.min_active_quantum_links = 0
        self.total_active_quantum_links = 0
        self.max_active_quantum_links = 0
        self.average_active_quantum_links = 0

        t_edge = 0

        for i in range(self.n_nodes_connected):
            # add routers at random locations
            node_swap_probability = max(
                0,
                min(
                    1,
                    self.topology_generator.normal(self.swap_probability, self.swap_probability_std),
                ),
            )

            new_router = QuantumRepeater(
                len(self.nodes),
                self.topology_generator.random() * self.world_width,
                self.topology_generator.random() * self.world_height,
                swap_prob=node_swap_probability,
                n_decoupling_pulses = self.get_n_decoupling_pulses()
            )
            self.nodes.append(new_router)
            self.G.add_node(i, pos=(new_router.x, new_router.y))

        for i in range(self.n_nodes_connected):
            # calculate (squared) distances to all other routers
            self.dis = []
            for j in range(self.n_nodes_connected):
                self.dis.append(
                    [
                        (self.nodes[j].x - self.nodes[i].x) ** 2
                        + (self.nodes[j].y - self.nodes[i].y) ** 2,
                        j,
                    ]
                )

            # sort by distance
            self.dis.sort(key=lambda x: x[0], reverse=False)

            # find new neighbors
            # exclude index 0 as we always have distance 0 to ourselves
            for j in range(1, self.n_nodes_connected):
                # we have found enough neighbors => break
                if len(self.nodes[i].neighbors) == neighbor_count:
                    break

                # check for neighbor candidates
                candidate_sq_dist, candidate_idx = self.dis[j]
                if (
                    len(self.nodes[candidate_idx].neighbors) < neighbor_count
                    and i not in self.nodes[candidate_idx].neighbors
                ):
                    # append new neighbor
                    self.nodes[i].neighbors.append(candidate_idx)
                    self.nodes[candidate_idx].neighbors.append(i)

                    if i < candidate_idx:
                        src_node = i
                        dst_node = candidate_idx
                    else:
                        src_node = candidate_idx
                        dst_node = i

                    link_length = self.get_node_distance(self.nodes[src_node], self.nodes[dst_node])
                    if _is_ground_connection(self.nodes[src_node], self.nodes[dst_node]):
                        base_entanglement_probability = self.calculate_base_entanglement_probability_ground(link_length)
                        speed_of_light = self.speed_of_light_glass
                    else:
                        base_entanglement_probability = self.calculate_base_entanglement_probability_air(link_length)
                        speed_of_light = self.speed_of_light_air
                    link_entanglement_probability = 1 - self.calculate_link_entanglement_probability(base_entanglement_probability, speed_of_light, link_length, [], 1)[0]
                    quantum_links = []
                    for _ in range(0, self.n_quantum_links):
                        if self.quantum_generator.random() < link_entanglement_probability:
                            link_fidelity = np.power(
                                self.timestep_decay,
                                self.quantum_generator.random() * self.max_link_age,
                            )
                            if link_fidelity >= QuantumLink.FIDELITY_THRESHOLD:
                                quantum_links.append(
                                    QuantumLink(src_node, dst_node, link_fidelity)
                                )
                    quantum_links.sort(key=lambda x: x.fidelity)

                    # create edges, always sorted by index
                    new_edge = Edge(src_node, dst_node, quantum_links)
                    self.min_active_quantum_links = max(
                        self.min_active_quantum_links, len(quantum_links)
                    )
                    self.total_active_quantum_links += len(quantum_links)
                    self.average_active_quantum_links += (
                        link_entanglement_probability * self.n_quantum_links
                    )
                    self.max_active_quantum_links += self.n_quantum_links

                    self.edges.append(new_edge)
                    self.nodes[src_node].edges.append(t_edge)
                    self.nodes[dst_node].edges.append(t_edge)
                    self.G.add_edge(
                        new_edge.start,
                        new_edge.end,
                        weight=1,
                    )
                    # TODO: Weight = 1 is just a preliminary

                    t_edge += 1

        # order router edges by neighbor node id to remove symmetries
        for i in range(self.n_nodes_connected):
            self.nodes[i].edges = sorted(
                self.nodes[i].edges,
                key=lambda edge_index: self.edges[edge_index].get_other_node(i),
            )

    def get_diameter(self):
        return self.diameter

    def _check_topology_constraints(self, neighbor_count):
        """
        Check if the current network topology fulfills the constraints, meaning it is
        connected and all nodes have three neighbors.

        :return: whether the topology is valid.
        """
        # for the case that there is no isolated island but nodes with less than k edges
        for i in range(self.n_nodes_connected):
            if len(self.nodes[i].neighbors) < neighbor_count:
                return False

        # this means not every nodes is reachable, we have got isolated islands
        if not nx.is_connected(self.G):
            return False

        return True

    def set_seeds(self, seed):
        self.quantum_generator = np.random.default_rng(seed)
        self.topology_generator = np.random.default_rng(seed)
        self.swap_generator = np.random.default_rng(seed)
        self.agent_generator = np.random.default_rng(seed)
        self.node_generator = np.random.default_rng(seed)

    def _create_valid_network(
        self, seed_list=None, seed_index=None, seeds_exclude=None, from_template=True
    ):
        """
        Generates a network based on a list of seeds.

        :param seed_list: List of seeds, can be None to create new valid topology
        :param seed_index: Index in seed list, chooses random index if None
        :param seeds_exclude: List of seeds (for random generation) are excluded
        :returns: the seed used to generate the network topology
        """

        # set seed for topology generation
        no_seed_provided = seed_list is None or len(seed_list) == 0
        if no_seed_provided:
            seed = np.random.randint(2**31 - 1)
            while seeds_exclude is not None and seed in seeds_exclude:
                seed = np.random.randint(2**31 - 1)
        elif seed_index is not None:
            seed = seed_list[seed_index]
        else:
            # choose one of the seeds from the list
            seed = np.random.choice(seed_list)

        self.set_seeds(seed)

        if self.node_degree is None:
            neighbor_count = self.neighbor_count
        else:
            neighbor_count = self.node_degree

        if not from_template:
            self.repetitions = 0
            while True:
                self.node_distances = {}
                self._create_random_topology(neighbor_count=neighbor_count)
                self.repetitions += 1
                if self._check_topology_constraints(neighbor_count=neighbor_count):
                    break
        else:
            if self.topohub_topology is None:
                while True:
                    self.node_distances = {}
                    self._create_random_topology_from_template()

                    if self.G is not None and nx.is_connected(self.G) and nx.diameter(self.G) >= self.fixed_path_length:
                        break
            else:
                self._create_random_topology_from_topohub()

        self.fill_missing()
        self._update_edge_node_association()

        self._update_shortest_paths()
        self._update_nodes_adjacency()
        self.current_topology_seed = seed

        # return the seed that was used to create this topology
        return seed

    def get_edge_weight(self, start, end, edge_attributes):
        if self.nodes[end].swap_prob > 0:
            return 1 / self.nodes[end].swap_prob
        else:
            return float("inf")

    def _update_shortest_paths(self):
        """
        Calculates shortest paths and stores them in self.shortest_paths. The
        corresponding weights (distances) are stored in self.shortest_paths_weights
        """
        self.shortest_paths = dict(
            nx.shortest_path(self.G, weight=self.get_edge_weight_binary)
        )
        self.shortest_paths_weights = dict(
            nx.shortest_path_length(self.G, weight=self.get_edge_weight_binary)
        )

    def get_path_length(self, start, end, edge_attributes=None):
        return self.get_node_distance(self.nodes[start], self.nodes[end])

    def get_edge_weight_binary(self, start, end, edge_attributes):
        if self.eval:
            return 1
        if self.nodes[end].swap_prob > 0.0:
            return 1
        else:
            return float("inf")

    def randomize_edge_weights(self, mode: str, **kwargs):
        """
        Randomizes edge weights in the graph (at runtime).

        :param mode: `shuffle` to shuffle existing weights, `randint` with additional
                     kwargs `low` and `high` to create new random weights
        :returns: tuple of (proportion of changed first hops on shortest paths, proportion
                  of changed shortest paths, proportion of changed shortest path lengths)
        """
        if mode == "shuffle":
            edge_lengths = np.array([e.length for e in self.edges])
            self.topology_generator.shuffle(edge_lengths)
            for i, e in enumerate(self.edges):
                e.length = edge_lengths[i]
        elif mode == "randint":
            for e in self.edges:
                e.length = self.topology_generator.integers(kwargs["low"], kwargs["high"])
        elif mode == "bottleneck-971182936":
            edge_update_list = [
                (2, 7),
            ]
            for e in self.edges:
                for start, end in edge_update_list:
                    if e.start == start and e.end == end:
                        e.length = 10
                        break
            if self.current_topology_seed != 971182936:
                print("Warning: mode only meant to be used in graph 971182936.")
        else:
            raise ValueError(f"Unknown mode {mode}")

        old_shortest_paths = self.shortest_paths.copy()
        old_shortest_path_weights = self.shortest_paths_weights.copy()

        for e in self.edges:
            self.G[e.start][e.end]["weight"] = e.length

        self._update_shortest_paths()

        # check how much has changed
        n_paths = self.n_nodes * (self.n_nodes - 1)
        n_paths_changed_first_hop = 0
        n_paths_changed = 0
        n_path_weights_changed = 0
        for a in range(self.n_nodes):
            for b in range(self.n_nodes):
                if a == b:
                    continue
                if self.shortest_paths[a][b][1] != old_shortest_paths[a][b][1]:
                    n_paths_changed_first_hop += 1
                if self.shortest_paths[a][b] != old_shortest_paths[a][b]:
                    n_paths_changed += 1
                if self.shortest_paths_weights[a][b] != old_shortest_path_weights[a][b]:
                    n_path_weights_changed += 1

        return (
            n_paths_changed_first_hop / n_paths,
            n_paths_changed / n_paths,
            n_path_weights_changed / n_paths,
        )

    def freeze_sequential_topology_seeds(self):
        self.sequential_topology_seeds_frozen = True

    def next_topology_seed_index(self, advance_index=True):
        seed_index = (
            self.sequential_topology_index
            if len(self.seeds) > 1 and self.sequential_topology_seeds
            else None
        )
        if seed_index is not None and advance_index:
            self.sequential_topology_index = (seed_index + 1) % len(self.seeds)
        return seed_index

    def reset(self):
        self.env_steps = 0

        self.reservation_cleanup = {}

        seed_index = self.next_topology_seed_index(
            advance_index=not self.sequential_topology_seeds_frozen
        )
        self._create_valid_network(self.seeds, seed_index, self.exclude_seeds)# , not self.eval)
        self.sampled_topology_seeds.append(self.current_topology_seed)
        self.probability_storage = {}

        self.node_distances = {}

        self.done = False

        self.current_max_env_steps = int(
            max(self.next_max_env_steps, self.current_max_env_steps)
        )
        self.next_max_env_steps = self.min_env_steps
        self.last_refresh = 0

        self.nodes_failed = 0

        self.coordination_node = None
        self.coordinator_distance = None
        self.coordinator_paths = None
        self.coordination_messages = []
        self.coordinator_G = None
        self.node_Gs = None

        self.transmitted_messages = 0
        self.transmitted_message_load_edges = {}
        self.transmitted_message_load_nodes = {}

        if nx.is_connected(self.G):
            self.diameter = nx.diameter(self.G)

    def render(self, G=None, show_plot=True, filename=None, fig=None, ax=None, fancy=False, starts=None, targets=None, positions=None):
        if G is None:
            G = self.G
        pos = nx.get_node_attributes(G, "pos")
        for i in range(G.number_of_nodes()):
            if i not in pos:
                pos[i] = [self.nodes[i].x, self.nodes[i].y]

        link_colors = []
        for edge in self.edges:
            if G.has_edge(edge.start, edge.end):
                if not edge.dead:
                    link_colors.append(len(edge.links))
                else:
                    link_colors.append(0)

        node_cmap = colormaps['summer_r']

        node_colors = []
        for i in range(G.number_of_nodes()):
            if self.coordination_node is not None and i == self.coordination_node:
                node_colors.append("blue")
            elif starts is not None and i in starts:
                node_colors.append("coral")
            elif targets is not None and i in targets:
                node_colors.append("lawngreen")
            else:
                node_colors.append(node_cmap(int(self.nodes[i].swap_prob * 256)))

        max_x = -float("inf")
        max_y = -float("inf")
        min_x = float("inf")
        min_y = float("inf")
        for node in self.nodes:
            max_x = max(max_x, node.x)
            min_x = min(min_x, node.x)
            max_y = max(max_y, node.y)
            min_y = min(min_y, node.y)

        if show_plot:
            plt.figure(figsize=(12, 12 / float(max_x - min_x) * (max_y - min_y)))

        nx.draw_networkx(
            G,
            pos = pos,
            node_size = 1000,
            with_labels = True,
            node_color = node_colors,
#                vmin = 0,
#                vmax = 1,
#                cmap = colormaps["summer_r"],
            edge_vmin = 0,
            edge_vmax = self.n_quantum_links,
            edge_color = link_colors,
            #edge_cmap = colormaps["Blues"],
            edge_cmap=colormaps["Blues"],
        )


        if show_plot:
            plt.box(False)
            if filename is not None:
                plt.savefig(filename, bbox_inches='tight')
            else:
                plt.show(bbox_inches='tight')
            plt.close()

    def get_nodes_adjacency(self):
        """
        Get the adjacency matrix for all routers (nodes) in the network.

        return: adjacency matrix of size (n_router, n_router)
        """
        return self.adj_matrix

    def get_nodes_adjacency_for_env(self, env_id):
        if env_id in self.virtual_adj_matrix:
            return self.virtual_adj_matrix[env_id]
        return self.get_nodes_adjacency()

    def get_nodes(self, env_id):
        if env_id in self.virtual_subgraphs:
            return self.virtual_subgraphs[env_id]
        return self.nodes

    def set_virtual_node_group(self, env_id, nodes):
        self.virtual_node_groups[env_id] = nodes

        virtual_nodes = []
        for node in self.nodes:
            virtual_node = copy.deepcopy(node)
            virtual_node.edges = []
            virtual_node.neighbors = []

            if node.id in nodes:
                for edge in node.edges:
                    if self.edges[edge].get_other_node(node.id) in nodes:
                        virtual_node.edges.append(edge)

                for neighbor in node.neighbors:
                    if neighbor in nodes:
                        virtual_node.neighbors.append(neighbor)

            virtual_nodes.append(virtual_node)

        self.virtual_subgraphs[env_id] = virtual_nodes


    def _update_nodes_adjacency(self):
        self.adj_matrix = np.eye(self.n_nodes, self.n_nodes, dtype=np.int8)
        for env_id in self.virtual_node_groups:
            self.virtual_adj_matrix[env_id] = np.eye(self.n_nodes, self.n_nodes, dtype=np.int8)
        for i in range(self.n_nodes):
            for neighbor in self.nodes[i].neighbors:
                if self.nodes[i].type == self.nodes[neighbor].type\
                        or (self.nodes[i].type == QuantumRepeater.Type.NORMAL and self.nodes[neighbor].type == QuantumRepeater.Type.GROUND_STATION)\
                        or (self.nodes[neighbor].type == QuantumRepeater.Type.NORMAL and self.nodes[i].type == QuantumRepeater.Type.GROUND_STATION):
                    self.adj_matrix[i][neighbor] = 1

                    for env_id in self.virtual_node_groups:
                        if i in self.virtual_node_groups[env_id] and neighbor in self.virtual_node_groups[env_id]:
                            self.virtual_adj_matrix[env_id][i][neighbor] = 1

    def _move_nodes(self):
        for node in self.nodes:
            if node.movable:
                node.x = (node.x + node.movement_vector[0] + self.world_width) % self.world_width
                node.y = (node.y + node.movement_vector[1] + self.world_height) % self.world_height
                node.z += node.movement_vector[2]

                nx.set_node_attributes(self.G, {node.id: (node.x, node.y)}, name="pos")

    def calculate_decay(self, start, end):
        return self.nodes[start].decay * self.nodes[end].decay

    def pre_step(self):
        for edge in self.edges:
            for link in edge.links:
                if self.use_realistic_decay:
                    link.decay_realistic(self.env_steps * self.delta_time, min(self.nodes[link.start].n_decoupling_pulses, self.nodes[link.end].n_decoupling_pulses))
                else:
                    link.decay(self.calculate_decay(edge.start, edge.end))
            self.clean_quantum_links(edge)

        self._refresh_quantum_links()

        self._move_nodes()

    def step(self):
        self.env_steps += 1

        if self.coordination_node is not None:
            self.send_update_messages()
        self.clean_reservations()

    def is_graph_quantum_connected(self):
        dead_edges = []
        for edge in self.edges:
            if len(edge.links) == 0:
                dead_edges.append(edge)

        G = self.G.copy()

        for dead_edge in dead_edges:
            if G.has_edge(dead_edge.start, dead_edge.end):
                G.remove_edge(dead_edge.start, dead_edge.end)

        return nx.is_connected(G)

    def set_done(self):
        self.done = True

    def is_done(self):
        return self.done
