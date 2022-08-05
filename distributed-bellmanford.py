#!/usr/bin/python3

# Implements message-passing for distributed Bellman-Ford for SSSP

import enum
from math import inf

class MessageType(enum.Enum):
    source = 1
    advertise = 2

class Node:
    def __init__(self, id, g):
        self.id = id
        self.parent = None
        self.d = inf
        self.neighbors = {}
        self.inbox = []
        self.input = []
        self.output = []
        self.graph = g
        self.time = 0
        self.state = 0
        self.echo_count = 0

    # Transfer outbox to inbox of neighbors
    def send(self):
        for (message, x, payload) in self.output:
            self.graph.v[x].inbox.append((message, self.id, payload))
            self.graph.message += 1
        self.output = []

    def process_input(self):
        for (message, x, payload) in self.input:
            self.process_message(message, x, payload)
        self.input = []


    # Put things into outbox
    def process_message(self, message, x, payload):
        print(f"{self.id}: Processing message {message} from {x}")

        if message == MessageType.source:
            self.d = 0
            self.parent = self.id
            for v in self.neighbors.keys():
                self.output.append((MessageType.advertise, v, self.d))
        if message == MessageType.advertise:
            if self.parent != self.id and payload + self.neighbors[x] < self.d:
                self.d = payload + self.neighbors[x]
                self.parent = x
                for v in self.neighbors.keys():
                    self.output.append((MessageType.advertise, v, self.d))

    def empty_inbox(self):
        for i in self.inbox:
            self.input.append(i)
        self.inbox = []
        
    def tick(self):
        self.process_input()
        self.send()

    def tock(self):
        self.empty_inbox()
        self.time += 1

class Graph:

    def __init__(self):
        self.v = {}
        self.time = 0
        self.message = 0

    def add_node(self, v1):
        if v1 not in self.v:
            self.v[v1] = Node(v1, self)
        else:
            print(f"WARNING: double-adding node {v1}")

    def add_edge(self, v1, v2, w):
        if not (v1 in self.v and v2 in self.v):
            print(f"ERROR: attempting to add edges between non-existent nodes!")
            return
        self.v[v1].neighbors[v2] = w
        self.v[v2].neighbors[v1] = w

    def play_round(self):
        for v in self.v:
            self.v[v].tick()
        for v in self.v:
            self.v[v].tock()
        self.time += 1

    def test_message(self, i, rounds):
        self.v['A'].inbox.append((MessageType.source, None, None))
        for t in range(rounds):
            print(f"Round: {self.time}")
            self.play_round()
            # self.debug_print()

    def debug_print(self):
        for v in self.v.values():
            print(f"Node {v.id}: d: {v.d} parent: {v.parent} state: {v.state}", flush=True)


def main():
    g = Graph()
    g.add_node('A')
    g.add_node('B')
    g.add_node('C')
    g.add_node('D')
    g.add_node('E')
    g.add_node('F')
    g.add_node('G')
    g.add_node('H')
    g.add_edge('A', 'B', 2)
    g.add_edge('A', 'C', 3)
    g.add_edge('A', 'D', 5)
    g.add_edge('B', 'E', 3)
    g.add_edge('B', 'F', 6)
    g.add_edge('C', 'D', 1)
    g.add_edge('C', 'E', 1)
    g.add_edge('D', 'E', 6)
    g.add_edge('D', 'H', 5)
    g.add_edge('E', 'F', 4)
    g.add_edge('E', 'G', 2)
    g.add_edge('E', 'H', 4)
    g.add_edge('F', 'G', 3)
    g.test_message(0, 12)
    g.debug_print()
    print(f"Message complexity: {g.message}")


if __name__=='__main__':
    main()