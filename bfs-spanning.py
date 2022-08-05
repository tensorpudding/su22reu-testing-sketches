#!/usr/bin/python3

# Implements message-passing for distributed BFS spanning tree creation
# Each node keeps track of its parent and children in the BFS

import enum

class Message(enum.Enum):
    accept = 1
    invite = 2
    echo = 3
    done = 4
    start = 5

class Node:
    def __init__(self, id, g):
        self.id = id
        self.parent = None
        self.neighbors = []
        self.children = []
        self.inbox = []
        self.input = []
        self.output = []
        self.graph = g
        self.time = 0
        self.state = 0
        self.echo_count = 0
        self.accept_timer = 0

    def send(self, v, message):
        self.output.append((message, v))
        self.graph.message += 1

    # Transfer outbox to inbox of neighbors
    def send(self):
        for (message, x) in self.output:
            self.graph.v[x].inbox.append((message, self.id))
            self.graph.message += 1
        self.output = []

    def process_input(self):
        for (message, x) in self.input:
            self.process_message(message, x)
        self.input = []
        if (self.accept_timer == 0 and self.state == 1 and len(self.children) == self.echo_count):
            if self.parent != None:
                self.state = 2
                self.output.append((Message.echo, self.parent)) 
            else:
                self.state = 3
                for v in self.children:
                    self.output.append((Message.done, v))

    # Put things into outbox
    def process_message(self, message, x):
        print(f"{self.id}: Processing message {message} from {x}")

        if message == Message.invite:
            if self.state == 0:
                self.output.append((Message.accept, x))
                self.parent = x
                self.state = 1
                self.accept_timer = 2
                for v in self.neighbors:
                    if v != x:
                        self.output.append((Message.invite, v))

        if message == Message.accept:
            self.children.append(x)
        if message == Message.start:
            self.parent = None
            self.state = 1
            self.accept_timer = 3
            for v in self.neighbors:
                self.output.append((Message.invite, v))
        if message == Message.echo:
            self.echo_count += 1

        if message == Message.done:
            self.echo_count = 0
            self.state = 3
            for v in self.children:
                self.output.append((Message.done, v))

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
        self.accept_timer = max(0, self.accept_timer - 1)

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

    def add_edge(self, v1, v2):
        if not (v1 in self.v and v2 in self.v):
            print(f"ERROR: attempting to add edges between non-existent nodes!")
            return
        self.v[v1].neighbors.append(v2)
        self.v[v2].neighbors.append(v1)

    def play_round(self):
        for v in self.v:
            self.v[v].tick()
        for v in self.v:
            self.v[v].tock()
        self.time += 1

    def test_message(self, i, rounds):
        self.v[i].inbox.append((Message.start, None))
        for t in range(rounds):
            print(f"Round: {self.time}")
            self.play_round()
            # self.debug_print()

    def debug_print(self):
        for v in self.v.values():
            print(f"Node {v.id}: parent: {v.parent} children: {v.children} accept_timer: {v.accept_timer} echo_count: {v.echo_count} state: {v.state}", flush=True)


def main():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)
    g.add_node(6)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.test_message(0, 25)
    g.debug_print()
    print(f"Message complexity: {g.message}")


if __name__=='__main__':
    main()