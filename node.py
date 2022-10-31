import math

class Node:
    def __init__(self, x, y, params):
        self.x = x
        self.y = y
        self.tx_power = 0
        self.params = params

    def set_coordination(self, x, y):
        self.x = x
        self.y = y

    def dist_node(self, node):
        return ((self.x - node.x)**2 + (self.y - node.y)**2 + 23.5**2)**0.5