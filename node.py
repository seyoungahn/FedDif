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
        # return ((self.x - node.x) ** 2 + (self.y - node.y) ** 2) ** 0.5

    def SNR_user(self, node):
        # TR 37.885 sidelink
        pathloss = 38.77 + 16.7 * math.log10(self.dist_node(node)) + 18.2 * math.log10(self.params.s_fc)
        noise = self.params.s_noise_power + 10.0 * math.log10(self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB)
        SNR_dB = node.tx_power - pathloss - noise
        return SNR_dB

    def spectral_efficiency_user(self, node):
        # TR 37.885 sidelink
        SNR_dB = self.SNR_user(node)
        SNR = 10.0**(SNR_dB / 10.0)
        spectral_efficiency = math.log2(1.0 + SNR)
        return spectral_efficiency

    ## TODO: 수형 추가함
    def datarate(self, node):
        result = self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB * self.spectral_efficiency_user(node)
        return result

    ## TODO: 수형 추가함
    def calc_num_RB(self, node):
        n_RB = math.ceil(self.params.s_model_size / self.datarate(node))
        return n_RB