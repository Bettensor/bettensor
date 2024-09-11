

class EntropySystem:
    '''
    The Entropy System is a component of the composite score, which measures the diversity of miner predictions. 
    The goal of the entropy system is to discourage copy trading and incentive farming.
    '''

    def init(self, e_weight=0.1, entropy_window=7):
        self.e_weight = e_weight
        self.entropy_window = entropy_window

   