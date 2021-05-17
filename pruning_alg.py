#GP pruning 

class Prunner:
    def __init__(self, cov_mat, curiosity):
        self.cov_mat = cov_mat
        self.curiosity = curiosity
    
    def eval(self, point):
        r"Compute pruning score of a point, we do that on a <state-action-new_state> transition basis"


