#GP pruning 

class Prunner:
    def __init__(self, cov_mat):
        self.cov_mat = cov_mat
    
    def eval(self, point):
        r"Compute pruning score of a point"
        
