#config class GP models

class configGP():
    dims_input = None     #depends on the problem, have to be modified in main
    no_restarts = 3
    eps = 0.33
    training_iter = 100
    valid_iter = 10
    no_controls = 2
    gamma = 0.99
    rand_search_cand = 10
    pre_filling_iters = 2 
    max_eval = 4000
          

