import numpy as np
#config class GP models

class configGP():
    dims_input        = None     #depends on the problem, have to be modified in main
    no_restarts       = 2
    eps               = 0.33
    valid_iter        = 10
    no_controls       = 2
    gamma             = 0.99
    rand_search_cand  = 10
    max_eval          = 2000
    pre_filling_iters = 20
    
    #training
    training_iter     = 5
    
    #alpha
    alp_begin         = 10
    alp_end           = 35
    iter_end          = 75  
    alp_up_anneal     = True 
    if alp_up_anneal == True:
        alp_rate          = training_iter / np.log(alp_end/alp_begin)
    else:
        alp_rate          = - iter_end / np.log(alp_end/alp_begin)

    
    #beta
    bet_begin         = 30  
    bet_end           = 1
    bet_rate          = - iter_end / np.log(bet_end/bet_begin)
    
    #increased penalty over constraint violation
    v_c               = 1    
    #1. constraint limit
    v1_q              = 420           
    save              = True
    #lenghtscale constraining
    ls_lb             = None
    ls_ub             = None

    #resolver un intervalo pequeño primero
    #quitarle la exploration
    #verificar la prediccion del GP
    #comparar con la integracino de la dinamica del sistema, debe estar dentro de la incertidumbre del GP
        #Ver las funciones de "violation"
    #
          

