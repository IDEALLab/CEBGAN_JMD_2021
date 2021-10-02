import torch

#######################################################################################################################
# Elementary operations .....................................................................
#######################################################################################################################

def lse(v_ij):
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1, 1)

def Sinkhorn_ops(ε, x_i, y_j, cost_func): 
    # We precompute the |x_i-y_j|^p matrix once and for all...
    C_e = cost_func(x_i, y_j) / ε

    # Before wrapping it up in a simple pair of operators - don't forget the minus!
    S_x = lambda f_i: -lse(f_i.view(1, -1) - C_e.T)
    S_y = lambda f_j: -lse(f_j.view(1, -1) - C_e)
    return S_x, S_y


#######################################################################################################################
# Sinkhorn iterations .....................................................................
#######################################################################################################################

def sink(α_i, x_i, β_j, y_j, cost_func, eps=.1, nits=100, tol=1e-3, assume_convergence=False, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    if type(nits) in [list, tuple]: nits = nits[0]  # The user may give different limits for Sink and SymSink
    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................
    
    α_i_log, β_j_log = α_i.log(), β_j.log() # Precompute the logs of the measures' weights
    B_i, A_j = torch.zeros_like(α_i), torch.zeros_like(β_j) # Sampled influence fields
    
    # if we assume convergence, we can skip all the "save computational history" stuff
    # torch.set_grad_enabled(not assume_convergence)
    with torch.set_grad_enabled(not assume_convergence):
        S_x, S_y = Sinkhorn_ops(ε, x_i, y_j, cost_func) # Softmin operators (divided by ε, as it's slightly cheaper...)
        for i in range(nits-1):
            B_i_prev = B_i

            A_j = S_x(B_i + α_i_log)   # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
            B_i = S_y(A_j + β_j_log)   # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε

            err = ε * (B_i - B_i_prev).abs().mean() # Stopping criterion: L1 norm of the updates
            if err.item() < tol: break

    # One last step, which allows us to bypass PyTorch's backprop engine if required (as explained in the paper)
    if not assume_convergence:
        A_j = S_x(B_i + α_i_log)
        B_i = S_y(A_j + β_j_log)
    else: # Assume that we have converged, and can thus use the "exact" (and cheap!) gradient's formula
        S_x, _ = Sinkhorn_ops(ε, [each.detach() for each in x_i], y_j, cost_func)
        _, S_y = Sinkhorn_ops(ε, x_i, [each.detach() for each in y_j], cost_func)
        # S_x, _ = Sinkhorn_ops(ε, x_i.detach(), y_j, cost_func)
        # _, S_y = Sinkhorn_ops(ε, x_i, y_j.detach(), cost_func)
        A_j = S_x((B_i + α_i_log).detach())
        B_i = S_y((A_j + β_j_log).detach())

    a_y, b_x = ε * A_j.view(-1), ε * B_i.view(-1)
    return a_y, b_x


def sym_sink(α_i, x_i, cost_func, eps=.1, nits=100, tol=1e-3, assume_convergence=False, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    if type(nits) in [list, tuple]: nits = nits[1]  # The user may give different limits for Sink and SymSink
    # Sinkhorn loop ......................................................................

    α_i_log = α_i.log()
    A_i = torch.zeros_like(α_i)
    S_x, _ = Sinkhorn_ops(ε, x_i, x_i, cost_func) # Sinkhorn operator from x_i to x_i (divided by ε, as it's slightly cheaper...)
    
    # if we assume convergence, we can skip all the "save computational history" stuff
    with torch.set_grad_enabled(not assume_convergence):
        for i in range(nits-1):
            A_i_prev = A_i

            A_i = 0.5 * (A_i + S_x(A_i + α_i_log) ) # a(x)/ε = .5*(a(x)/ε + Smin_ε,y~α [ C(x,y) - a(y) ] / ε)
            
            err = ε * (A_i - A_i_prev).abs().mean()    # Stopping criterion: L1 norm of the updates
            if err.item() < tol: break

    # One last step, which allows us to bypass PyTorch's backprop engine if required
    if not assume_convergence:
        W_i = A_i + α_i_log
    else:
        W_i = (A_i + α_i_log).detach()
        S_x, _ = Sinkhorn_ops(ε, [each.detach() for each in x_i], x_i, cost_func)
        # S_x, _ = Sinkhorn_ops(ε, x_i.detach(), x_i, cost_func)

    a_x = ε * S_x(W_i).view(-1) # a(x) = Smin_e,z~α [ C(x,z) - a(z) ]
    return a_x


#######################################################################################################################
# Derived Functionals .....................................................................
#######################################################################################################################

def regularized_ot(α, x, β, y, **params): # OT_ε
    a_y, b_x = sink(α, x, β, y, **params)
    return b_x @ α + a_y @ β

def sinkhorn_divergence(α, x, β, y, **params): # S_ε
    a_y, b_x = sink(α, x, β, y, **params)
    a_x = sym_sink(α, x, **params)
    b_y = sym_sink(β, y, **params)
    return (b_x - a_x) @ α + (a_y - b_y) @ β





