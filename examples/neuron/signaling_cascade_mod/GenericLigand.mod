TITLE Modular Ligand Class Supporting External Input

NEURON {
    POINT_PROCESS GenericLigand
    RANGE C, C_init, decay_rate, external_input
}

PARAMETER {
    C_init = 0 (uM)
    decay_rate = 0.01 (/ms)
    external_input = 0 (uM/ms)
}

STATE {
    C (uM)
}

INITIAL {
    C = C_init
}

BREAKPOINT {
    SOLVE state METHOD cnexp
}

DERIVATIVE state {
    C' = -C * decay_rate + external_input
}

NET_RECEIVE(weight (uM)) {
    C = C + weight
}
