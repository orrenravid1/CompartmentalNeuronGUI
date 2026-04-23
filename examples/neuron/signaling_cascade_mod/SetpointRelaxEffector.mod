TITLE Generic bounded setpoint relaxation effector (drive -> s in [0,1])

NEURON {
    POINT_PROCESS SetpointRelaxEffector
    POINTER drive

    RANGE s, effect, s_inf
    RANGE s_min, s_max
    RANGE K, n
    RANGE tau_on, tau_off
}

PARAMETER {
    s_min = 0 (1)
    s_max = 1 (1)

    K = 0.5 (1)
    n = 2

    tau_on = 5000 (ms)
    tau_off = 30000 (ms)
}

ASSIGNED {
    drive (1)
    effect (1)
    s_inf (1)
}

STATE {
    s (1)
}

INITIAL {
    s = s_min
    s_inf = s_min
    effect = s
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    effect = s
}

DERIVATIVE states {
    LOCAL d, dn, Kd, f, tau

    d = drive
    if (d < 0) { d = 0 }
    if (d > 1) { d = 1 }

    dn = d^n
    Kd = K^n
    f = dn / (dn + Kd)

    s_inf = s_min + (s_max - s_min) * f

    tau = tau_off + (tau_on - tau_off) * f

    s' = (s_inf - s) / tau
}
