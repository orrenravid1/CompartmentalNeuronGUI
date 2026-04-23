TITLE General Receptor with Per-Ligand Decay and Flexible Targeting

NEURON {
    POINT_PROCESS GenericReceptor
    RANGE baseline_activity
    RANGE activation, capacity, occupancy
    RANGE n_ligands

    RANGE kd1, efficacy1, bound1, decay1
    RANGE kd2, efficacy2, bound2, decay2
    RANGE kd3, efficacy3, bound3, decay3
    RANGE kd4, efficacy4, bound4, decay4

    POINTER C_lig1, C_lig2, C_lig3, C_lig4
}

PARAMETER {
    baseline_activity = 0

    n_ligands = 1

    capacity = 1.0

    kd1 = 1.0 (uM)
    efficacy1 = 1.0
    decay1 = 0.1 (/ms)

    kd2 = 1.0 (uM)
    efficacy2 = 1.0
    decay2 = 0.1 (/ms)

    kd3 = 1.0 (uM)
    efficacy3 = 1.0
    decay3 = 0.1 (/ms)

    kd4 = 1.0 (uM)
    efficacy4 = 1.0
    decay4 = 0.1 (/ms)
}

ASSIGNED {
    C_lig1 (uM)
    C_lig2 (uM)
    C_lig3 (uM)
    C_lig4 (uM)
    activation (1)
    occupancy (1)
}

STATE {
    bound1 (1)
    bound2 (1)
    bound3 (1)
    bound4 (1)
}

INITIAL {
    bound1 = 0
    bound2 = 0
    bound3 = 0
    bound4 = 0
    activation = 0
    occupancy = 0
}

BREAKPOINT {
    LOCAL net_activation
    SOLVE states METHOD cnexp
    occupancy = bound1 + bound2 + bound3 + bound4
    net_activation = bound1 * efficacy1 + bound2 * efficacy2 + bound3 * efficacy3 + bound4 * efficacy4
    activation = min(1, max(0, baseline_activity + net_activation))
}

DERIVATIVE states {
    LOCAL occ1, occ2, occ3, occ4, total_demand, remaining_capacity, scale_factor

    if (n_ligands >= 1) {
        occ1 = occ(C_lig1, kd1)
    } else {
        occ1 = 0
    }
    if (n_ligands >= 2) {
        occ2 = occ(C_lig2, kd2)
    } else {
        occ2 = 0
    }
    if (n_ligands >= 3) {
        occ3 = occ(C_lig3, kd3)
    } else {
        occ3 = 0
    }
    if (n_ligands >= 4) {
        occ4 = occ(C_lig4, kd4)
    } else {
        occ4 = 0
    }

    total_demand = occ1 + occ2 + occ3 + occ4

    remaining_capacity = capacity - occupancy

    if (total_demand > remaining_capacity && total_demand > 0) {
        scale_factor = remaining_capacity / total_demand
    } else {
        scale_factor = 1.0
    }

    if (n_ligands >= 1) {
        bound1' = scale_factor * occ1 - bound1 * decay1
    }
    if (n_ligands >= 2) {
        bound2' = scale_factor * occ2 - bound2 * decay2
    }
    if (n_ligands >= 3) {
        bound3' = scale_factor * occ3 - bound3 * decay3
    }
    if (n_ligands >= 4) {
        bound4' = scale_factor * occ4 - bound4 * decay4
    }
}

FUNCTION occ(C_lig(uM), kd(uM)) {
    if (C_lig + kd <= 0) {
        occ = 0
    } else {
        occ = C_lig / (kd + C_lig)
    }
}

FUNCTION min(x1, x2) {
    if (x1 <= x2) {
        min = x1
    } else {
        min = x2
    }
}

FUNCTION max(x1, x2) {
    if (x1 >= x2) {
        max = x1
    } else {
        max = x2
    }
}
