def load_swc_model(swc_path):
    """Import an SWC file into NEURON and return the list of sections."""
    from neuron import h
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    reader = h.Import3d_SWC_read()
    reader.input(swc_path)
    i3d = h.Import3d_GUI(reader, False)
    i3d.instantiate(None)
    secs = list(h.allsec())
    for sec in secs:
        if not "soma" in sec.name():
            sec.nseg = 5
        sec.insert('hh')
    return secs