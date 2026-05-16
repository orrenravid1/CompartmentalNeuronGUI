def __getattr__(name: str):
    if name in ("VispyFrontendHost", "VispyFrontendWindow"):
        from compneurovis.frontends.vispy.frontend import VispyFrontendWindow
        from compneurovis.frontends.vispy.host import VispyFrontendHost
        g = globals()
        g["VispyFrontendHost"] = VispyFrontendHost
        g["VispyFrontendWindow"] = VispyFrontendWindow
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VispyFrontendHost", "VispyFrontendWindow"]
