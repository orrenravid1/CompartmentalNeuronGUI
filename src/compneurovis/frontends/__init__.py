from compneurovis.frontends.base import FrontendBase
from compneurovis.frontends.host import FrontendHost


def __getattr__(name: str):
    if name in ("VispyFrontendHost", "VispyFrontendWindow"):
        from compneurovis.frontends.vispy import VispyFrontendHost, VispyFrontendWindow
        g = globals()
        g["VispyFrontendHost"] = VispyFrontendHost
        g["VispyFrontendWindow"] = VispyFrontendWindow
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FrontendBase", "FrontendHost", "VispyFrontendHost", "VispyFrontendWindow"]
