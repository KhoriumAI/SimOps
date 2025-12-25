"""
SimOps Geometry Module
======================

Geometry utilities including wind tunnel generation for external CFD.
"""

from .wind_tunnel import VirtualWindTunnel, WindTunnelConfig, create_wind_tunnel_mesh

__all__ = ['VirtualWindTunnel', 'WindTunnelConfig', 'create_wind_tunnel_mesh']
