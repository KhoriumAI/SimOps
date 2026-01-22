# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['simops-backend\\api_server.py'],
    pathex=[],
    binaries=[],
    datas=[('simops_pipeline.py', '.'), ('core', 'core'), ('tools', 'tools'), ('simops', 'simops'), ('simops-backend/simops_output', 'simops_output')],
    hiddenimports=['gmsh', 'engineio.async_drivers.threading', 'flask_cors'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='api_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
