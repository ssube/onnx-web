# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)


block_cipher = None

dynamic_packages = [
    "filelock",
    "huggingface-hub",
    "numpy",
    "onnxruntime",
    "onnxruntime-directml",
    "packaging",
    "pyyaml",
    "regex",
    "requests",
    "tokenizers",
    "tqdm",
    "omegaconf",
    "win10toast",
]
metadatas = [copy_metadata(pkg) for pkg in dynamic_packages]
metadatas = sum(metadatas, [])

datas = [
    collect_data_files("basicsr", include_py_files=True, includes=[
        "archs/**",
        "data/**",
        "losses/**",
        "models/**",
        "utils/**",
    ]),
    collect_data_files("realesrgan", include_py_files=True, includes=[
        "archs/**",
        "data/**",
        "losses/**",
        "models/**",
        "utils/**",
    ]),
    collect_data_files("onnxruntime", include_py_files=True, includes=[
        "transformers/**",
        "tools/**",
    ]),
    collect_data_files("transformers", include_py_files=True, includes=[
        "**",
    ]),
]
datas = sum(datas, [])

a = Analysis(
    ['../api/entry.py'],
    pathex=[],
    binaries=[],
    datas=[
        *metadatas,
        *datas,
    ],
    hiddenimports=[
        "coloredlogs",
        "omegaconf",
        "onnxruntime",
        "onnxruntime-directml",
        "pytorch_lightning",
        "tqdm",
        "win10toast",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='onnx-web',
    debug=False,
    exclude_binaries=True,
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

dir = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server',
)