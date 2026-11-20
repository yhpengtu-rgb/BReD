#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7.6 (I): whole-scene image pre-cache, bit-exact.

Why: the locked Stage1 eval loop is DataLoader-bound.  Per frame it runs, for
12 image slots (only 6 are unique -- the sweep axis re-reads the key frame
because `cam_sweeps` is empty in scene0523_allframe_infos.pkl):

    Image.open(jpg)                  ~8 ms  (lazy JPEG decode, 900x1600)
    img_transform_local (PIL resize) ~30 ms (bicubic -> 1024x576)
    mmcv.imnormalize                 ~8 ms  (float32 HWC, to_rgb=True)
    -> ~46 ms/image x 12 = ~550 ms CPU per frame (~190 ms/frame with 4 workers)

All three steps are deterministic (zero crop / flip / rotate / colour jitter),
so the *final float32 HWC tensor* can be rendered once per unique image and
replayed bit-exactly.  The scene has only 1388 unique images -> 9.15 GB.

Build (once):
    python i76_img_precache.py --build

Use (opt-in, default off):
    V7_IMG_PRECACHE=1      replace the per-image PIL work with a memcpy
    V7_IP_VERIFY=1         also re-run the real pipeline on 3 images at install

Nothing in the protocol changes: same dataset class, same return layout, same
pixel values (byte-identical), same frame order.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

BRED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
INFO_PKL = os.path.join(BRED, "assets", "metadata", "scene0523_allframe_infos.pkl")
DATA_ROOT = os.path.join(BRED, "assets", "datasets", "nuScenes")
CACHE_DIR = os.path.join(BRED, "assets", "cache", "v7_img_f32")
META = os.path.join(CACHE_DIR, "meta.json")
DATA = os.path.join(CACHE_DIR, "imgs_f32.dat")
U8_DIR = os.path.join(BRED, "assets", "cache", "v8_img_u8")

FINAL_H, FINAL_W = 576, 1024
RAW_H, RAW_W = 900, 1600
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
TO_RGB = True


# ---------------------------------------------------------------- build ----
def _render(rel_path, img_transform, mmcv, Image):
    img = Image.open(os.path.join(DATA_ROOT, rel_path))
    if (img.size[1], img.size[0]) != (RAW_H, RAW_W):
        raise RuntimeError("unexpected raw size %s for %s" % (img.size, rel_path))
    img = img_transform(img, 0.64, (FINAL_W, FINAL_H), (0, 0, FINAL_W, FINAL_H),
                        False, 0.0)[0]
    return np.ascontiguousarray(
        mmcv.imnormalize(np.array(img), MEAN, STD, TO_RGB))


def _unique_rels(info_pkl=INFO_PKL):
    infos = pickle.load(open(info_pkl, "rb"))
    rels, seen = [], set()
    for it in infos:
        groups = [it["cam_infos"]] + list(it.get("cam_sweeps") or [])
        for g in groups:
            for cam in sorted(g):
                fn = g[cam]["filename"]
                if fn not in seen:
                    seen.add(fn)
                    rels.append(fn)
    return rels


def build(info_pkl=INFO_PKL, out_dir=CACHE_DIR):
    sys.path.insert(0, os.path.join(BRED, "vendor", "BEVDepth"))
    import mmcv
    from PIL import Image
    from bevdepth.datasets.nusc_det_dataset import img_transform

    rels = _unique_rels(info_pkl)
    names = [os.path.basename(r) for r in rels]
    if len(set(names)) != len(names):
        raise RuntimeError("basename collision across %d images" % len(names))
    n = len(names)
    os.makedirs(out_dir, exist_ok=True)
    dat = os.path.join(out_dir, "imgs_f32.dat")
    mm = np.memmap(dat, dtype=np.float32, mode="w+",
                   shape=(n, FINAL_H, FINAL_W, 3))
    import time
    t0 = time.time()
    for i, rel in enumerate(rels):
        mm[i] = _render(rel, img_transform, mmcv, Image)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print("  %4d/%d  %.0fs  (%.0f ms/img)" % (i + 1, n, el, el / (i + 1) * 1e3),
                  flush=True)
    mm.flush()
    del mm
    json.dump({"names": names, "shape": [n, FINAL_H, FINAL_W, 3],
               "dtype": "float32", "mean": MEAN.tolist(), "std": STD.tolist(),
               "to_rgb": TO_RGB, "raw_size": [RAW_W, RAW_H], "info_pkl": info_pkl},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("built %d images -> %s (%.2f GB)" % (n, dat, os.path.getsize(dat) / 2**30))


def _render_u8(rel_path, img_transform, Image):
    img = Image.open(os.path.join(DATA_ROOT, rel_path))
    img = img_transform(img, 0.64, (FINAL_W, FINAL_H), (0, 0, FINAL_W, FINAL_H),
                        False, 0.0)[0]
    return np.ascontiguousarray(np.array(img))          # uint8 HWC, pre-normalise


def build_u8(info_pkl=INFO_PKL, out_dir=U8_DIR):
    """v8.3: store the resized uint8 frame; normalisation moves to the GPU."""
    sys.path.insert(0, os.path.join(BRED, "vendor", "BEVDepth"))
    from PIL import Image
    from bevdepth.datasets.nusc_det_dataset import img_transform

    rels = _unique_rels(info_pkl)
    names = [os.path.basename(r) for r in rels]
    if len(set(names)) != len(names):
        raise RuntimeError("basename collision across %d images" % len(names))
    n = len(names)
    os.makedirs(out_dir, exist_ok=True)
    dat = os.path.join(out_dir, "imgs_u8.dat")
    mm = np.memmap(dat, dtype=np.uint8, mode="w+",
                   shape=(n, FINAL_H, FINAL_W, 3))
    import time
    t0 = time.time()
    for i, rel in enumerate(rels):
        mm[i] = _render_u8(rel, img_transform, Image)
        if (i + 1) % 200 == 0:
            print("  %4d/%d  %.0fs" % (i + 1, n, time.time() - t0), flush=True)
    mm.flush()
    del mm
    json.dump({"names": names, "shape": [n, FINAL_H, FINAL_W, 3],
               "dtype": "uint8", "mean": MEAN.tolist(), "std": STD.tolist(),
               "to_rgb": TO_RGB, "raw_size": [RAW_W, RAW_H], "info_pkl": info_pkl},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("built %d uint8 images -> %s (%.2f GB)" % (n, dat, os.path.getsize(dat) / 2**30))


# -------------------------------------------------------------- install ----
class _CachedImg(object):
    """Duck-typed stand-in for a PIL image already reduced to the final tensor.

    Only the methods `img_transform` touches are implemented; `__array__`
    hands back the pre-normalised float32 HWC array, and the mmcv shim below
    passes that through untouched.
    """

    __slots__ = ("_row",)

    def __init__(self, row):
        self._row = row

    def resize(self, *a, **k):
        return self

    def crop(self, *a, **k):
        return self

    def rotate(self, *a, **k):
        return self

    def transpose(self, *a, **k):
        return self

    @property
    def size(self):
        return (RAW_W, RAW_H)

    def __array__(self, dtype=None):
        a = self._row
        return a if dtype is None else a.astype(dtype)


class _ImageShim(object):
    """Replacement for the module-level `Image` import."""

    def __init__(self, real, rows, strict=False):
        self._real = real
        self._rows = rows
        self._strict = strict

    def open(self, path, *a, **k):
        row = self._rows.get(os.path.basename(path))
        if row is None:
            if self._strict:
                raise KeyError("v8.3 uint8 cache miss: %s (cache must be complete "
                               "for the uint8 path, no silent fallback)" % path)
            return self._real.open(path, *a, **k)
        return _CachedImg(row)

    def __getattr__(self, k):
        return getattr(self._real, k)


class _MmcvShim(object):
    """Replacement for the module-level `mmcv` import (only imnormalize)."""

    def __init__(self, real, passthrough_dtype=np.float32):
        self._real = real
        self._dtype = passthrough_dtype

    def imnormalize(self, img, mean, std, to_rgb=True):
        if (isinstance(img, np.ndarray) and img.dtype == self._dtype and to_rgb
                and np.array_equal(np.asarray(mean, dtype=np.float32), MEAN)
                and np.array_equal(np.asarray(std, dtype=np.float32), STD)):
            return img
        return self._real.imnormalize(img, mean, std, to_rgb)

    def __getattr__(self, k):
        return getattr(self._real, k)


def install(cache_dir=None, verify=False, verbose=True, mode="f32"):
    import mmcv
    import nusc_det_dataset_road as dsmod

    u8 = str(mode).lower() in ("u8", "uint8")
    cache_dir = cache_dir or (U8_DIR if u8 else CACHE_DIR)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    names = meta["names"]
    dtype = np.uint8 if u8 else np.float32
    mm = np.memmap(os.path.join(cache_dir, os.path.basename(
        "imgs_u8.dat" if u8 else "imgs_f32.dat")), dtype=dtype,
        mode="r+", shape=tuple(meta["shape"]))
    rows = {nm: mm[i] for i, nm in enumerate(names)}
    dsmod.Image = _ImageShim(dsmod.Image, rows, strict=u8)
    dsmod.mmcv = _MmcvShim(mmcv, passthrough_dtype=dtype)

    if verify:
        from PIL import Image as _PIL
        from bevdepth.datasets.nusc_det_dataset import img_transform
        infos = pickle.load(open(meta["info_pkl"], "rb"))
        rels = [v["filename"] for v in infos[0]["cam_infos"].values()][:3]
        for rel in rels:
            ref = (_render_u8(rel, img_transform, _PIL) if u8 else
                   _render(rel, img_transform, mmcv, _PIL))
            got = np.asarray(mm[names.index(os.path.basename(rel))])
            print("V7_IP verify %-46s exact=%s maxabs=%.3e" % (
                os.path.basename(rel)[-34:], np.array_equal(ref, got),
                float(np.abs(ref.astype(np.float32) - got.astype(np.float32)).max())),
                flush=True)

    if verbose:
        n = int(meta["shape"][0])
        print("V7_IMG_PRECACHE installed: %d images, %.2f GB %s memmap, "
              "identity image transform (%s)" % (
                  n, os.path.getsize(os.path.join(cache_dir, os.path.basename(
                      "imgs_u8.dat" if u8 else "imgs_f32.dat"))) / 2**30,
                  "uint8" if u8 else "float32", "v8.3" if u8 else "v7.6"),
              flush=True)
    return mm


def install_gpu_norm(verbose=True):
    """v8.3: normalise the uint8 frame on the GPU (bit-exact vs mmcv.imnormalize).

    Must be installed AFTER c73_cuda_graph so that the graph's static float32
    input buffer never sees uint8.
    """
    import torch
    import bevstereo_rogs_model

    m = torch.tensor(MEAN, dtype=torch.float32, device="cuda").view(1, 1, 3, 1, 1)
    s = torch.tensor(STD, dtype=torch.float32, device="cuda").view(1, 1, 3, 1, 1)
    prev = bevstereo_rogs_model.BEVStereoRoGSModel.forward
    state = {"n": 0}

    def forward(self, imgs, mats_dict, *a, **k):
        if torch.is_tensor(imgs) and imgs.dtype == torch.uint8:
            # channel order: mmcv.imnormalize(to_rgb=True) swaps BGR<->RGB
            u = imgs.flip(-3).float()
            imgs = (u - m) / s
            # the dataset derives bda_mat from sweep_imgs.new_zeros(), which
            # inherits the uint8 image dtype -> restore the float contract.
            mats_dict = {n: (v.float() if torch.is_tensor(v) and
                             v.dtype == torch.uint8 else v)
                         for n, v in mats_dict.items()}
            state["n"] += 1
        return prev(self, imgs, mats_dict, *a, **k)

    bevstereo_rogs_model.BEVStereoRoGSModel.forward = forward
    if verbose:
        print("V8.3 gpu normalise installed (uint8 H2D + exact fp32 afffine)", flush=True)
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-u8", action="store_true")
    ap.add_argument("--info-pkl", default=INFO_PKL)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.build:
        build(a.info_pkl, a.out_dir or CACHE_DIR)
    elif a.build_u8:
        build_u8(a.info_pkl, a.out_dir or U8_DIR)
    else:
        install(a.out_dir)
