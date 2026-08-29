"""
matcher.py – Feature matching module for multi-modal lunar image registration.

Provides three matching strategies designed for the challenges of
Chandrayaan-2 OHRC / TMC-2 ↔ LROC reference alignment:

1. **Traditional SIFT + FLANN** – fast baseline with Lowe's ratio test.
2. **Deep LoFTR matcher** – transformer-based dense matcher (via Kornia)
   robust to extreme illumination / texture-poor lunar terrain.
3. **Scale-invariant sliding-window** – tiles a high-res OHRC strip and
   matches each tile against a lower-res reference, aggregating inliers.

All public methods return a standardised :class:`MatchResult` dict with
``src_pts`` and ``ref_pts`` as *(N, 2)* NumPy float32 arrays.

Usage::

    matcher = LunarFeatureMatcher()
    result  = matcher.match_sift(ohrc_gray, lroc_gray)
    H, mask = cv2.findHomography(result["src_pts"],
                                 result["ref_pts"], cv2.RANSAC)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class MatchResult(TypedDict):
    """Standardised output from every matching method.

    Attributes
    ----------
    src_pts : npt.NDArray[np.float32]
        Matched keypoint coordinates in the *source* image, shape ``(N, 2)``.
    ref_pts : npt.NDArray[np.float32]
        Corresponding keypoint coordinates in the *reference* image,
        shape ``(N, 2)``.
    num_matches : int
        Number of accepted matches (``N``).
    method : str
        Label identifying the algorithm that produced the matches.
    """

    src_pts: npt.NDArray[np.float32]
    ref_pts: npt.NDArray[np.float32]
    num_matches: int
    method: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _to_gray_u8(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Ensure *image* is a single-channel ``uint8`` array.

    Handles float [0, 1], float [0, 255], uint16, and multi-band inputs.
    """
    if image.ndim == 3:
        # (bands, H, W) → (H, W) – take first band
        if image.shape[0] <= image.shape[2]:
            image = image[0]
        else:
            # (H, W, C) – standard BGR/RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            return (image * 255.0).clip(0, 255).astype(np.uint8)
        return image.clip(0, 255).astype(np.uint8)

    # uint16 / int32 / etc. – normalise to full 8-bit range
    lo, hi = float(image.min()), float(image.max())
    if hi - lo == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image.astype(np.float64) - lo) / (hi - lo) * 255).astype(
        np.uint8
    )


def _apply_clahe(
    image: npt.NDArray[np.uint8],
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> npt.NDArray[np.uint8]:
    """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Significantly improves SIFT feature yield on low-contrast lunar
    terrain with harsh sun-angle shadows.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def _empty_match_result(method: str) -> MatchResult:
    """Return a valid but empty :class:`MatchResult`."""
    return MatchResult(
        src_pts=np.empty((0, 2), dtype=np.float32),
        ref_pts=np.empty((0, 2), dtype=np.float32),
        num_matches=0,
        method=method,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LunarFeatureMatcher:
    """Multi-strategy feature matcher for lunar image registration.

    Parameters
    ----------
    sift_nfeatures : int
        Maximum features for SIFT (0 = unlimited).
    ratio_thresh : float
        Lowe's ratio threshold for the SIFT matcher (default 0.75).
    clahe_preprocess : bool
        Whether to apply CLAHE before traditional matching (recommended
        for sun-angle-variant data).
    loftr_pretrained : str
        Kornia LoFTR pretrained weight tag.  ``"outdoor"`` generalises
        best to planetary surfaces.
    loftr_confidence : float
        Minimum LoFTR confidence score for a match to be accepted.
    device : str | None
        PyTorch device string (``"cuda"``, ``"cpu"``).  ``None`` will
        auto-select CUDA when available.

    Examples
    --------
    >>> matcher = LunarFeatureMatcher(ratio_thresh=0.7)
    >>> result = matcher.match_sift(source_gray, reference_gray)
    >>> print(result["num_matches"])
    342
    """

    def __init__(
        self,
        sift_nfeatures: int = 0,
        ratio_thresh: float = 0.75,
        clahe_preprocess: bool = True,
        loftr_pretrained: str = "outdoor",
        loftr_confidence: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        # Traditional (SIFT) parameters
        self._sift_nfeatures = sift_nfeatures
        self._ratio_thresh = ratio_thresh
        self._clahe = clahe_preprocess

        # Deep-learning (LoFTR) parameters
        self._loftr_pretrained = loftr_pretrained
        self._loftr_confidence = loftr_confidence
        self._device_str = device

        # Lazily initialised heavy objects
        self._sift: Optional[cv2.SIFT] = None
        self._loftr_model: Any = None
        self._torch_device: Any = None

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _get_sift(self) -> cv2.SIFT:
        """Return a cached SIFT detector instance."""
        if self._sift is None:
            self._sift = cv2.SIFT_create(nfeatures=self._sift_nfeatures)
            logger.debug(
                "SIFT detector created (nfeatures=%d).",
                self._sift_nfeatures,
            )
        return self._sift

    def _get_torch_device(self) -> Any:
        """Resolve and cache the PyTorch device."""
        if self._torch_device is None:
            import torch

            if self._device_str is not None:
                self._torch_device = torch.device(self._device_str)
            else:
                self._torch_device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            logger.info("PyTorch device: %s", self._torch_device)
        return self._torch_device

    def _get_loftr(self) -> Any:
        """Return a cached LoFTR model (Kornia)."""
        if self._loftr_model is None:
            try:
                import torch
                from kornia.feature import LoFTR
            except ImportError as exc:
                raise ImportError(
                    "LoFTR matching requires 'kornia' and 'torch'. "
                    "Install them with:  pip install torch kornia"
                ) from exc

            device = self._get_torch_device()
            self._loftr_model = LoFTR(pretrained=self._loftr_pretrained)
            self._loftr_model = self._loftr_model.to(device).eval()
            logger.info(
                "LoFTR model loaded (pretrained=%s) on %s.",
                self._loftr_pretrained,
                device,
            )
        return self._loftr_model

    # ==================================================================
    # Task 1 – Traditional SIFT + FLANN matcher
    # ==================================================================

    def match_sift(
        self,
        source: npt.NDArray[Any],
        reference: npt.NDArray[Any],
        *,
        ratio_thresh: Optional[float] = None,
        cross_check: bool = False,
    ) -> MatchResult:
        """Detect SIFT keypoints and match via FLANN with Lowe's ratio test.

        Parameters
        ----------
        source : ndarray
            Source (OHRC / moving) image – any dtype, any band count.
        reference : ndarray
            Reference (TMC-2 / LROC) image.
        ratio_thresh : float, optional
            Override the instance-level Lowe ratio threshold.
        cross_check : bool
            If ``True``, perform a secondary cross-check: match
            ``reference → source`` and keep only mutual best matches.
            Increases precision at the cost of recall.

        Returns
        -------
        MatchResult
        """
        method_label = "sift_flann"
        thresh = ratio_thresh if ratio_thresh is not None else self._ratio_thresh

        # ---- Preprocessing --------------------------------------------------
        src_u8 = _to_gray_u8(source)
        ref_u8 = _to_gray_u8(reference)

        if self._clahe:
            src_u8 = _apply_clahe(src_u8)
            ref_u8 = _apply_clahe(ref_u8)

        # ---- Detect + Compute -----------------------------------------------
        sift = self._get_sift()
        kp_src, des_src = sift.detectAndCompute(src_u8, None)
        kp_ref, des_ref = sift.detectAndCompute(ref_u8, None)

        if des_src is None or des_ref is None:
            logger.warning("SIFT found no descriptors in one or both images.")
            return _empty_match_result(method_label)

        logger.debug(
            "SIFT keypoints – source: %d, reference: %d",
            len(kp_src),
            len(kp_ref),
        )

        # ---- FLANN matching --------------------------------------------------
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # knnMatch returns list[list[DMatch]] – k=2 for ratio test
        try:
            raw_matches = flann.knnMatch(des_src, des_ref, k=2)
        except cv2.error as exc:
            logger.error("FLANN knnMatch failed: %s", exc)
            return _empty_match_result(method_label)

        # ---- Lowe's ratio test -----------------------------------------------
        good: List[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < thresh * n.distance:
                    good.append(m)

        # ---- Optional cross-check -------------------------------------------
        if cross_check and len(good) > 0:
            try:
                reverse_matches = flann.knnMatch(des_ref, des_src, k=2)
            except cv2.error:
                reverse_matches = []

            reverse_set: set[Tuple[int, int]] = set()
            for pair in reverse_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < thresh * n.distance:
                        reverse_set.add((m.trainIdx, m.queryIdx))

            good = [
                m for m in good if (m.queryIdx, m.trainIdx) in reverse_set
            ]

        if len(good) == 0:
            logger.warning("SIFT matching produced 0 good matches.")
            return _empty_match_result(method_label)

        # ---- Build output arrays ---------------------------------------------
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good])
        ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in good])

        logger.info("SIFT matched %d keypoint pairs.", len(good))

        return MatchResult(
            src_pts=src_pts,
            ref_pts=ref_pts,
            num_matches=len(good),
            method=method_label,
        )

    # ==================================================================
    # Task 2 – Deep Learning LoFTR matcher (Kornia)
    # ==================================================================

    def match_loftr(
        self,
        source: npt.NDArray[Any],
        reference: npt.NDArray[Any],
        *,
        confidence: Optional[float] = None,
        resize_max: int = 1024,
    ) -> MatchResult:
        """Match features with LoFTR (transformer-based, Kornia).

        LoFTR is a detector-free matcher that directly produces dense
        correspondences.  It handles drastic illumination changes,
        texture-poor areas, and large viewpoint shifts far better than
        hand-crafted features.

        Parameters
        ----------
        source : ndarray
            Source image (any dtype / band count).
        reference : ndarray
            Reference image.
        confidence : float, optional
            Minimum confidence score to keep a correspondence.
            Overrides the instance-level ``loftr_confidence``.
        resize_max : int
            Longest-edge cap for the images fed into the network.
            LoFTR is memory-hungry; this avoids OOM on large strips.
            Matched coordinates are scaled back to original resolution.

        Returns
        -------
        MatchResult
        """
        import torch

        method_label = "loftr"
        min_conf = confidence if confidence is not None else self._loftr_confidence

        # ---- Preprocessing --------------------------------------------------
        src_u8 = _to_gray_u8(source)
        ref_u8 = _to_gray_u8(reference)

        src_resized, src_scale = self._resize_for_loftr(src_u8, resize_max)
        ref_resized, ref_scale = self._resize_for_loftr(ref_u8, resize_max)

        # Ensure dimensions are multiples of 8 (LoFTR's FPN requirement)
        src_resized = self._pad_to_multiple(src_resized, multiple=8)
        ref_resized = self._pad_to_multiple(ref_resized, multiple=8)

        device = self._get_torch_device()
        model = self._get_loftr()

        # Convert to tensors: (1, 1, H, W), float32, [0, 1]
        src_t = (
            torch.from_numpy(src_resized)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        ref_t = (
            torch.from_numpy(ref_resized)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )

        # ---- Inference -------------------------------------------------------
        with torch.no_grad():
            correspondences = model({"image0": src_t, "image1": ref_t})

        kpts0 = correspondences["keypoints0"].cpu().numpy()  # (N, 2)
        kpts1 = correspondences["keypoints1"].cpu().numpy()
        scores = correspondences["confidence"].cpu().numpy()  # (N,)

        # ---- Confidence filter -----------------------------------------------
        mask = scores >= min_conf
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]

        if kpts0.shape[0] == 0:
            logger.warning(
                "LoFTR returned 0 correspondences above confidence %.2f.",
                min_conf,
            )
            return _empty_match_result(method_label)

        # ---- Scale back to original resolution -------------------------------
        src_pts = kpts0.astype(np.float32) / np.array(
            [src_scale], dtype=np.float32
        )
        ref_pts = kpts1.astype(np.float32) / np.array(
            [ref_scale], dtype=np.float32
        )

        logger.info(
            "LoFTR matched %d correspondences (confidence >= %.2f).",
            src_pts.shape[0],
            min_conf,
        )

        return MatchResult(
            src_pts=src_pts,
            ref_pts=ref_pts,
            num_matches=int(src_pts.shape[0]),
            method=method_label,
        )

    # ------------------------------------------------------------------
    # LoFTR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_for_loftr(
        image: npt.NDArray[np.uint8],
        max_edge: int,
    ) -> Tuple[npt.NDArray[np.uint8], np.float32]:
        """Down-scale *image* so its longest edge ≤ *max_edge*.

        Returns the resized image **and** the scale factor applied
        (``resized_size / original_size``) so coordinates can be
        mapped back.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_edge:
            return image, np.float32(1.0)

        scale = max_edge / longest
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        return resized, np.float32(scale)

    @staticmethod
    def _pad_to_multiple(
        image: npt.NDArray[np.uint8],
        multiple: int = 8,
    ) -> npt.NDArray[np.uint8]:
        """Zero-pad *image* so both dimensions are multiples of *multiple*."""
        h, w = image.shape[:2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return image
        return cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

    # ==================================================================
    # Task 3 – Scale-Invariant Sliding-Window Tile Matcher
    # ==================================================================

    def match_tiled(
        self,
        source_highres: npt.NDArray[Any],
        reference_lowres: npt.NDArray[Any],
        *,
        tile_size: int = 512,
        overlap: int = 128,
        match_method: Literal["sift", "loftr"] = "sift",
        min_matches_per_tile: int = 8,
        scale_factors: Optional[Sequence[float]] = None,
    ) -> MatchResult:
        """Tile the high-res *source* and match each tile to the *reference*.

        This tackles the OHRC-to-TMC-2 / LROC problem where the GSD
        ratio can be 5–20×.  Each tile is optionally rescaled to several
        candidate scales and matched independently.  Valid matches from
        every tile are aggregated into a single result.

        Parameters
        ----------
        source_highres : ndarray
            High-resolution image (e.g. OHRC, 0.25 m/px).
        reference_lowres : ndarray
            Lower-resolution reference (e.g. TMC-2 at 5 m/px or
            LROC NAC at 0.5 m/px).
        tile_size : int
            Side length of each square tile (pixels in the *source*).
        overlap : int
            Overlap in pixels between adjacent tiles.  Ensures features
            near tile edges are not lost.
        match_method : ``"sift"`` | ``"loftr"``
            Which internal matcher to use for each tile.
        min_matches_per_tile : int
            Minimum accepted matches for a tile's result to be included
            in the aggregate.
        scale_factors : sequence of float, optional
            Down-scale factors to try for each tile.  For example,
            ``[0.05, 0.1, 0.2]`` would shrink each 512 px tile to 26,
            51, and 102 px before matching against the reference.
            If ``None``, a reasonable set is computed automatically from
            image size ratios.

        Returns
        -------
        MatchResult
            Aggregated ``src_pts`` (in *source* pixel coordinates) and
            ``ref_pts`` (in *reference* pixel coordinates).
        """
        method_label = f"tiled_{match_method}"

        src_u8 = _to_gray_u8(source_highres)
        ref_u8 = _to_gray_u8(reference_lowres)
        src_h, src_w = src_u8.shape[:2]
        ref_h, ref_w = ref_u8.shape[:2]

        # ---- Compute default scale factors if needed -------------------------
        if scale_factors is None:
            scale_factors = self._auto_scale_factors(
                src_h, src_w, ref_h, ref_w
            )
        logger.info(
            "Tiled matching: tile=%d, overlap=%d, scales=%s, method=%s",
            tile_size,
            overlap,
            [f"{s:.3f}" for s in scale_factors],
            match_method,
        )

        # ---- Generate tile positions ----------------------------------------
        tiles = self._generate_tiles(src_h, src_w, tile_size, overlap)
        logger.info("Generated %d tiles from source image.", len(tiles))

        # ---- Match each tile -------------------------------------------------
        all_src: List[npt.NDArray[np.float32]] = []
        all_ref: List[npt.NDArray[np.float32]] = []

        for idx, (y0, x0, y1, x1) in enumerate(tiles):
            tile_img = src_u8[y0:y1, x0:x1]
            best_result: Optional[MatchResult] = None

            for sf in scale_factors:
                new_h = max(int(round(tile_img.shape[0] * sf)), 16)
                new_w = max(int(round(tile_img.shape[1] * sf)), 16)
                tile_scaled = cv2.resize(
                    tile_img,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA,
                )

                if match_method == "loftr":
                    result = self.match_loftr(tile_scaled, ref_u8)
                else:
                    result = self.match_sift(tile_scaled, ref_u8)

                if result["num_matches"] >= min_matches_per_tile:
                    if (
                        best_result is None
                        or result["num_matches"] > best_result["num_matches"]
                    ):
                        # Map matched points back to full-source coords
                        mapped_src = result["src_pts"] / sf + np.array(
                            [[x0, y0]], dtype=np.float32
                        )
                        best_result = MatchResult(
                            src_pts=mapped_src,
                            ref_pts=result["ref_pts"],
                            num_matches=result["num_matches"],
                            method=method_label,
                        )

            if best_result is not None:
                all_src.append(best_result["src_pts"])
                all_ref.append(best_result["ref_pts"])
                logger.debug(
                    "Tile %d/%d (%d:%d, %d:%d): %d matches.",
                    idx + 1,
                    len(tiles),
                    y0,
                    y1,
                    x0,
                    x1,
                    best_result["num_matches"],
                )
            else:
                logger.debug(
                    "Tile %d/%d (%d:%d, %d:%d): no sufficient matches.",
                    idx + 1,
                    len(tiles),
                    y0,
                    y1,
                    x0,
                    x1,
                )

        # ---- Aggregate -------------------------------------------------------
        if len(all_src) == 0:
            logger.warning("Tiled matching produced 0 aggregate matches.")
            return _empty_match_result(method_label)

        src_pts = np.concatenate(all_src, axis=0)
        ref_pts = np.concatenate(all_ref, axis=0)

        logger.info(
            "Tiled matching aggregated %d correspondences from %d/%d tiles.",
            src_pts.shape[0],
            len(all_src),
            len(tiles),
        )

        return MatchResult(
            src_pts=src_pts,
            ref_pts=ref_pts,
            num_matches=int(src_pts.shape[0]),
            method=method_label,
        )

    # ------------------------------------------------------------------
    # Tiling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_tiles(
        img_h: int,
        img_w: int,
        tile_size: int,
        overlap: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Yield ``(y0, x0, y1, x1)`` bounding boxes covering the image.

        Tiles step by ``tile_size - overlap`` and are clamped to image
        bounds so that the last column / row of tiles is flush with the
        edge rather than overhanging.
        """
        step = max(tile_size - overlap, 1)
        tiles: List[Tuple[int, int, int, int]] = []
        for y0 in range(0, img_h, step):
            y1 = min(y0 + tile_size, img_h)
            for x0 in range(0, img_w, step):
                x1 = min(x0 + tile_size, img_w)
                tiles.append((y0, x0, y1, x1))
        return tiles

    @staticmethod
    def _auto_scale_factors(
        src_h: int,
        src_w: int,
        ref_h: int,
        ref_w: int,
    ) -> List[float]:
        """Heuristically derive down-scale factors from source / reference
        size ratios.

        Returns 3–4 scale candidates centred around the estimated GSD
        ratio so the sliding-window search can accommodate uncertainty.
        """
        ratio_h = ref_h / max(src_h, 1)
        ratio_w = ref_w / max(src_w, 1)
        centre = (ratio_h + ratio_w) / 2.0
        centre = max(min(centre, 1.0), 0.01)

        factors = sorted(
            {
                max(round(centre * 0.5, 4), 0.01),
                max(round(centre, 4), 0.01),
                max(round(centre * 1.5, 4), 0.01),
                max(round(centre * 2.0, 4), 0.01),
            }
        )
        return factors

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ratio_thresh={self._ratio_thresh}, "
            f"clahe={self._clahe}, "
            f"loftr_pretrained={self._loftr_pretrained!r})"
        )
