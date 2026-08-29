"""
preprocessor.py – Spatial preprocessing for lunar image registration.

Provides two core capabilities needed to bridge the gap between
Chandrayaan-2 OHRC (∼0.25 m/px) and lower-resolution reference imagery
(TMC-2 ∼5 m/px, IIRS, LROC NAC/WAC):

1. **Metadata extraction & scale pyramiding** (Task 3)
   Extract geospatial bounds, CRS, transform, and GSD from loaded raster
   metadata.  Build Gaussian image pyramids at configurable down-sample
   factors so that multi-scale matching can operate on commensurate
   resolutions.

2. **Illumination & shadow normalisation** (Task 4)
   Compensate for extreme sun-angle variations and the harsh
   light/shadow boundaries typical of the lunar surface using CLAHE,
   percentile-clipping, and optional Wallis filtering.

All public methods accept and return plain ``numpy`` arrays alongside
:class:`SpatialMetadata` containers from the data-loader module so the
full spatial context is preserved through the pipeline.

Usage::

    from src.data_loader import LunarDataLoader
    from src.preprocessor import LunarImagePreprocessor

    loader = LunarDataLoader()
    data   = loader.load_image("ohrc_strip.tif")

    pre = LunarImagePreprocessor()
    pyramid = pre.build_pyramid(data.image, data.metadata, factors=[2, 4, 8])
    normed  = pre.normalize_illumination(data.image, data.metadata)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
import numpy.typing as npt

# Re-use the canonical metadata container from the loader module so that
# the whole pipeline speaks the same "type language".
from src.data_loader import LunarImageData, SpatialMetadata

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PyramidLevel:
    """A single level of a multi-resolution image pyramid.

    Attributes
    ----------
    image : numpy.ndarray
        Down-sampled image array, shape ``(H, W)`` or ``(bands, H, W)``.
    metadata : SpatialMetadata
        Spatial metadata adjusted for this resolution level (GSD, transform,
        width/height all reflect the down-sampled geometry).
    factor : int
        Down-sample factor relative to the original image (e.g. 4 means
        each pixel at this level covers a 4×4 block in the original).
    """

    image: npt.NDArray[Any]
    metadata: SpatialMetadata
    factor: int


@dataclass(frozen=True)
class ProcessedImage:
    """Container returned by the illumination normalisation pipeline.

    Attributes
    ----------
    image : numpy.ndarray
        Normalised image array.
    metadata : SpatialMetadata
        Spatial metadata (unchanged from input – normalisation does not
        alter geometry).
    applied_ops : list[str]
        Ordered list of processing steps that were applied (useful for
        provenance logging).
    """

    image: npt.NDArray[Any]
    metadata: SpatialMetadata
    applied_ops: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RasterBounds:
    """Axis-aligned bounding box in map coordinates.

    Attributes
    ----------
    left, bottom, right, top : float
        Bounding coordinates derived from the affine transform and
        image dimensions.
    crs : str | None
        CRS of the bounds (propagated from metadata).
    """

    left: float
    bottom: float
    right: float
    top: float
    crs: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _to_float32(image: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Convert *image* to float32 in [0, 1] range."""
    if image.dtype == np.float32 and image.max() <= 1.0:
        return image
    if np.issubdtype(image.dtype, np.floating):
        mx = image.max()
        if mx > 0:
            return (image / mx).astype(np.float32)
        return image.astype(np.float32)
    info = np.iinfo(image.dtype)
    return ((image.astype(np.float64) - info.min) / (info.max - info.min)).astype(
        np.float32
    )


def _to_gray_u8(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Reduce to single-channel uint8 for algorithms that require it."""
    if image.ndim == 3:
        # (bands, H, W) convention from rasterio
        if image.shape[0] <= image.shape[2]:
            image = image[0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            return (image * 255.0).clip(0, 255).astype(np.uint8)
        return image.clip(0, 255).astype(np.uint8)
    lo, hi = float(image.min()), float(image.max())
    if hi - lo == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image.astype(np.float64) - lo) / (hi - lo) * 255).astype(np.uint8)


def _scale_transform(
    transform: Optional[Tuple[float, ...]],
    factor: int,
) -> Optional[Tuple[float, ...]]:
    """Scale an ``(a, b, c, d, e, f)`` affine transform by *factor*.

    Pixel size is multiplied by *factor* while the origin (c, f) stays
    at the same map coordinate.
    """
    if transform is None:
        return None
    a, b, c, d, e, f = transform
    return (a * factor, b, c, d, e * factor, f)


def _scale_gsd(
    gsd: Optional[Tuple[float, float]],
    factor: int,
) -> Optional[Tuple[float, float]]:
    """Return GSD scaled by *factor*."""
    if gsd is None:
        return None
    return (gsd[0] * factor, gsd[1] * factor)


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════


class LunarImagePreprocessor:
    """Geospatial preprocessing for multi-modal lunar image registration.

    Parameters
    ----------
    clahe_clip_limit : float
        Default CLAHE clip-limit (controls contrast amplification).
        Higher values reveal more shadow detail but risk amplifying
        noise; 3.0–4.0 is a good start for lunar imagery.
    clahe_tile_grid : tuple[int, int]
        CLAHE tile grid size.  ``(8, 8)`` works for most strip widths;
        wider tiles smooth out more global illumination gradients.
    percentile_low : float
        Lower percentile for min-max percentile clipping (default 2 %).
    percentile_high : float
        Upper percentile for clipping (default 98 %).

    Examples
    --------
    >>> pre = LunarImagePreprocessor()
    >>> bounds = pre.extract_bounds(data.metadata)
    >>> pyramid = pre.build_pyramid(data.image, data.metadata)
    >>> result = pre.normalize_illumination(data.image, data.metadata)
    """

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> None:
        self._clip_limit = clahe_clip_limit
        self._tile_grid = clahe_tile_grid
        self._pct_lo = percentile_low
        self._pct_hi = percentile_high

    # ══════════════════════════════════════════════════════════════════
    # Task 3 – Metadata extraction & scale pyramiding
    # ══════════════════════════════════════════════════════════════════

    # ------------------------------------------------------------------
    # 3-A  Metadata extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_bounds(metadata: SpatialMetadata) -> RasterBounds:
        """Compute the axis-aligned bounding box from *metadata*.

        Uses the affine transform and image dimensions to derive the
        four corner coordinates.

        Parameters
        ----------
        metadata : SpatialMetadata

        Returns
        -------
        RasterBounds

        Raises
        ------
        ValueError
            If the metadata has no transform from which to derive bounds.
        """
        if metadata.transform is None:
            raise ValueError(
                "Cannot compute bounds: metadata has no affine transform."
            )

        a, b, c, d, e, f = metadata.transform
        w, h = metadata.width, metadata.height

        # Four corners in map coordinates
        xs = [
            c,
            c + a * w,
            c + b * h,
            c + a * w + b * h,
        ]
        ys = [
            f,
            f + d * w,
            f + e * h,
            f + d * w + e * h,
        ]

        bounds = RasterBounds(
            left=min(xs),
            bottom=min(ys),
            right=max(xs),
            top=max(ys),
            crs=metadata.crs,
        )

        logger.debug(
            "Extracted bounds: left=%.6f, bottom=%.6f, right=%.6f, top=%.6f",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
        return bounds

    @staticmethod
    def extract_spatial_summary(metadata: SpatialMetadata) -> Dict[str, Any]:
        """Return a flat dictionary summarising key spatial parameters.

        Handy for quick inspection / logging without unpacking the full
        :class:`SpatialMetadata` dataclass.

        Returns
        -------
        dict
            Keys: ``crs``, ``transform``, ``gsd_x``, ``gsd_y``,
            ``width``, ``height``, ``band_count``, ``dtype``.
        """
        gsd_x = metadata.gsd[0] if metadata.gsd else None
        gsd_y = metadata.gsd[1] if metadata.gsd else None
        return {
            "crs": metadata.crs,
            "transform": metadata.transform,
            "gsd_x": gsd_x,
            "gsd_y": gsd_y,
            "width": metadata.width,
            "height": metadata.height,
            "band_count": metadata.band_count,
            "dtype": metadata.dtype,
        }

    @staticmethod
    def compute_gsd_ratio(
        high_res: SpatialMetadata,
        low_res: SpatialMetadata,
    ) -> float:
        """Estimate the GSD ratio between two images.

        Returns ``gsd_low / gsd_high`` (e.g. 20.0 for 5 m vs 0.25 m).
        Falls back to pixel-area ratio if GSD is unavailable.
        """
        if high_res.gsd and low_res.gsd:
            ratio_x = low_res.gsd[0] / max(high_res.gsd[0], 1e-12)
            ratio_y = low_res.gsd[1] / max(high_res.gsd[1], 1e-12)
            return (ratio_x + ratio_y) / 2.0

        # Fallback: ratio of total pixel areas
        hr_pixels = max(high_res.width * high_res.height, 1)
        lr_pixels = max(low_res.width * low_res.height, 1)
        return math.sqrt(hr_pixels / lr_pixels)

    # ------------------------------------------------------------------
    # 3-B  Gaussian image pyramid
    # ------------------------------------------------------------------

    def build_pyramid(
        self,
        image: npt.NDArray[Any],
        metadata: SpatialMetadata,
        *,
        factors: Optional[Sequence[int]] = None,
        interpolation: int = cv2.INTER_AREA,
        antialias: bool = True,
    ) -> List[PyramidLevel]:
        """Generate a multi-resolution Gaussian image pyramid.

        Parameters
        ----------
        image : ndarray
            Input image (``(H, W)`` or ``(bands, H, W)``).
        metadata : SpatialMetadata
            Spatial metadata of the original image.
        factors : sequence of int, optional
            Down-sample factors.  Defaults to ``[1, 2, 4, 8]``.
            Factor 1 returns the original image as level 0.
        interpolation : int
            OpenCV interpolation flag.  ``INTER_AREA`` is best for
            down-sampling; ``INTER_CUBIC`` for up-sampling.
        antialias : bool
            If ``True``, apply a Gaussian blur before each down-sample
            step to suppress aliasing (important for texture-rich
            crater rims and regolith patterns).

        Returns
        -------
        list[PyramidLevel]
            One entry per factor, sorted from coarsest to finest.
        """
        if factors is None:
            factors = [1, 2, 4, 8]

        factors_sorted = sorted(set(factors))
        is_multiband = image.ndim == 3

        levels: List[PyramidLevel] = []

        for fac in factors_sorted:
            if fac == 1:
                # Original resolution
                level_meta = replace(metadata)
                levels.append(
                    PyramidLevel(image=image.copy(), metadata=level_meta, factor=1)
                )
                continue

            if is_multiband:
                # (bands, H, W) – resize each band independently
                bands_out: List[npt.NDArray[Any]] = []
                for b in range(image.shape[0]):
                    bands_out.append(
                        self._downsample_2d(
                            image[b], fac, interpolation, antialias
                        )
                    )
                level_img: npt.NDArray[Any] = np.stack(bands_out, axis=0)
            else:
                level_img = self._downsample_2d(
                    image, fac, interpolation, antialias
                )

            new_h, new_w = (
                level_img.shape[-2],
                level_img.shape[-1],
            )

            level_meta = SpatialMetadata(
                crs=metadata.crs,
                transform=_scale_transform(metadata.transform, fac),
                gsd=_scale_gsd(metadata.gsd, fac),
                width=new_w,
                height=new_h,
                band_count=metadata.band_count,
                dtype=str(level_img.dtype),
                extra={**metadata.extra, "pyramid_factor": fac},
            )

            levels.append(
                PyramidLevel(image=level_img, metadata=level_meta, factor=fac)
            )

            logger.debug(
                "Pyramid level ×%d: %dx%d (GSD=%s)",
                fac,
                new_w,
                new_h,
                level_meta.gsd,
            )

        logger.info(
            "Built %d-level pyramid (factors=%s) from %dx%d source.",
            len(levels),
            [l.factor for l in levels],
            metadata.width,
            metadata.height,
        )
        return levels

    def build_pyramid_to_match(
        self,
        source_image: npt.NDArray[Any],
        source_meta: SpatialMetadata,
        target_meta: SpatialMetadata,
    ) -> PyramidLevel:
        """Build the single pyramid level whose GSD best matches *target_meta*.

        Convenience wrapper: computes the GSD ratio, rounds to the
        nearest power of two, and returns the corresponding level.

        Parameters
        ----------
        source_image : ndarray
        source_meta : SpatialMetadata
            Metadata of the high-resolution source.
        target_meta : SpatialMetadata
            Metadata of the reference / target.

        Returns
        -------
        PyramidLevel
            The pyramid level closest in GSD to *target_meta*.
        """
        ratio = self.compute_gsd_ratio(source_meta, target_meta)
        # Snap to nearest power of 2 (≥1)
        factor = max(1, int(2 ** round(math.log2(ratio))))

        levels = self.build_pyramid(
            source_image, source_meta, factors=[factor]
        )
        return levels[0]

    # ------------------------------------------------------------------
    # Pyramid internals
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_2d(
        image: npt.NDArray[Any],
        factor: int,
        interpolation: int = cv2.INTER_AREA,
        antialias: bool = True,
    ) -> npt.NDArray[Any]:
        """Down-sample a 2-D array by *factor* with optional anti-alias blur."""
        h, w = image.shape[:2]
        new_h = max(h // factor, 1)
        new_w = max(w // factor, 1)

        work = image
        if antialias and factor > 1:
            # Gaussian σ ≈ 0.5 × factor is the Nyquist-safe choice
            ksize = int(2 * math.ceil(2 * (0.5 * factor)) + 1)
            work = cv2.GaussianBlur(
                work.astype(np.float32),
                (ksize, ksize),
                sigmaX=0.5 * factor,
            )

        resized = cv2.resize(
            work, (new_w, new_h), interpolation=interpolation
        )
        return resized.astype(image.dtype)

    # ══════════════════════════════════════════════════════════════════
    # Task 4 – Illumination & shadow normalisation
    # ══════════════════════════════════════════════════════════════════

    def normalize_illumination(
        self,
        image: npt.NDArray[Any],
        metadata: SpatialMetadata,
        *,
        methods: Optional[
            Sequence[Literal["percentile_clip", "clahe", "wallis"]]
        ] = None,
        clahe_clip_limit: Optional[float] = None,
        clahe_tile_grid: Optional[Tuple[int, int]] = None,
        percentile_low: Optional[float] = None,
        percentile_high: Optional[float] = None,
        wallis_target_mean: float = 127.0,
        wallis_target_std: float = 50.0,
        wallis_window: int = 61,
    ) -> ProcessedImage:
        """Apply a configurable illumination-normalisation pipeline.

        The *methods* parameter controls which steps run and in what
        order.  The default sequence —
        ``["percentile_clip", "clahe"]`` — is designed for the
        typical lunar scenario: vast dynamic range with near-black
        shadows and saturated sun-lit rims.

        Parameters
        ----------
        image : ndarray
            Input image (any dtype, single- or multi-band).
        metadata : SpatialMetadata
            Spatial metadata (passed through unchanged to output).
        methods : sequence, optional
            Ordered list of normalisation steps.  Supported values:

            * ``"percentile_clip"`` – robust min/max stretch
            * ``"clahe"`` – Contrast-Limited Adaptive Histogram
              Equalisation (local contrast enhancement)
            * ``"wallis"`` – Wallis statistical filter (locally
              normalises mean and variance)
        clahe_clip_limit : float, optional
            Override instance default.
        clahe_tile_grid : tuple[int, int], optional
            Override instance default.
        percentile_low : float, optional
            Override instance default.
        percentile_high : float, optional
            Override instance default.
        wallis_target_mean : float
            Target mean for Wallis filter (in [0, 255] space).
        wallis_target_std : float
            Target standard deviation for Wallis filter.
        wallis_window : int
            Side length of the Wallis local window (must be odd).

        Returns
        -------
        ProcessedImage
            Normalised image (uint8) alongside metadata and provenance.
        """
        if methods is None:
            methods = ["percentile_clip", "clahe"]

        clip_lo = percentile_low if percentile_low is not None else self._pct_lo
        clip_hi = (
            percentile_high if percentile_high is not None else self._pct_hi
        )
        clip_lim = clahe_clip_limit if clahe_clip_limit is not None else self._clip_limit
        tile_g = clahe_tile_grid if clahe_tile_grid is not None else self._tile_grid

        applied: List[str] = []

        # Work in single-channel uint8 for CLAHE / Wallis.
        # Multi-band: process each band independently.
        is_multiband = image.ndim == 3

        if is_multiband:
            bands = [image[b] for b in range(image.shape[0])]
        else:
            bands = [image]

        processed_bands: List[npt.NDArray[np.uint8]] = []

        for band in bands:
            out = band

            for step in methods:
                if step == "percentile_clip":
                    out = self._percentile_clip(out, clip_lo, clip_hi)
                    if "percentile_clip" not in applied:
                        applied.append(
                            f"percentile_clip(low={clip_lo}, high={clip_hi})"
                        )

                elif step == "clahe":
                    out_u8 = _to_gray_u8(out)
                    out = self._apply_clahe(out_u8, clip_lim, tile_g)
                    if "clahe" not in applied:
                        applied.append(
                            f"clahe(clip={clip_lim}, grid={tile_g})"
                        )

                elif step == "wallis":
                    out_u8 = _to_gray_u8(out)
                    out = self._wallis_filter(
                        out_u8,
                        wallis_target_mean,
                        wallis_target_std,
                        wallis_window,
                    )
                    if "wallis" not in applied:
                        applied.append(
                            f"wallis(mean={wallis_target_mean}, "
                            f"std={wallis_target_std}, win={wallis_window})"
                        )

                else:
                    raise ValueError(
                        f"Unknown normalisation method '{step}'. "
                        f"Supported: 'percentile_clip', 'clahe', 'wallis'."
                    )

            processed_bands.append(_to_gray_u8(out))

        if is_multiband:
            result_img: npt.NDArray[np.uint8] = np.stack(
                processed_bands, axis=0
            )
        else:
            result_img = processed_bands[0]

        logger.info("Illumination normalisation applied: %s", applied)

        return ProcessedImage(
            image=result_img,
            metadata=metadata,
            applied_ops=applied,
        )

    # ------------------------------------------------------------------
    # 4-A  Percentile clip (robust min-max stretch)
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_clip(
        image: npt.NDArray[Any],
        p_low: float = 2.0,
        p_high: float = 98.0,
    ) -> npt.NDArray[np.float32]:
        """Stretch pixel values using robust percentile bounds.

        Maps ``[P_low, P_high]`` to ``[0, 1]`` float32, clamping
        outliers.  This removes the influence of extreme shadow pixels
        (≈0 DN) and saturated bright pixels without shifting the bulk
        distribution.
        """
        arr = image.astype(np.float64)
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)

        if hi - lo < 1e-6:
            logger.warning(
                "Percentile clip range is near-zero (lo=%.2f, hi=%.2f); "
                "returning zero array.",
                lo,
                hi,
            )
            return np.zeros(image.shape, dtype=np.float32)

        clipped = (arr - lo) / (hi - lo)
        return clipped.clip(0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 4-B  CLAHE
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clahe(
        image: npt.NDArray[np.uint8],
        clip_limit: float = 3.0,
        tile_grid: Tuple[int, int] = (8, 8),
    ) -> npt.NDArray[np.uint8]:
        """Apply Contrast-Limited Adaptive Histogram Equalisation.

        Unlike global histogram equalisation, CLAHE operates on local
        tiles and clips the histogram to prevent over-amplification in
        uniform regions (e.g. flat mare basalt).  This is critical for
        lunar imagery where crater floors sit at 5–10 DN while
        neighbouring sun-lit ridges saturate at 250+ DN.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image)

    # ------------------------------------------------------------------
    # 4-C  Wallis statistical filter
    # ------------------------------------------------------------------

    @staticmethod
    def _wallis_filter(
        image: npt.NDArray[np.uint8],
        target_mean: float = 127.0,
        target_std: float = 50.0,
        window: int = 61,
    ) -> npt.NDArray[np.uint8]:
        """Wallis adaptive contrast filter.

        Locally normalises each pixel so that the neighbourhood mean
        and standard deviation converge toward ``target_mean`` and
        ``target_std``.  This is especially effective for matching
        images acquired under different solar elevation angles because
        it makes the *local contrast pattern* (the signal used by
        feature detectors) consistent regardless of overall brightness.

        The filter computes local statistics via box-filtering for speed
        (O(1) per pixel).

        Parameters
        ----------
        image : ndarray, uint8
        target_mean : float
            Desired local mean in output (0–255 scale).
        target_std : float
            Desired local standard deviation.
        window : int
            Side length of the local window (must be odd ≥3).

        Returns
        -------
        ndarray, uint8
        """
        if window % 2 == 0:
            window += 1

        img_f = image.astype(np.float64)

        # Local mean via box filter
        local_mean = cv2.blur(img_f, (window, window))
        # Local variance via E[X²] - E[X]²
        local_sq_mean = cv2.blur(img_f * img_f, (window, window))
        local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
        local_std = np.sqrt(local_var) + 1e-8  # avoid division by zero

        # Wallis transform: scale local contrast then shift to target mean
        gain = target_std / local_std
        result = gain * (img_f - local_mean) + target_mean

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Convenience: full preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        data: LunarImageData,
        *,
        pyramid_factors: Optional[Sequence[int]] = None,
        norm_methods: Optional[
            Sequence[Literal["percentile_clip", "clahe", "wallis"]]
        ] = None,
    ) -> Tuple[List[PyramidLevel], ProcessedImage]:
        """Run the complete preprocessing pipeline in one call.

        1. Build the scale pyramid from the raw image.
        2. Normalise illumination on the original-resolution image.

        Parameters
        ----------
        data : LunarImageData
            Output of :pymeth:`LunarDataLoader.load_image`.
        pyramid_factors : sequence of int, optional
            Passed to :pymeth:`build_pyramid`.
        norm_methods : sequence of str, optional
            Passed to :pymeth:`normalize_illumination`.

        Returns
        -------
        pyramid : list[PyramidLevel]
        normalised : ProcessedImage
        """
        pyramid = self.build_pyramid(
            data.image, data.metadata, factors=pyramid_factors
        )
        normalised = self.normalize_illumination(
            data.image, data.metadata, methods=norm_methods
        )
        return pyramid, normalised

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clahe_clip={self._clip_limit}, "
            f"tile_grid={self._tile_grid}, "
            f"pct=[{self._pct_lo}, {self._pct_hi}])"
        )
