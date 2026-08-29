"""
data_loader.py – Multi-format loader for lunar image registration pipeline.

Supports:
  • GeoTIFF  (.tif / .tiff)  via ``rasterio``
  • PDS4     (.xml label + .img/.tif array) via ``pds4_tools``

Usage::

    loader = LunarDataLoader()
    result = loader.load_image("path/to/image.tif")
    print(result.metadata)
    plt.imshow(result.image, cmap="gray")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class UnsupportedFormatError(Exception):
    """Raised when the file extension is not supported by the loader."""


class ImageLoadError(Exception):
    """Raised when a supported file cannot be parsed or read."""


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialMetadata:
    """Geospatial metadata extracted from a lunar image file.

    Attributes
    ----------
    crs : str | None
        Coordinate Reference System as a WKT or PROJ string.
        ``None`` when the source format carries no CRS information
        (common for raw PDS4 data without map-projection).
    transform : tuple[float, ...] | None
        Affine transform coefficients that map pixel coordinates to
        geospatial coordinates (rasterio convention: a, b, c, d, e, f).
        ``None`` when the source has no affine transform.
    gsd : tuple[float, float] | None
        Ground-Sample Distance (spatial resolution) in map units
        ``(gsd_x, gsd_y)``.  Derived from the transform when available.
    width : int
        Number of columns (pixels) in the image.
    height : int
        Number of rows (pixels) in the image.
    band_count : int
        Number of bands / channels.
    dtype : str
        Numpy dtype string of the image array (e.g. ``'float32'``).
    extra : dict[str, Any]
        Any additional metadata that does not fit the fields above
        (PDS4 label contents, TIFF tags, etc.).
    """

    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    gsd: Optional[Tuple[float, float]] = None
    width: int = 0
    height: int = 0
    band_count: int = 1
    dtype: str = "float64"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LunarImageData:
    """Container returned by :pymeth:`LunarDataLoader.load_image`.

    Attributes
    ----------
    image : numpy.ndarray
        The image pixel data.  Shape is ``(H, W)`` for single-band or
        ``(bands, H, W)`` for multi-band imagery.
    metadata : SpatialMetadata
        Parsed geospatial metadata.
    source_path : str
        Absolute path of the file that was loaded.
    """

    image: npt.NDArray[Any]
    metadata: SpatialMetadata
    source_path: str


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

_GEOTIFF_EXTENSIONS: frozenset[str] = frozenset({".tif", ".tiff"})
_PDS4_EXTENSIONS: frozenset[str] = frozenset({".xml"})
_ALL_SUPPORTED: frozenset[str] = _GEOTIFF_EXTENSIONS | _PDS4_EXTENSIONS


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------


class LunarDataLoader:
    """Unified loader for GeoTIFF and PDS4 lunar image formats.

    Parameters
    ----------
    default_crs : str, optional
        A fallback CRS string to attach when the source file does not
        carry its own CRS (e.g. ``"EPSG:104903"`` for the IAU Moon 2000
        geographic CRS).  Defaults to ``None``.

    Examples
    --------
    >>> loader = LunarDataLoader(default_crs="EPSG:104903")
    >>> data = loader.load_image("ohrc_strip.tif")
    >>> data.image.shape
    (4096, 4096)
    >>> data.metadata.gsd
    (0.25, 0.25)
    """

    def __init__(self, default_crs: Optional[str] = None) -> None:
        self._default_crs = default_crs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, filepath: Union[str, Path]) -> LunarImageData:
        """Load a lunar image from *filepath*, auto-detecting its format.

        Parameters
        ----------
        filepath : str | Path
            Path to a GeoTIFF (``.tif`` / ``.tiff``) or PDS4 label
            (``.xml``) file.

        Returns
        -------
        LunarImageData
            Dataclass containing the image array, spatial metadata, and
            the resolved source path.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist on disk.
        UnsupportedFormatError
            If the file extension is not one of the supported formats.
        ImageLoadError
            If the file exists and has a recognised extension but
            cannot be read or parsed.
        """
        path = Path(filepath).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        suffix = path.suffix.lower()

        if suffix in _GEOTIFF_EXTENSIONS:
            logger.info("Detected GeoTIFF format for '%s'", path.name)
            return self._load_geotiff(path)

        if suffix in _PDS4_EXTENSIONS:
            logger.info("Detected PDS4 label for '%s'", path.name)
            return self._load_pds4(path)

        raise UnsupportedFormatError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported extensions: {sorted(_ALL_SUPPORTED)}"
        )

    @staticmethod
    def supported_extensions() -> frozenset[str]:
        """Return the set of file extensions this loader can handle."""
        return _ALL_SUPPORTED

    # ------------------------------------------------------------------
    # GeoTIFF loader (rasterio)
    # ------------------------------------------------------------------

    def _load_geotiff(self, path: Path) -> LunarImageData:
        """Read a GeoTIFF file via ``rasterio``.

        Reads all bands into a single numpy array.  For single-band
        images the returned array is squeezed to ``(H, W)``.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "The 'rasterio' library is required to load GeoTIFF files. "
                "Install it with:  pip install rasterio"
            ) from exc

        try:
            with rasterio.open(path) as src:
                # Read all bands – shape (bands, H, W)
                image: npt.NDArray[Any] = src.read()

                # Squeeze single-band to (H, W)
                if image.shape[0] == 1:
                    image = image.squeeze(axis=0)

                crs_str: Optional[str] = None
                if src.crs is not None:
                    crs_str = src.crs.to_wkt()
                elif self._default_crs is not None:
                    crs_str = self._default_crs

                transform_tuple: Optional[Tuple[float, ...]] = None
                gsd: Optional[Tuple[float, float]] = None
                if src.transform is not None:
                    t = src.transform
                    transform_tuple = (t.a, t.b, t.c, t.d, t.e, t.f)
                    gsd = (abs(t.a), abs(t.e))

                meta = SpatialMetadata(
                    crs=crs_str,
                    transform=transform_tuple,
                    gsd=gsd,
                    width=src.width,
                    height=src.height,
                    band_count=src.count,
                    dtype=str(image.dtype),
                    extra={
                        "driver": src.driver,
                        "nodata": src.nodata,
                        "tags": src.tags(),
                    },
                )

                logger.debug(
                    "GeoTIFF loaded: %dx%d, %d band(s), dtype=%s",
                    src.width,
                    src.height,
                    src.count,
                    image.dtype,
                )

        except ImportError:
            raise  # re-raise so the outer handler doesn't mask it
        except Exception as exc:
            raise ImageLoadError(
                f"Failed to read GeoTIFF '{path}': {exc}"
            ) from exc

        return LunarImageData(
            image=image,
            metadata=meta,
            source_path=str(path),
        )

    # ------------------------------------------------------------------
    # PDS4 loader (pds4_tools)
    # ------------------------------------------------------------------

    def _load_pds4(self, path: Path) -> LunarImageData:
        """Read a PDS4 product from its XML label via ``pds4_tools``.

        The method locates the first ``Array_2D_Image`` or
        ``Array_2D`` structure in the label and extracts its data.
        If the label contains multiple arrays the first one is used and
        a warning is emitted.
        """
        try:
            import pds4_tools
        except ImportError as exc:
            raise ImportError(
                "The 'pds4_tools' library is required to load PDS4 files. "
                "Install it with:  pip install pds4_tools"
            ) from exc

        try:
            structures: Sequence[Any] = pds4_tools.pds4_read(
                str(path), quiet=True
            )
        except Exception as exc:
            raise ImageLoadError(
                f"pds4_tools failed to parse label '{path}': {exc}"
            ) from exc

        if len(structures) == 0:
            raise ImageLoadError(
                f"PDS4 label '{path}' contains no data structures."
            )

        if len(structures) > 1:
            logger.warning(
                "PDS4 label '%s' contains %d structures; "
                "using the first one.",
                path.name,
                len(structures),
            )

        structure = structures[0]

        try:
            image: npt.NDArray[Any] = np.asarray(structure.data, dtype=float)
        except Exception as exc:
            raise ImageLoadError(
                f"Could not convert PDS4 data to numpy array: {exc}"
            ) from exc

        # ------- attempt to extract spatial metadata from the label -------
        crs_str, transform_tuple, gsd = self._extract_pds4_spatial(
            structure, path
        )

        band_count = 1 if image.ndim == 2 else image.shape[0]
        height = image.shape[-2]
        width = image.shape[-1]

        # Collect extra metadata from the PDS4 label when available.
        extra: Dict[str, Any] = {}
        if hasattr(structure, "label"):
            try:
                extra["pds4_label"] = str(structure.label)
            except Exception:
                pass
        if hasattr(structure, "meta_data"):
            meta_obj = structure.meta_data
            for attr in ("description", "unit", "name"):
                val = getattr(meta_obj, attr, None)
                if val is not None:
                    extra[f"pds4_{attr}"] = str(val)

        meta = SpatialMetadata(
            crs=crs_str if crs_str is not None else self._default_crs,
            transform=transform_tuple,
            gsd=gsd,
            width=width,
            height=height,
            band_count=band_count,
            dtype=str(image.dtype),
            extra=extra,
        )

        logger.debug(
            "PDS4 loaded: %dx%d, %d band(s), dtype=%s",
            width,
            height,
            band_count,
            image.dtype,
        )

        return LunarImageData(
            image=image,
            metadata=meta,
            source_path=str(path),
        )

    # ------------------------------------------------------------------
    # PDS4 spatial metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pds4_spatial(
        structure: Any,
        path: Path,
    ) -> Tuple[
        Optional[str],
        Optional[Tuple[float, ...]],
        Optional[Tuple[float, float]],
    ]:
        """Best-effort extraction of CRS / transform / GSD from a PDS4
        label.

        PDS4 labels are highly variable; this parser covers the common
        ``Cartography`` / ``Map_Projection`` discipline areas used by
        ISRO and NASA for lunar products.

        Returns
        -------
        crs : str | None
        transform : tuple[float, ...] | None
        gsd : tuple[float, float] | None
        """
        crs_str: Optional[str] = None
        transform_tuple: Optional[Tuple[float, ...]] = None
        gsd: Optional[Tuple[float, float]] = None

        if not hasattr(structure, "label"):
            return crs_str, transform_tuple, gsd

        label_text = str(structure.label)

        # ---- CRS: look for target body name ----------------------------
        try:
            import re

            body_match = re.search(
                r"<name>\s*(Moon|Luna)\s*</name>",
                label_text,
                re.IGNORECASE,
            )
            if body_match:
                crs_str = "IAU:30100"  # IAU Moon 2015 Sphere
        except Exception:
            pass

        # ---- GSD / pixel scale -----------------------------------------
        try:
            import re

            scale_match = re.search(
                r"<pixel_resolution_x[^>]*>\s*"
                r"<value>([0-9.eE+-]+)</value>",
                label_text,
            )
            if scale_match:
                res_x = float(scale_match.group(1))
            else:
                # Fallback: look for <map_scale>
                scale_match = re.search(
                    r"<map_scale[^>]*>\s*([0-9.eE+-]+)",
                    label_text,
                )
                res_x = float(scale_match.group(1)) if scale_match else None

            scale_match_y = re.search(
                r"<pixel_resolution_y[^>]*>\s*"
                r"<value>([0-9.eE+-]+)</value>",
                label_text,
            )
            res_y = (
                float(scale_match_y.group(1))
                if scale_match_y
                else res_x
            )

            if res_x is not None and res_y is not None:
                gsd = (res_x, res_y)
        except Exception:
            logger.debug(
                "Could not extract GSD from PDS4 label '%s'.", path.name
            )

        # ---- Affine transform from upper-left coords + GSD -------------
        try:
            import re

            ul_x_match = re.search(
                r"<upperleft_corner_x[^>]*>\s*"
                r"<value>([0-9.eE+-]+)</value>",
                label_text,
            )
            ul_y_match = re.search(
                r"<upperleft_corner_y[^>]*>\s*"
                r"<value>([0-9.eE+-]+)</value>",
                label_text,
            )
            if ul_x_match and ul_y_match and gsd is not None:
                ul_x = float(ul_x_match.group(1))
                ul_y = float(ul_y_match.group(1))
                # Affine: (scale_x, 0, origin_x, 0, -scale_y, origin_y)
                transform_tuple = (
                    gsd[0],
                    0.0,
                    ul_x,
                    0.0,
                    -gsd[1],
                    ul_y,
                )
        except Exception:
            logger.debug(
                "Could not build affine transform from PDS4 label '%s'.",
                path.name,
            )

        return crs_str, transform_tuple, gsd

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(default_crs={self._default_crs!r})"
        )
