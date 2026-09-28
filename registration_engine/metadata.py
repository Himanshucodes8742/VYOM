"""Metadata extraction utilities for lunar orbital imagery (PDS4 XML labels)."""

from pathlib import Path
import xml.etree.ElementTree as ET


class MetadataError(Exception):
    """Raised when metadata fields cannot be found or parsed from product labels."""
    pass


def get_sun_elevation(label_path: str | Path) -> float:
    """Parse a PDS4 XML label and return the sun elevation angle in degrees.

    Args:
        label_path: Path to the PDS4 XML label file (.xml).

    Returns:
        float: Sun elevation angle in degrees.

    Raises:
        MetadataError: If the label file does not exist, cannot be parsed as XML,
                       or if the sun elevation field is not found or is invalid.
    """
    path = Path(label_path)
    if not path.is_file():
        raise MetadataError(f"Label file not found: {label_path}")

    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except ET.ParseError as e:
        raise MetadataError(f"Failed to parse XML from {label_path}: {e}")
    except Exception as e:
        raise MetadataError(f"Error reading label file {label_path}: {e}")

    # Search for sun_elevation element regardless of XML namespace
    sun_elev_elem = root.find(".//{*}sun_elevation")
    if sun_elev_elem is None or sun_elev_elem.text is None:
        for elem in root.iter():
            tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag_name.lower() == "sun_elevation" and elem.text is not None:
                sun_elev_elem = elem
                break

    if sun_elev_elem is None or sun_elev_elem.text is None or not sun_elev_elem.text.strip():
        raise MetadataError(
            f"Sun elevation field ('sun_elevation') not found in PDS4 label: {label_path}"
        )

    val_str = sun_elev_elem.text.strip()
    try:
        sun_elevation = float(val_str)
    except ValueError as e:
        raise MetadataError(
            f"Failed to convert sun elevation value '{val_str}' to float in {label_path}: {e}"
        )

    return sun_elevation
