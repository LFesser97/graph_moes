"""Utility functions for handling encoding dimensions and configurations."""

from typing import Any, Dict, Optional


# Mapping of encoding names to their dimensions and whether they replace base features.
#
# **Graph encodings** are computed by ``compute_single_graph_encoding``
# (in ``scripts/compute_encodings/compute_encodings_for_datasets.py``).
# All of them *append* features to the original node features – none replace them.
#   - LDP  → PyG ``LocalDegreeProfile``: 5 features (deg, min/max/mean/std neighbour deg)
#   - RWPE → PyG ``AddRandomWalkPE``: *k* features (walk_length=16 → 16)
#   - LAPE → PyG ``AddLaplacianEigenvectorPE``: *k* features (k=8 → 8)
#   - ORC  → ``LocalCurvatureProfile.compute_orc``: 5 features (min/max/mean/std/median ORC)
#
# **Hypergraph encodings** are computed via ``HypergraphEncodings`` and also append.
# Dimensions marked "approximate" should be verified against the hypergraph encoder.
ENCODING_INFO: Dict[str, Dict[str, Any]] = {
    # Graph encodings — all append to original features (replaces_base=False)
    "g_ldp": {
        "encoding_dim": 5,
        "replaces_base": False,
    },  # PyG LocalDegreeProfile: deg, min/max/mean/std neighbour deg
    "g_rwpe_k16": {"encoding_dim": 16, "replaces_base": False},  # walk_length=16
    "g_lape_k8": {"encoding_dim": 8, "replaces_base": False},  # k=8
    "g_orc": {
        "encoding_dim": 5,
        "replaces_base": False,
    },  # LocalCurvatureProfile.compute_orc: min/max/mean/std/median ORC
    # Hypergraph encodings (dimensions need to be verified against HypergraphEncodings)
    # These typically append to original features
    "hg_ldp": {
        "encoding_dim": 4,
        "replaces_base": False,
    },  # Approximate, need to verify
    "hg_frc": {
        "encoding_dim": 1,
        "replaces_base": False,
    },  # Approximate, need to verify
    "hg_orc": {
        "encoding_dim": 2,
        "replaces_base": False,
    },  # Approximate, need to verify
    "hg_rwpe_we_k20": {"encoding_dim": 20, "replaces_base": False},  # k=20
    "hg_lape_normalized_k8": {"encoding_dim": 8, "replaces_base": False},  # k=8
}


def get_encoding_info(encoding_name: str) -> Dict[str, Any]:
    """Get information about an encoding type.

    Args:
        encoding_name: Name of encoding (e.g., "hg_lape_normalized_k8", "g_rwpe_k16", "None")

    Returns:
        Dict with:
            - encoding_dim: int (dimension of encoding features)
            - replaces_base: bool (True if encoding replaces base features, False if appends)
    """
    # Handle "None" encoding (no encoding, just base features)
    if encoding_name == "None" or encoding_name is None:
        return {"encoding_dim": 0, "replaces_base": False}

    if encoding_name in ENCODING_INFO:
        return ENCODING_INFO[encoding_name].copy()

    # Fallback: try to infer from name
    if encoding_name.startswith("g_ldp") or encoding_name.startswith("hg_ldp"):
        # Both graph and hypergraph LDP append features (never replace)
        return {"encoding_dim": 5, "replaces_base": False}
    elif "rwpe" in encoding_name.lower():
        # Extract k/walk_length from name if possible
        if "k16" in encoding_name or "walk_length=16" in encoding_name:
            return {"encoding_dim": 16, "replaces_base": False}
        elif "k20" in encoding_name:
            return {"encoding_dim": 20, "replaces_base": False}
        else:
            return {"encoding_dim": 16, "replaces_base": False}  # Default
    elif "lape" in encoding_name.lower():
        # Extract k from name if possible
        if "k8" in encoding_name:
            return {"encoding_dim": 8, "replaces_base": False}
        else:
            return {"encoding_dim": 8, "replaces_base": False}  # Default
    elif "orc" in encoding_name.lower():
        # LocalCurvatureProfile.compute_orc returns 5 features (min/max/mean/std/median)
        return {"encoding_dim": 5, "replaces_base": False}
    elif "frc" in encoding_name.lower():
        return {"encoding_dim": 1, "replaces_base": False}
    else:
        # Unknown encoding - default assumptions
        return {"encoding_dim": 8, "replaces_base": False}


def extract_encoding_dim_from_graph(
    base_features: Optional[int], encoded_features: int, encoding_name: str
) -> int:
    """Extract encoding dimension from precomputed encoding.

    Args:
        base_features: Dimension of base features (before encoding), or None if unknown
        encoded_features: Dimension of features in encoded graph
        encoding_name: Name of encoding

    Returns:
        encoding_dim: Dimension of encoding features
    """
    encoding_info = get_encoding_info(encoding_name)

    if encoding_info["replaces_base"]:
        # Encoding replaces base - total dims is encoding dim
        return encoded_features
    else:
        # Encoding appends - subtract base dims
        if base_features is not None:
            return encoded_features - base_features
        else:
            # Fallback: use known encoding dim
            dim: int = encoding_info["encoding_dim"]
            return dim


def create_encoding_config(
    encoding_name: str,
    base_input_dim: Optional[int] = None,
    encoded_input_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Create encoding configuration dict for EncodingMoE.

    Args:
        encoding_name: Name of encoding (e.g., "hg_lape_normalized_k8")
        base_input_dim: Dimension of base features (optional, for validation)
        encoded_input_dim: Dimension of encoded features (optional, for validation)

    Returns:
        Dict with:
            - encoding_name: str
            - encoding_dim: int
            - replaces_base: bool
    """
    encoding_info = get_encoding_info(encoding_name)

    config = {
        "encoding_name": encoding_name,
        "encoding_dim": encoding_info["encoding_dim"],
        "replaces_base": encoding_info["replaces_base"],
    }

    # Validate dimensions if provided
    if base_input_dim is not None and encoded_input_dim is not None:
        if encoding_info["replaces_base"]:
            # Encoding replaces base
            expected_encoded = encoding_info["encoding_dim"]
            if encoded_input_dim != expected_encoded:
                print(
                    f"⚠️  Warning: Encoding {encoding_name} expected {expected_encoded} dims "
                    f"but got {encoded_input_dim}"
                )
        else:
            # Encoding appends to base
            expected_encoded = base_input_dim + encoding_info["encoding_dim"]
            if encoded_input_dim != expected_encoded:
                print(
                    f"⚠️  Warning: Encoding {encoding_name} expected {expected_encoded} dims "
                    f"(base={base_input_dim} + encoding={encoding_info['encoding_dim']}) "
                    f"but got {encoded_input_dim}"
                )
                # Try to infer actual encoding dim
                actual_encoding_dim = encoded_input_dim - base_input_dim
                if actual_encoding_dim > 0:
                    config["encoding_dim"] = actual_encoding_dim
                    print(f"   Using inferred encoding dim: {actual_encoding_dim}")

    return config
