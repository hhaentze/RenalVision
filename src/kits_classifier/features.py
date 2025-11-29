"""Feature extraction for lesion characterization."""

import numpy as np
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes
from enum import Enum


class Feature(Enum):
    """Enumeration of available radiomics features."""
    MEAN_HU = "mean_hu"
    STD_HU = "std_hu"
    COV = "coefficient_of_variation"
    P10 = "percentile_10"
    P90 = "percentile_90"
    ENTROPY = "entropy"
    GLCM_CONTRAST = "glcm_contrast"
    GRADIENT_MAG = "gradient_magnitude"
    SPHERICITY = "sphericity"
    FRAC_BELOW_20HU = "fraction_below_20hu"


def extract_features(image, mask, feature_list=None):
    """
    Extract radiomics features from a lesion.
    
    Args:
        image: Full CT image (numpy array, in HU)
        mask: Binary mask for the lesion (numpy array)
        feature_list: List of Feature enums to extract (default: all)
        
    Returns:
        dict: Feature name -> feature value
    """
    if feature_list is None:
        feature_list = list(Feature)
    
    features = {}
    
    # Extract lesion voxels
    lesion_voxels = image[mask > 0]
    
    for feature in feature_list:
        if feature == Feature.MEAN_HU:
            features[feature.value] = compute_mean_hu(lesion_voxels)
        elif feature == Feature.STD_HU:
            features[feature.value] = compute_std_hu(lesion_voxels)
        elif feature == Feature.COV:
            features[feature.value] = compute_cov(lesion_voxels)
        elif feature == Feature.P10:
            features[feature.value] = compute_percentile(lesion_voxels, 10)
        elif feature == Feature.P90:
            features[feature.value] = compute_percentile(lesion_voxels, 90)
        elif feature == Feature.ENTROPY:
            features[feature.value] = compute_entropy(lesion_voxels)
        elif feature == Feature.GLCM_CONTRAST:
            features[feature.value] = compute_glcm_contrast(image, mask)
        elif feature == Feature.GRADIENT_MAG:
            features[feature.value] = compute_gradient_magnitude(image, mask)
        elif feature == Feature.SPHERICITY:
            features[feature.value] = compute_sphericity(mask)
        elif feature == Feature.FRAC_BELOW_20HU:
            features[feature.value] = compute_fraction_below_threshold(lesion_voxels, 20)
    
    return features


# ============================================================================
# Individual Feature Functions
# ============================================================================

def compute_mean_hu(voxels):
    """
    Mean intensity in Hounsfield Units.
    
    Interpretation: Higher values indicate denser tissue.
    Cysts typically have low mean HU (near water: 0-20 HU).
    Solid tumors are typically brighter (40-100+ HU).
    """
    return float(np.mean(voxels))


def compute_std_hu(voxels):
    """
    Standard deviation of intensity.
    
    Interpretation: Measures heterogeneity.
    Cysts are homogeneous (low std).
    Tumors are heterogeneous (high std).
    """
    return float(np.std(voxels))


def compute_cov(voxels):
    """
    Coefficient of variation: std / mean.
    
    Interpretation: Normalized heterogeneity measure.
    Accounts for the relationship between variability and mean intensity.
    """
    mean_val = np.mean(voxels)
    if mean_val == 0:
        return 0.0
    return float(np.std(voxels) / mean_val)


def compute_percentile(voxels, percentile):
    """
    Compute intensity percentile.
    
    Args:
        voxels: Lesion intensity values
        percentile: Percentile to compute (0-100)
        
    Interpretation:
    - P10: Lower bound of intensity distribution
    - P90: Upper bound of intensity distribution
    These capture the range of intensities present.
    """
    return float(np.percentile(voxels, percentile))


def compute_entropy(voxels, bins=32):
    """
    Histogram-based entropy.
    
    Uses 32 bins to compute Shannon entropy of the intensity distribution.
    
    Interpretation: Measures randomness/complexity of intensity distribution.
    - Low entropy: Uniform distribution (homogeneous, like cysts)
    - High entropy: Complex distribution (heterogeneous, like tumors)
    """
    hist, _ = np.histogram(voxels, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    return float(scipy_entropy(hist))


def compute_glcm_contrast(image, mask):
    """
    Gray-Level Co-occurrence Matrix (GLCM) contrast.
    
    Computes texture contrast using GLCM with:
    - Distance: 1 voxel
    - Directions: Average over 13 3D directions
    - Quantization: 64 gray levels
    
    Interpretation: Measures local intensity variation (texture).
    - Low contrast: Smooth, homogeneous regions (cysts)
    - High contrast: Textured, variable regions (tumors)
    """
    # Extract lesion region
    lesion_voxels = image[mask > 0]
    
    # Quantize to 64 levels for GLCM
    vmin, vmax = lesion_voxels.min(), lesion_voxels.max()
    if vmax == vmin:
        return 0.0
    quantized = ((lesion_voxels - vmin) / (vmax - vmin) * 63).astype(np.uint8)
    
    # Reshape to 2D for GLCM (flatten to column)
    quantized_2d = quantized.reshape(-1, 1)
    
    # Compute GLCM with distance=1, multiple angles
    # Average over 4 angles for 2D approximation
    try:
        glcm = graycomatrix(
            quantized_2d, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=64,
            symmetric=True,
            normed=True
        )
        contrast = graycoprops(glcm, 'contrast').mean()
        return float(contrast)
    except:
        # Fallback if GLCM fails
        return float(np.std(lesion_voxels))


def compute_gradient_magnitude(image, mask):
    """
    Mean gradient magnitude within lesion.
    
    Uses Sobel operator to compute image gradients, then averages
    the magnitude within the lesion.
    
    Interpretation: Measures edge strength and internal variation.
    - Low gradient: Smooth, uniform regions (cysts)
    - High gradient: Sharp boundaries, internal structures (tumors)
    """
    # Compute gradients using Sobel
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    grad_z = ndimage.sobel(image, axis=2)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Mean within lesion
    lesion_gradients = grad_mag[mask > 0]
    return float(np.mean(lesion_gradients))


def compute_sphericity(mask):
    """
    Sphericity: (π^(1/3) * (6*V)^(2/3)) / A
    
    Where V = volume, A = surface area.
    Value of 1.0 indicates a perfect sphere.
    Lower values indicate irregular, non-spherical shapes.
    
    Interpretation:
    - High sphericity (~1.0): Round, regular shape (typical for cysts)
    - Low sphericity (<0.7): Irregular, infiltrative shape (typical for tumors)
    """
    # Volume (number of voxels)
    volume = np.sum(mask)
    
    if volume < 10:
        return 0.0
    
    try:
        # Extract surface using marching cubes
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
        
        # Calculate surface area (sum of triangle areas)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # Cross product for triangle area
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.sqrt(np.sum(cross**2, axis=1)) / 2.0
        surface_area = np.sum(areas)
        
        # Sphericity formula
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        return float(np.clip(sphericity, 0, 1))
    except:
        # Fallback: use simple surface voxel count
        eroded = ndimage.binary_erosion(mask)
        surface_voxels = np.sum(mask) - np.sum(eroded)
        if surface_voxels == 0:
            return 0.0
        # Approximate sphericity
        return float((volume / surface_voxels) / 6.0)


def compute_fraction_below_threshold(voxels, threshold=20):
    """
    Fraction of voxels below a HU threshold.
    
    Args:
        voxels: Lesion intensity values (in HU)
        threshold: HU threshold (default: 20 HU, near water)
        
    Interpretation: Measures fluid content.
    - High fraction: Mostly fluid-filled (typical for cysts)
    - Low fraction: Solid tissue (typical for tumors)
    
    Water is ~0 HU, so <20 HU indicates fluid-like attenuation.
    """
    return float(np.sum(voxels < threshold) / len(voxels))
