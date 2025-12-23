# =============================================================================
# XAI TO TEXT CONVERTER
# Converts Grad-CAM visual explanations to textual descriptions
# =============================================================================
"""
XAI to Text Converter: The Bridge from Visual to Textual Explanations.

THIS IS THE NOVEL CONTRIBUTION OF THE PROJECT.

WHAT DOES THIS DO?
    1. Analyzes Grad-CAM heatmap to identify high-attention regions
    2. Determines spatial location (peripheral, central, upper, lower)
    3. Estimates intensity and distribution of attention
    4. Converts these visual cues into natural language descriptions

WHY IS THIS IMPORTANT?
    - Grad-CAM shows WHERE the model looks (visual)
    - This module describes WHERE in words (textual)
    - These words become queries for the knowledge base
    - The knowledge base explains WHY this matters (medical context)

PIPELINE:
    Grad-CAM Heatmap → Spatial Analysis → Textual Description → RAG Query

EXAMPLE:
    Input: Heatmap showing high activation in upper-right peripheral region
    Output: "Model focused on peripheral upper lobe region with high intensity"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class XAITextConverter:
    """
    Converts Grad-CAM heatmaps to textual descriptions.
    
    This class bridges the gap between visual XAI and RAG by:
        1. Analyzing spatial distribution of attention
        2. Quantifying attention intensity
        3. Generating natural language descriptions
    
    Example:
        >>> converter = XAITextConverter()
        >>> description = converter.convert(heatmap, predicted_class="adenocarcinoma")
        >>> print(description['summary'])
        "High attention in peripheral upper region, consistent with adenocarcinoma patterns."
    """
    
    def __init__(self):
        """Initialize the XAI to Text converter."""
        # Attention intensity thresholds
        self.intensity_thresholds = {
            'very_high': 0.8,
            'high': 0.6,
            'moderate': 0.4,
            'low': 0.2
        }
        
        # Region names for spatial description
        self.region_names = {
            'top_left': 'upper left',
            'top_center': 'upper central',
            'top_right': 'upper right',
            'middle_left': 'middle left',
            'middle_center': 'central',
            'middle_right': 'middle right',
            'bottom_left': 'lower left',
            'bottom_center': 'lower central',
            'bottom_right': 'lower right'
        }
    
    def convert(
        self,
        heatmap: np.ndarray,
        predicted_class: str,
        confidence: float = 0.0
    ) -> Dict:
        """
        Convert a Grad-CAM heatmap to textual description.
        
        Args:
            heatmap: Grad-CAM heatmap [H, W] with values in [0, 1]
            predicted_class: Name of the predicted class
            confidence: Prediction confidence (0-1)
        
        Returns:
            Dictionary containing:
                - summary: Main textual description
                - spatial_info: Detailed spatial analysis
                - intensity_info: Attention intensity analysis
                - keywords: Keywords for knowledge base query
        """
        # Analyze the heatmap
        spatial_info = self._analyze_spatial_distribution(heatmap)
        intensity_info = self._analyze_intensity(heatmap)
        focus_regions = self._get_focus_regions(heatmap)
        
        # Generate textual descriptions
        spatial_description = self._generate_spatial_description(spatial_info)
        intensity_description = self._generate_intensity_description(intensity_info)
        
        # Generate keywords for RAG query
        keywords = self._generate_keywords(
            predicted_class, spatial_info, intensity_info
        )
        
        # Create summary
        summary = self._generate_summary(
            predicted_class,
            confidence,
            spatial_description,
            intensity_description
        )
        
        return {
            'summary': summary,
            'spatial_info': spatial_info,
            'intensity_info': intensity_info,
            'spatial_description': spatial_description,
            'intensity_description': intensity_description,
            'focus_regions': focus_regions,
            'keywords': keywords,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    
    def _analyze_spatial_distribution(self, heatmap: np.ndarray) -> Dict:
        """
        Analyze spatial distribution of attention in the heatmap.
        
        Divides image into 3x3 grid and calculates attention in each region.
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
        
        Returns:
            Dictionary with attention values for each region
        """
        h, w = heatmap.shape
        h_third, w_third = h // 3, w // 3
        
        regions = {}
        region_keys = [
            ('top_left', 0, h_third, 0, w_third),
            ('top_center', 0, h_third, w_third, 2*w_third),
            ('top_right', 0, h_third, 2*w_third, w),
            ('middle_left', h_third, 2*h_third, 0, w_third),
            ('middle_center', h_third, 2*h_third, w_third, 2*w_third),
            ('middle_right', h_third, 2*h_third, 2*w_third, w),
            ('bottom_left', 2*h_third, h, 0, w_third),
            ('bottom_center', 2*h_third, h, w_third, 2*w_third),
            ('bottom_right', 2*h_third, h, 2*w_third, w)
        ]
        
        for name, y1, y2, x1, x2 in region_keys:
            region = heatmap[y1:y2, x1:x2]
            regions[name] = {
                'mean': float(np.mean(region)),
                'max': float(np.max(region)),
                'coverage': float(np.mean(region > 0.5))  # % of high attention
            }
        
        # Calculate peripheral vs central
        peripheral_regions = ['top_left', 'top_right', 'bottom_left', 'bottom_right',
                             'top_center', 'bottom_center', 'middle_left', 'middle_right']
        central_region = ['middle_center']
        
        peripheral_mean = np.mean([regions[r]['mean'] for r in peripheral_regions])
        central_mean = regions['middle_center']['mean']
        
        regions['peripheral_vs_central'] = {
            'peripheral_mean': peripheral_mean,
            'central_mean': central_mean,
            'is_peripheral': peripheral_mean > central_mean,
            'is_central': central_mean > peripheral_mean
        }
        
        # Find primary focus region
        max_region = max(regions.items(), 
                        key=lambda x: x[1]['mean'] if isinstance(x[1], dict) and 'mean' in x[1] else 0)
        regions['primary_focus'] = max_region[0]
        
        return regions
    
    def _analyze_intensity(self, heatmap: np.ndarray) -> Dict:
        """
        Analyze overall attention intensity.
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
        
        Returns:
            Dictionary with intensity statistics
        """
        return {
            'mean': float(np.mean(heatmap)),
            'max': float(np.max(heatmap)),
            'std': float(np.std(heatmap)),
            'high_attention_ratio': float(np.mean(heatmap > 0.5)),
            'very_high_attention_ratio': float(np.mean(heatmap > 0.8)),
            'is_focused': float(np.std(heatmap)) > 0.2,  # Concentrated attention
            'is_diffuse': float(np.std(heatmap)) < 0.15  # Spread out attention
        }
    
    def _get_focus_regions(self, heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """
        Get specific regions of high attention.
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
            threshold: Attention threshold for focus regions
        
        Returns:
            List of focus region dictionaries
        """
        h, w = heatmap.shape
        focus_mask = heatmap > threshold
        
        # Find centroid of high-attention region
        if focus_mask.any():
            y_coords, x_coords = np.where(focus_mask)
            centroid_y = np.mean(y_coords) / h
            centroid_x = np.mean(x_coords) / w
            
            # Determine location description
            y_loc = 'upper' if centroid_y < 0.33 else ('middle' if centroid_y < 0.66 else 'lower')
            x_loc = 'left' if centroid_x < 0.33 else ('central' if centroid_x < 0.66 else 'right')
            
            return [{
                'centroid': (centroid_y, centroid_x),
                'location': f"{y_loc} {x_loc}",
                'size_ratio': float(np.sum(focus_mask)) / (h * w),
                'max_intensity': float(heatmap[focus_mask].max())
            }]
        
        return []
    
    def _generate_spatial_description(self, spatial_info: Dict) -> str:
        """Generate natural language description of spatial distribution."""
        primary_focus = spatial_info.get('primary_focus', 'middle_center')
        focus_name = self.region_names.get(primary_focus, primary_focus)
        
        pvc = spatial_info.get('peripheral_vs_central', {})
        
        if pvc.get('is_peripheral', False):
            location_type = "peripheral"
        elif pvc.get('is_central', False):
            location_type = "central"
        else:
            location_type = "distributed"
        
        description = f"Attention is primarily focused on the {focus_name} region, "
        description += f"with {location_type} distribution pattern."
        
        return description
    
    def _generate_intensity_description(self, intensity_info: Dict) -> str:
        """Generate natural language description of attention intensity."""
        max_val = intensity_info['max']
        
        if max_val > 0.8:
            intensity_level = "very high"
        elif max_val > 0.6:
            intensity_level = "high"
        elif max_val > 0.4:
            intensity_level = "moderate"
        else:
            intensity_level = "low"
        
        if intensity_info.get('is_focused', False):
            pattern = "concentrated in specific areas"
        elif intensity_info.get('is_diffuse', False):
            pattern = "diffusely distributed across the image"
        else:
            pattern = "moderately distributed"
        
        description = f"Model shows {intensity_level} attention intensity, {pattern}."
        
        return description
    
    def _generate_keywords(
        self,
        predicted_class: str,
        spatial_info: Dict,
        intensity_info: Dict
    ) -> List[str]:
        """
        Generate keywords for knowledge base retrieval.
        
        These keywords will be used to query the MedicalKnowledgeBase.
        """
        keywords = []
        
        # Add class-specific keywords
        # Handles both original names and folder names with spaces
        class_keywords = {
            'adenocarcinoma': ['adenocarcinoma', 'peripheral', 'ground glass'],
            'squamous_cell_carcinoma': ['squamous', 'central', 'cavitation'],
            'squamous cell carcinoma': ['squamous', 'central', 'cavitation'],
            'large_cell_carcinoma': ['large cell', 'mass', 'aggressive'],
            'large cell carcinoma': ['large cell', 'mass', 'aggressive'],
            'normal': ['normal', 'healthy', 'clear'],
            'normal cases': ['normal', 'healthy', 'clear'],
            'benign': ['benign', 'normal', 'non-malignant'],
            'benign cases': ['benign', 'normal', 'non-malignant', 'healthy']
        }
        
        keywords.extend(class_keywords.get(predicted_class.lower(), [predicted_class.lower()]))
        
        # Add spatial keywords
        pvc = spatial_info.get('peripheral_vs_central', {})
        if pvc.get('is_peripheral', False):
            keywords.append('peripheral')
        if pvc.get('is_central', False):
            keywords.extend(['central', 'hilar'])
        
        # Add location keywords
        primary_focus = spatial_info.get('primary_focus', '')
        if 'top' in primary_focus:
            keywords.append('upper')
        if 'bottom' in primary_focus:
            keywords.append('lower')
        
        # Add intensity keywords
        if intensity_info.get('max', 0) > 0.7:
            keywords.extend(['high', 'intensity', 'activation'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _generate_summary(
        self,
        predicted_class: str,
        confidence: float,
        spatial_description: str,
        intensity_description: str
    ) -> str:
        """Generate a complete summary of the XAI analysis."""
        # Format class name nicely
        class_display = predicted_class.replace('_', ' ').title()
        
        summary = f"The model predicts {class_display}"
        if confidence > 0:
            summary += f" with {confidence*100:.1f}% confidence"
        summary += ". "
        
        summary += spatial_description + " "
        summary += intensity_description
        
        return summary
