"""
Realistic SAR Simulator for Oil Spill Detection
Generates physics-based synthetic SAR imagery with oil spill signatures
"""

import numpy as np
import rasterio
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import random
import json
from datetime import datetime


@dataclass
class SARParameters:
    """Physical parameters for realistic SAR simulation"""
    # Sensor parameters (Sentinel-1 C-band)
    wavelength: float = 0.055  # 5.5 cm
    incidence_angle: float = 23.0  # Degrees, IW mode
    resolution_m: float = 10.0  # 10m resolution
    
    # Environmental conditions
    wind_speed: float = 5.0  # m/s
    wind_direction: float = 0.0  # Degrees from North
    current_velocity: float = 0.5  # m/s
    significant_wave_height: float = 2.0  # meters
    
    # Oil properties
    oil_viscosity: float = 500.0  # cSt, typical crude
    spill_thickness_mm: float = 0.1  # 0.1 to 10mm range
    oil_age_hours: float = 24.0  # Weathering time
    
    # Noise parameters
    speckle_looks: int = 4  # Equivalent number of looks
    thermal_noise_db: float = -25.0


class RealisticSARSimulator:
    """
    Generate synthetic SAR with realistic physics-based backscatter
    for oil spill detection training and validation
    """
    
    def __init__(self, image_size: Tuple[int, int] = (1024, 1024), seed: Optional[int] = None):
        self.size = image_size
        self.params = SARParameters()
        self.rng = np.random.RandomState(seed)
        
    def generate_oil_spill_scenario(
        self,
        spill_centers: List[Tuple[int, int]],
        spill_radii: List[int],
        weather_condition: str = "moderate",
        oil_properties: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate complete scenario with metadata
        
        Args:
            spill_centers: List of (y, x) coordinates for spill centers
            spill_radii: List of radii for each spill (pixels)
            weather_condition: "calm", "moderate", "rough", "storm"
            oil_properties: Override default oil parameters
            
        Returns:
            sar_image: Synthetic SAR backscatter (linear scale)
            ground_truth: Binary mask (1 = oil, 0 = water)
            confidence_map: Pixel-wise confidence (0-1)
            metadata: Scenario parameters and statistics
        """
        # Update parameters if provided
        if oil_properties:
            for key, value in oil_properties.items():
                setattr(self.params, key, value)
        
        # Generate base ocean backscatter
        base_image = self._generate_ocean_backscatter(weather_condition)
        
        # Create ground truth mask
        ground_truth = self._create_ground_truth_mask(spill_centers, spill_radii)
        
        # Add oil spill signatures
        clean_sar = self._add_oil_signatures(base_image, ground_truth)
        
        # Add realistic sensor noise
        noisy_sar = self._add_sensor_noise(clean_sar)
        
        # Calculate uncertainty/confidence
        confidence_map = self._calculate_confidence_map(ground_truth, weather_condition)
        
        # Compile metadata
        metadata = self._compile_metadata(
            spill_centers, spill_radii, weather_condition, ground_truth
        )
        
        return noisy_sar, ground_truth, confidence_map, metadata
    
    def _generate_ocean_backscatter(self, weather: str) -> np.ndarray:
        """
        Generate realistic ocean backscatter using:
        - Bragg scattering model
        - Multi-scale fractal roughness
        - Wind speed modulation
        """
        # Wind speed mapping
        wind_speeds = {
            "calm": 2.0, "moderate": 5.0, 
            "rough": 10.0, "storm": 15.0
        }
        wind_speed = wind_speeds.get(weather, 5.0)
        self.params.wind_speed = wind_speed
        
        # Base backscatter from CMOD5 model (simplified)
        # Higher wind = higher backscatter
        base_sigma = 0.01 * (wind_speed ** 1.5)
        
        # Generate multi-scale fractal noise for ocean texture
        noise = np.zeros(self.size)
        for scale in [256, 128, 64, 32, 16, 8, 4]:
            amplitude = 1.0 / np.log2(scale)
            layer = self.rng.randn(*self.size)
            layer = ndimage.gaussian_filter(layer, sigma=scale/4)
            noise += amplitude * layer
        
        # Normalize and apply wind modulation
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        backscatter = base_sigma * (1 + 0.3 * noise)
        
        return np.clip(backscatter, 0.001, 0.5)
    
    def _create_ground_truth_mask(
        self, 
        centers: List[Tuple[int, int]], 
        radii: List[int]
    ) -> np.ndarray:
        """
        Create binary mask for oil spill regions
        """
        mask = np.zeros(self.size, dtype=np.uint8)
        
        for (cy, cx), radius in zip(centers, radii):
            y, x = np.ogrid[:self.size[0], :self.size[1]]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            
            # Create circular spill with irregular edges
            edge_noise = self.rng.randn(*self.size) * (radius * 0.1)
            irregular_dist = dist + edge_noise
            
            spill_region = irregular_dist < radius
            mask[spill_region] = 1
            
        return mask
    
    def _add_oil_signatures(
        self, 
        ocean_backscatter: np.ndarray, 
        oil_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply oil damping effect on backscatter
        Oil damps capillary waves → reduces backscatter
        Thicker oil = stronger damping
        """
        sar_image = ocean_backscatter.copy()
        
        # Calculate damping factor based on oil properties
        # Thicker oil = more damping (darker in SAR)
        thickness_factor = np.exp(-self.params.spill_thickness_mm / 5.0)
        damping_strength = 0.6 + 0.4 * thickness_factor  # 0.6 to 1.0
        
        # Weathering reduces damping over time
        weathering_factor = np.exp(-self.params.oil_age_hours / 48.0)
        effective_damping = 0.3 + 0.7 * damping_strength * weathering_factor
        
        # Apply damping to oil regions
        oil_pixels = oil_mask > 0
        sar_image[oil_pixels] *= (1 - effective_damping)
        
        # Add internal oil texture (less homogeneous than water)
        oil_texture = self.rng.randn(*self.size) * 0.05
        sar_image[oil_pixels] *= (1 + oil_texture[oil_pixels])
        
        return np.clip(sar_image, 0.0001, 1.0)
    
    def _add_sensor_noise(self, clean_sar: np.ndarray) -> np.ndarray:
        """
        Add realistic SAR sensor noise:
        1. Speckle noise (multiplicative, Gamma distribution)
        2. Thermal noise (additive)
        3. Quantization noise
        """
        # Speckle noise (multiplicative)
        # Gamma distribution for multi-look SAR
        looks = self.params.speckle_looks
        speckle = self.rng.gamma(looks, scale=1.0/looks, size=self.size)
        
        # Apply speckle
        with_speckle = clean_sar * speckle
        
        # Thermal noise (additive, Gaussian)
        thermal_linear = 10 ** (self.params.thermal_noise_db / 10.0)
        thermal_noise = self.rng.normal(0, thermal_linear * 0.1, size=self.size)
        
        # Add thermal noise
        with_thermal = with_speckle + thermal_noise
        
        # Ensure positive values
        return np.clip(with_thermal, 0.0001, 1.0)
    
    def _calculate_confidence_map(
        self, 
        oil_mask: np.ndarray, 
        weather: str
    ) -> np.ndarray:
        """
        Generate pixel-wise confidence scores based on:
        - Oil thickness (thinner = less certain)
        - Weather conditions (high wind = more noise = less certain)
        - Edge regions (boundaries are ambiguous)
        """
        confidence = np.ones(self.size, dtype=np.float32)
        
        # Weather uncertainty (higher wind = lower confidence)
        wind_uncertainty = {
            "calm": 0.95, "moderate": 0.85, 
            "rough": 0.70, "storm": 0.55
        }
        base_confidence = wind_uncertainty.get(weather, 0.85)
        confidence *= base_confidence
        
        # Thickness uncertainty in oil regions
        oil_pixels = oil_mask > 0
        thickness_conf = 0.5 + 0.5 * np.exp(-self.params.spill_thickness_mm / 3.0)
        confidence[oil_pixels] *= thickness_conf
        
        # Edge uncertainty (dilate and erode to find boundaries)
        from scipy import ndimage
        dilated = ndimage.binary_dilation(oil_mask, iterations=2)
        eroded = ndimage.binary_erosion(oil_mask, iterations=2)
        edges = dilated ^ eroded
        
        # Reduce confidence at edges
        confidence[edges] *= 0.7
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _compile_metadata(
        self,
        centers: List[Tuple[int, int]],
        radii: List[int],
        weather: str,
        ground_truth: np.ndarray
    ) -> Dict:
        """
        Compile scenario metadata for traceability
        """
        oil_area_pixels = np.sum(ground_truth)
        total_pixels = self.size[0] * self.size[1]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "image_size": self.size,
            "sensor_params": {
                "wavelength_m": self.params.wavelength,
                "incidence_angle_deg": self.params.incidence_angle,
                "resolution_m": self.params.resolution_m
            },
            "environmental_conditions": {
                "weather": weather,
                "wind_speed_ms": self.params.wind_speed,
                "wind_direction_deg": self.params.wind_direction,
                "wave_height_m": self.params.significant_wave_height
            },
            "oil_properties": {
                "viscosity_cst": self.params.oil_viscosity,
                "thickness_mm": self.params.spill_thickness_mm,
                "age_hours": self.params.oil_age_hours
            },
            "spills": [
                {
                    "center": {"y": cy, "x": cx},
                    "radius_pixels": r,
                    "estimated_area_m2": np.pi * (r * self.params.resolution_m) ** 2
                }
                for (cy, cx), r in zip(centers, radii)
            ],
            "statistics": {
                "oil_area_pixels": int(oil_area_pixels),
                "oil_coverage_percent": float(oil_area_pixels / total_pixels * 100),
                "total_area_m2": self.size[0] * self.size[1] * self.params.resolution_m ** 2
            }
        }
    
    def save_scenario(
        self,
        sar_image: np.ndarray,
        ground_truth: np.ndarray,
        confidence_map: np.ndarray,
        metadata: dict,
        output_dir: str,
        scenario_id: str
    ):
        """
        Save generated scenario to disk
        """
        import os
        import json
        import rasterio
        from rasterio.transform import Affine
        
        os.makedirs(output_dir, exist_ok=True)
        
        # FIXED: Create proper Affine transform
        pixel_size = float(self.params.resolution_m)
        transform = Affine.scale(pixel_size, pixel_size)
        
        # Save SAR image as GeoTIFF
        sar_path = os.path.join(output_dir, f"{scenario_id}_sar.tif")
        with rasterio.open(
            sar_path, 'w',
            driver='GTiff',
            height=sar_image.shape[0],
            width=sar_image.shape[1],
            count=1,
            dtype=sar_image.dtype,
            crs='EPSG:4326',
            transform=rasterio.Affine.scale(self.params.resolution_m, self.params.resolution_m)
        ) as dst:
            dst.write(sar_image, 1)
        
        # Save ground truth mask
        mask_path = os.path.join(output_dir, f"{scenario_id}_mask.tif")
        with rasterio.open(
            mask_path, 'w',
            driver='GTiff',
            height=ground_truth.shape[0],
            width=ground_truth.shape[1],
            count=1,
            dtype=ground_truth.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(ground_truth, 1)
        
        # Save confidence map
        conf_path = os.path.join(output_dir, f"{scenario_id}_confidence.tif")
        with rasterio.open(
            conf_path, 'w',
            driver='GTiff',
            height=confidence_map.shape[0],
            width=confidence_map.shape[1],
            count=1,
            dtype=confidence_map.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(confidence_map, 1)
        
        # Save metadata as JSON
        meta_path = os.path.join(output_dir, f"{scenario_id}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "sar_image": sar_path,
            "ground_truth": mask_path,
            "confidence": conf_path,
            "metadata": meta_path
        }

# Batch generation utility
def generate_training_dataset(
    n_samples: int,
    output_dir: str,
    image_size: Tuple[int, int] = (512, 512),
    n_spills_range: Tuple[int, int] = (1, 3),
    weather_conditions: List[str] = None,
    seed: int = 42
):
    """
    Generate a complete training dataset with multiple scenarios
    """
    if weather_conditions is None:
        weather_conditions = ["calm", "moderate", "rough"]
    
    simulator = RealisticSARSimulator(image_size=image_size, seed=seed)
    generated_files = []
    
    for i in range(n_samples):
        # Random scenario parameters
        n_spills = random.randint(*n_spills_range)
        weather = random.choice(weather_conditions)
        
        # Random spill locations and sizes
        centers = []
        radii = []
        for _ in range(n_spills):
            cy = random.randint(image_size[0]//4, 3*image_size[0]//4)
            cx = random.randint(image_size[1]//4, 3*image_size[1]//4)
            radius = random.randint(20, 80)
            centers.append((cy, cx))
            radii.append(radius)
        
        # Random oil properties
        oil_props = {
            "spill_thickness_mm": random.uniform(0.1, 5.0),
            "oil_age_hours": random.uniform(0, 72),
            "oil_viscosity": random.uniform(100, 1000)
        }
        
        # Generate scenario
        sar, mask, conf, meta = simulator.generate_oil_spill_scenario(
            spill_centers=centers,
            spill_radii=radii,
            weather_condition=weather,
            oil_properties=oil_props
        )
        
        # Save
        scenario_id = f"scenario_{i:05d}"
        files = simulator.save_scenario(
            sar, mask, conf, meta, 
            output_dir, scenario_id
        )
        generated_files.append(files)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples} scenarios")
    
    print(f"Dataset generation complete. Saved to {output_dir}")
    return generated_files


if __name__ == "__main__":
    # Example: Generate small test dataset
    generate_training_dataset(
        n_samples=10,
        output_dir="data/synthetic_training",
        image_size=(512, 512),
        n_spills_range=(1, 2),
        seed=42
    )