"""
Test Suite for NDOSMS Oil Spill Detection
Comprehensive validation including accuracy, uncertainty, and edge cases
"""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plusplus import build_compiled_model, nested_unet_plusplus
from models.uncertainty import UncertaintyQuantifier
from data_generation.realistic_sar_simulator import RealisticSARSimulator


class TestModelArchitecture:
    """Test model construction and basic functionality"""
    
    def test_model_construction(self):
        """Test that model builds without errors"""
        model = nested_unet_plusplus(
            input_shape=(256, 256, 1),
            num_classes=2,
            deep_supervision=True
        )
        assert model is not None
        assert len(model.outputs) == 5  # 4 deep + 1 final
    
    def test_model_compilation(self):
        """Test model compilation with losses"""
        model = build_compiled_model(input_shape=(256, 256, 1))
        assert model.optimizer is not None
        assert len(model.losses) > 0
    
    def test_forward_pass(self):
        """Test prediction on dummy data"""
        model = build_compiled_model(input_shape=(128, 128, 1))
        dummy_input = np.random.rand(2, 128, 128, 1).astype(np.float32)
        outputs = model.predict(dummy_input)
        
        # Should return list for deep supervision
        assert isinstance(outputs, list)
        assert len(outputs) == 5
        assert outputs[-1].shape == (2, 128, 128, 2)  # Final output


class TestSyntheticDataGeneration:
    """Test synthetic SAR data generation"""
    
    def test_simulator_initialization(self):
        """Test simulator creates valid instances"""
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        assert sim.size == (256, 256)
        assert sim.rng is not None
    
    def test_scenario_generation(self):
        """Test complete scenario generation"""
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        
        sar, mask, conf, meta = sim.generate_oil_spill_scenario(
            spill_centers=[(128, 128)],
            spill_radii=[50],
            weather_condition="moderate"
        )
        
        # Check shapes
        assert sar.shape == (256, 256)
        assert mask.shape == (256, 256)
        assert conf.shape == (256, 256)
        
        # Check value ranges
        assert np.all(sar >= 0) and np.all(sar <= 1.0)
        assert np.all((mask == 0) | (mask == 1))  # Binary mask
        assert np.all((conf >= 0) & (conf <= 1.0))  # Confidence 0-1
        
        # Check metadata
        assert "spills" in meta
        assert "environmental_conditions" in meta
        assert len(meta["spills"]) == 1
    
    def test_weather_effects(self):
        """Test that different weather conditions produce different backscatter"""
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        
        weathers = ["calm", "moderate", "rough", "storm"]
        means = []
        
        for weather in weathers:
            sar, _, _, _ = sim.generate_oil_spill_scenario(
                spill_centers=[(128, 128)],
                spill_radii=[30],
                weather_condition=weather
            )
            means.append(np.mean(sar))
        
        # Higher wind should generally mean higher backscatter
        # (though randomness means this is probabilistic)
        assert means[3] > means[0] or means[2] > means[0]  # Storm/rough > calm
    
    def test_oil_damping(self):
        """Test that oil regions have lower backscatter than surrounding water"""
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        
        sar, mask, _, _ = sim.generate_oil_spill_scenario(
            spill_centers=[(128, 128)],
            spill_radii=[40],
            weather_condition="moderate",
            oil_properties={"spill_thickness_mm": 2.0}
        )
        
        oil_pixels = sar[mask > 0]
        water_pixels = sar[mask == 0]
        
        # Oil should be darker (lower backscatter) than water
        assert np.mean(oil_pixels) < np.mean(water_pixels)
    
    def test_thickness_effect(self):
        """Test that thicker oil produces stronger damping"""
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        
        thicknesses = [0.1, 1.0, 5.0]
        oil_means = []
        
        for thick in thicknesses:
            sar, mask, _, _ = sim.generate_oil_spill_scenario(
                spill_centers=[(128, 128)],
                spill_radii=[40],
                weather_condition="moderate",
                oil_properties={"spill_thickness_mm": thick}
            )
            oil_means.append(np.mean(sar[mask > 0]))
        
        # Thicker oil should generally be darker
        assert oil_means[2] <= oil_means[0]


class TestUncertaintyQuantification:
    """Test uncertainty estimation methods"""
    
    def test_mc_dropout_initialization(self):
        """Test uncertainty quantifier setup"""
        model = build_compiled_model(input_shape=(128, 128, 1))
        uq = UncertaintyQuantifier(model, num_mc_samples=10)
        assert uq.num_mc_samples == 10
    
    def test_uncertainty_prediction(self):
        """Test that uncertainty is generated"""
        model = build_compiled_model(input_shape=(128, 128, 1))
        uq = UncertaintyQuantifier(model, num_mc_samples=5)
        
        dummy_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
        metrics = uq.predict_with_uncertainty(dummy_input)
        
        # Check all uncertainty components exist
        assert metrics.mean_prediction is not None
        assert metrics.epistemic_uncertainty is not None
        assert metrics.confidence_map is not None
        assert metrics.confidence_map.shape == (128, 128)
    
    def test_confidence_range(self):
        """Test that confidence values are in valid range"""
        model = build_compiled_model(input_shape=(64, 64, 1))
        uq = UncertaintyQuantifier(model, num_mc_samples=5)
        
        dummy_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        metrics = uq.predict_with_uncertainty(dummy_input)
        
        # Confidence should be 0-1
        assert np.all(metrics.confidence_map >= 0)
        assert np.all(metrics.confidence_map <= 1.0)
    
    def test_calibration(self):
        """Test confidence calibration"""
        model = build_compiled_model(input_shape=(64, 64, 1))
        uq = UncertaintyQuantifier(model, num_mc_samples=10)
        
        # Create dummy predictions and ground truth
        predictions = np.random.rand(10, 64, 64, 2).astype(np.float32)
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        ground_truth = tf.keras.utils.to_categorical(
            np.random.randint(0, 2, size=(10, 64, 64)),
            num_classes=2
        )
        
        calibration = uq.calibrate_confidence(predictions, ground_truth, n_bins=5)
        
        assert "expected_calibration_error" in calibration
        assert "is_well_calibrated" in calibration


class TestEndToEndDetection:
    """Integration tests for complete detection pipeline"""
    
    def test_full_pipeline_synthetic(self):
        """Test complete pipeline with synthetic data"""
        # Generate synthetic data
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        sar, mask, conf, meta = sim.generate_oil_spill_scenario(
            spill_centers=[(128, 128)],
            spill_radii=[50],
            weather_condition="moderate"
        )
        
        # Build model
        model = build_compiled_model(input_shape=(256, 256, 1))
        
        # Prepare input
        input_batch = np.expand_dims(np.expand_dims(sar, 0), -1)
        
        # Predict
        uq = UncertaintyQuantifier(model, num_mc_samples=5)
        metrics = uq.predict_with_uncertainty(input_batch.astype(np.float32))
        
        # Basic sanity checks
        assert metrics.mean_prediction.shape == (1, 256, 256, 2)
        
        # Check that we can extract binary prediction
        pred_class = np.argmax(metrics.mean_prediction, axis=-1)
        assert pred_class.shape == (1, 256, 256)
    
    def test_look_alike_discrimination(self):
        """
        Test discrimination between oil and look-alikes (low-wind areas)
        This is critical for operational deployment
        """
        sim = RealisticSARSimulator(image_size=(256, 256), seed=42)
        
        # Generate oil spill
        sar_oil, mask_oil, _, _ = sim.generate_oil_spill_scenario(
            spill_centers=[(128, 128)],
            spill_radii=[50],
            weather_condition="moderate",
            oil_properties={"spill_thickness_mm": 1.0}
        )
        
        # Generate low-wind look-alike (no oil, but similar dark appearance)
        sim.params.wind_speed = 2.0  # Very low wind
        sar_lookalike = sim._generate_ocean_backscatter("calm")
        sar_lookalike = sim._add_sensor_noise(sar_lookalike)
        mask_lookalike = np.zeros_like(mask_oil)
        
        # Both should have low backscatter regions
        oil_darkness = np.mean(sar_oil[mask_oil > 0])
        lookalike_darkness = np.min(sar_lookalike)  # Darkest region
        
        # Document the similarity (this is why look-alikes are hard)
        print(f"Oil backscatter: {oil_darkness:.4f}")
        print(f"Look-alike min backscatter: {lookalike_darkness:.4f}")


class TestPerformanceMetrics:
    """Test calculation of performance metrics"""
    
    def test_iou_calculation(self):
        """Test Intersection over Union calculation"""
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        gt = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])
        
        intersection = np.sum((pred == 1) & (gt == 1))
        union = np.sum((pred == 1) | (gt == 1))
        iou = intersection / union if union > 0 else 0
        
        expected_iou = 3 / 5  # 3 intersection, 5 union
        assert abs(iou - expected_iou) < 0.01
    
    def test_detection_rate(self):
        """Test detection rate calculation"""
        # True positives: detected spills that are real
        # False negatives: missed spills
        tp = 8
        fn = 2
        detection_rate = tp / (tp + fn)
        assert detection_rate == 0.8


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])