"""
Uncertainty Quantification for Oil Spill Detection
Monte Carlo Dropout and ensemble methods for confidence estimation
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty metrics"""
    mean_prediction: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    confidence_map: np.ndarray
    coefficient_of_variation: np.ndarray


class UncertaintyQuantifier:
    """
    Monte Carlo Dropout for uncertainty estimation in oil spill detection
    
    Epistemic uncertainty: Model uncertainty (reducible with more data)
    Aleatoric uncertainty: Data noise (inherent in SAR imagery)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        num_mc_samples: int = 30,
        dropout_rate: float = 0.1
    ):
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        
    def predict_with_uncertainty(
        self,
        image_batch: np.ndarray,
        return_individual_samples: bool = False
    ) -> UncertaintyMetrics:
        """
        Generate predictions with uncertainty quantification using MC Dropout
        
        Args:
            image_batch: Input SAR images (N, H, W, C)
            return_individual_samples: If True, return all MC samples
            
        Returns:
            UncertaintyMetrics containing various uncertainty measures
        """
        # Enable dropout at inference time
        predictions = []
        
        for _ in range(self.num_mc_samples):
            # Training=True keeps dropout active during inference
            pred = self.model(image_batch, training=True)
            
            # Handle deep supervision (take final output)
            if isinstance(pred, list):
                pred = pred[-1]
            
            predictions.append(pred.numpy())
        
        # Stack predictions: (num_samples, batch, H, W, classes)
        stacked_preds = np.stack(predictions, axis=0)
        
        # Calculate statistics
        mean_pred = np.mean(stacked_preds, axis=0)
        variance_pred = np.var(stacked_preds, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = variance_pred
        
        # Estimate aleatoric uncertainty (data noise)
        # Using predictive entropy as proxy
        epsilon = 1e-8
        predictive_entropy = -np.sum(
            mean_pred * np.log(mean_pred + epsilon), 
            axis=-1, 
            keepdims=True
        )
        aleatoric = predictive_entropy / np.log(2)  # Normalize to [0, 1]
        
        # Total uncertainty
        total_unc = epistemic + aleatoric[..., np.newaxis]
        
        # Confidence map (inverse of uncertainty, normalized)
        max_unc = np.max(total_unc, axis=-1, keepdims=True) + epsilon
        confidence = 1 - (total_unc / max_unc)
        confidence_map = np.mean(confidence, axis=-1)
        
        # Coefficient of variation
        cv = np.std(stacked_preds, axis=0) / (mean_pred + epsilon)
        
        metrics = UncertaintyMetrics(
            mean_prediction=mean_pred,
            epistemic_uncertainty=np.mean(epistemic, axis=-1),
            aleatoric_uncertainty=aleatoric[..., 0],
            total_uncertainty=np.mean(total_unc, axis=-1),
            confidence_map=confidence_map,
            coefficient_of_variation=np.mean(cv, axis=-1)
        )
        
        if return_individual_samples:
            metrics.individual_samples = stacked_preds
        
        return metrics
    
    def calibrate_confidence(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Calibrate confidence scores using reliability diagram
        
        Measures how well confidence matches actual accuracy
        
        Args:
            predictions: Model predictions (N, H, W, classes)
            ground_truth: One-hot encoded ground truth
            n_bins: Number of confidence bins
            
        Returns:
            Dictionary with calibration metrics
        """
        # Flatten spatial dimensions
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
        
        # Get confidence (max probability) and predicted class
        confidences = np.max(pred_flat, axis=1)
        predicted_classes = np.argmax(pred_flat, axis=1)
        true_classes = np.argmax(gt_flat, axis=1)
        
        # Accuracy per bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            bin_count = np.sum(in_bin)
            
            if bin_count > 0:
                accuracy = np.mean(predicted_classes[in_bin] == true_classes[in_bin])
                avg_confidence = np.mean(confidences[in_bin])
                
                bin_accuracies.append(accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0)
                bin_confidences.append((lower + upper) / 2)
                bin_counts.append(0)
        
        # Expected Calibration Error (ECE)
        bin_weights = np.array(bin_counts) / np.sum(bin_counts)
        ece = np.sum(bin_weights * np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        
        return {
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "bin_edges": bin_edges.tolist(),
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "is_well_calibrated": ece < 0.05,
            "reliability_diagram": {
                "perfect": [0, 1],
                "actual": [bin_confidences, bin_accuracies]
            }
        }
    
    def threshold_with_uncertainty(
        self,
        predictions: np.ndarray,
        uncertainty_map: np.ndarray,
        base_threshold: float = 0.5,
        uncertainty_factor: float = 0.1
    ) -> np.ndarray:
        """
        Adaptive thresholding based on uncertainty
        
        Higher uncertainty = higher threshold (more conservative)
        """
        # Adjust threshold based on uncertainty
        # High uncertainty regions need higher confidence to classify as oil
        adjusted_threshold = base_threshold + (uncertainty_factor * uncertainty_map)
        
        # Apply threshold
        binary_predictions = (predictions[..., 1] > adjusted_threshold).astype(np.uint8)
        
        return binary_predictions
    
    def generate_uncertainty_report(
        self,
        metrics: UncertaintyMetrics,
        output_path: str = None
    ) -> Dict:
        """
        Generate human-readable uncertainty report
        """
        report = {
            "summary_statistics": {
                "mean_confidence": float(np.mean(metrics.confidence_map)),
                "std_confidence": float(np.std(metrics.confidence_map)),
                "min_confidence": float(np.min(metrics.confidence_map)),
                "max_confidence": float(np.max(metrics.confidence_map)),
                "high_uncertainty_pixels": int(np.sum(metrics.total_uncertainty > 0.5)),
                "low_confidence_regions": int(np.sum(metrics.confidence_map < 0.5))
            },
            "uncertainty_breakdown": {
                "mean_epistemic": float(np.mean(metrics.epistemic_uncertainty)),
                "mean_aleatoric": float(np.mean(metrics.aleatoric_uncertainty)),
                "epistemic_ratio": float(
                    np.mean(metrics.epistemic_uncertainty) / 
                    (np.mean(metrics.total_uncertainty) + 1e-8)
                )
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if report["summary_statistics"]["mean_confidence"] < 0.7:
            report["recommendations"].append(
                "Low average confidence. Consider collecting more training data or "
                "reducing scene complexity."
            )
        
        if report["uncertainty_breakdown"]["epistemic_ratio"] > 0.6:
            report["recommendations"].append(
                "High model uncertainty (epistemic). Model needs more training data "
                "or architecture improvements."
            )
        
        if report["summary_statistics"]["high_uncertainty_pixels"] > 1000:
            report["recommendations"].append(
                "Significant high-uncertainty regions detected. Manual review recommended "
                "for these areas."
            )
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


class EnsembleUncertainty:
    """
    Uncertainty quantification using model ensemble
    Alternative to MC Dropout - trains multiple models
    """
    
    def __init__(self, models: List[tf.keras.Model]):
        self.models = models
        
    def predict(self, image_batch: np.ndarray) -> UncertaintyMetrics:
        """
        Generate ensemble predictions
        """
        predictions = [model.predict(image_batch) for model in self.models]
        
        if isinstance(predictions[0], list):
            predictions = [p[-1] for p in predictions]
        
        stacked = np.stack(predictions, axis=0)
        
        mean_pred = np.mean(stacked, axis=0)
        uncertainty = np.var(stacked, axis=0)
        
        # Ensemble disagreement as uncertainty measure
        predicted_classes = [np.argmax(p, axis=-1) for p in predictions]
        disagreement = np.std(predicted_classes, axis=0)
        
        return UncertaintyMetrics(
            mean_prediction=mean_pred,
            epistemic_uncertainty=uncertainty.mean(axis=-1),
            aleatoric_uncertainty=np.zeros_like(uncertainty.mean(axis=-1)),
            total_uncertainty=uncertainty.mean(axis=-1),
            confidence_map=1 - (disagreement / len(self.models)),
            coefficient_of_variation=uncertainty.mean(axis=-1) / (mean_pred.mean(axis=-1) + 1e-8)
        )


if __name__ == "__main__":
    # Test uncertainty quantification
    print("UncertaintyQuantifier module ready")
    print("Use with trained model for MC Dropout inference")