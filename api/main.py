"""
NDOSMS API - FastAPI Service for Oil Spill Detection
Production-ready REST API with uncertainty quantification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import tensorflow as tf
from io import BytesIO
import uuid
import json
import os
from datetime import datetime
import logging

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plusplus import build_compiled_model
from models.uncertainty import UncertaintyQuantifier
from data_generation.realistic_sar_simulator import RealisticSARSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NDOSMS API",
    description="Niger Delta Oil Spill Monitoring System - Operational API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once)
MODEL = None
UNCERTAINTY_QUANTIFIER = None


class DetectionRequest(BaseModel):
    """Request model for spill detection"""
    confidence_threshold: float = 0.75
    uncertainty_threshold: float = 0.5
    return_uncertainty_map: bool = True
    return_confidence_report: bool = True


class DetectionResponse(BaseModel):
    """Response model for spill detection"""
    detection_id: str
    timestamp: str
    status: str
    spill_detected: bool
    area_m2: float
    confidence_score: float
    uncertainty_level: str
    geojson: Dict
    metadata: Dict


class BatchDetectionRequest(BaseModel):
    """Batch processing request"""
    confidence_threshold: float = 0.75
    callback_url: Optional[str] = None


def load_model():
    """Load model at startup"""
    global MODEL, UNCERTAINTY_QUANTIFIER
    
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/ndosms_v2.h5")
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        MODEL = tf.keras.models.load_model(model_path)
    else:
        logger.warning("No trained model found, using compiled architecture")
        MODEL = build_compiled_model(input_shape=(512, 512, 1))
    
    UNCERTAINTY_QUANTIFIER = UncertaintyQuantifier(MODEL, num_mc_samples=30)
    logger.info("Model and uncertainty quantifier loaded")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_model()


def read_uploaded_image(file: UploadFile) -> tuple:
    """
    Read uploaded file as numpy array with geospatial metadata
    """
    try:
        contents = file.file.read()
        
        # Try to read as GeoTIFF
        with rasterio.open(BytesIO(contents)) as src:
            image = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # Normalize to model input size
            if image.shape != (512, 512):
                image = tf.image.resize(
                    image[..., np.newaxis], 
                    (512, 512)
                ).numpy()[..., 0]
            
            return image, {
                "transform": transform,
                "crs": str(crs) if crs else "EPSG:4326",
                "bounds": bounds,
                "original_shape": src.shape,
                "resolution": src.res
            }
            
    except Exception as e:
        # Try as regular image
        from PIL import Image
        img = Image.open(BytesIO(contents)).convert('L')
        img_array = np.array(img)
        
        if img_array.shape != (512, 512):
            img_array = tf.image.resize(
                img_array[..., np.newaxis],
                (512, 512)
            ).numpy()[..., 0]
        
        return img_array, {
            "transform": None,
            "crs": "EPSG:4326",
            "bounds": None,
            "original_shape": img_array.shape,
            "resolution": (10, 10)
        }


def vectorize_prediction(
    prediction_mask: np.ndarray,
    confidence_map: np.ndarray,
    geospatial_meta: Dict,
    threshold: float = 0.5
) -> Dict:
    """
    Convert prediction mask to GeoJSON with properties
    """
    from rasterio import features
    import shapely.geometry as geometry
    
    # Apply threshold
    binary_mask = (prediction_mask > threshold).astype(np.uint8)
    
    # Find connected components
    shapes = list(features.shapes(
        binary_mask,
        mask=binary_mask > 0,
        transform=geospatial_meta.get("transform")
    ))
    
    features_list = []
    for i, (shape, value) in enumerate(shapes):
        if value == 1:  # Oil spill class
            # Calculate mean confidence for this polygon
            geom = geometry.shape(shape)
            
            # Create feature
            feature = {
                "type": "Feature",
                "geometry": geometry.mapping(geom),
                "properties": {
                    "id": i,
                    "area_m2": geom.area * (geospatial_meta.get("resolution", (10, 10))[0] ** 2),
                    "mean_confidence": float(np.mean(confidence_map[binary_mask > 0])),
                    "detection_class": "oil_spill",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            features_list.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features_list,
        "total_spills": len(features_list),
        "total_area_m2": sum(f["properties"]["area_m2"] for f in features_list)
    }


@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "NDOSMS API",
        "version": "2.0.0",
        "status": "operational",
        "model_loaded": MODEL is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_status": "loaded" if MODEL else "not_loaded",
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_spill(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="SAR image file (GeoTIFF or PNG/JPG)"),
    confidence_threshold: float = Query(0.75, ge=0.0, le=1.0),
    return_uncertainty: bool = Query(True),
    return_visualization: bool = Query(False)
):
    """
    Detect oil spills in uploaded SAR image with uncertainty quantification
    """
    detection_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    try:
        # Read image
        logger.info(f"Processing detection {detection_id}")
        image, geo_meta = read_uploaded_image(file)
        
        # Prepare for model (add batch and channel dimensions)
        input_batch = np.expand_dims(image, 0)  # (1, H, W)
        input_batch = np.expand_dims(input_batch, -1)  # (1, H, W, 1)
        
        # Predict with uncertainty
        uncertainty_metrics = UNCERTAINTY_QUANTIFIER.predict_with_uncertainty(input_batch)
        
        # Get prediction and confidence
        prediction = uncertainty_metrics.mean_prediction[0]  # Remove batch dim
        confidence_map = uncertainty_metrics.confidence_map[0]
        
        # Apply adaptive thresholding
        binary_pred = UNCERTAINTY_QUANTIFIER.threshold_with_uncertainty(
            prediction,
            uncertainty_metrics.total_uncertainty[0],
            base_threshold=confidence_threshold
        )
        
        # Check if spill detected
        spill_detected = np.sum(binary_pred) > 100  # Minimum 100 pixels
        
        # Vectorize results
        geojson = vectorize_prediction(
            prediction[..., 1],  # Oil class probability
            confidence_map,
            geo_meta,
            threshold=confidence_threshold
        )
        
        # Calculate overall metrics
        mean_confidence = float(np.mean(confidence_map))
        uncertainty_level = (
            "low" if mean_confidence > 0.8 
            else "medium" if mean_confidence > 0.6 
            else "high"
        )
        
        # Save results if needed
        output_dir = f"outputs/{detection_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction raster
        pred_path = os.path.join(output_dir, "prediction.tif")
        with rasterio.open(
            pred_path, 'w',
            driver='GTiff',
            height=512,
            width=512,
            count=1,
            dtype=prediction[..., 1].dtype,
            crs=geo_meta.get("crs", "EPSG:4326"),
            transform=geo_meta.get("transform") or rasterio.Affine.identity()
        ) as dst:
            dst.write(prediction[..., 1], 1)
        
        # Generate response
        response = DetectionResponse(
            detection_id=detection_id,
            timestamp=timestamp,
            status="success",
            spill_detected=spill_detected,
            area_m2=geojson["total_area_m2"],
            confidence_score=mean_confidence,
            uncertainty_level=uncertainty_level,
            geojson=geojson,
            metadata={
                "image_shape": image.shape,
                "processing_time_ms": 0,  # TODO: Add timing
                "model_version": "2.0",
                "threshold_applied": confidence_threshold,
                "uncertainty_stats": {
                    "mean_epistemic": float(np.mean(uncertainty_metrics.epistemic_uncertainty)),
                    "mean_aleatoric": float(np.mean(uncertainty_metrics.aleatoric_uncertainty))
                }
            }
        )
        
        # Save full report
        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, 'w') as f:
            json.dump(response.dict(), f, indent=2)
        
        return response
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-detect")
async def batch_detect(
    files: List[UploadFile] = File(...),
    request: BatchDetectionRequest = None
):
    """
    Process multiple images in batch
    """
    results = []
    
    for file in files:
        try:
            # Process each file
            result = await detect_spill(
                background_tasks=None,
                file=file,
                confidence_threshold=request.confidence_threshold if request else 0.75
            )
            results.append({
                "filename": file.filename,
                "status": "success",
                "detection_id": result.detection_id
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "batch_id": str(uuid.uuid4()),
        "processed": len(results),
        "results": results
    }


@app.get("/download/{detection_id}/{file_type}")
async def download_result(detection_id: str, file_type: str):
    """
    Download detection results (prediction, uncertainty map, or report)
    """
    file_paths = {
        "prediction": f"outputs/{detection_id}/prediction.tif",
        "report": f"outputs/{detection_id}/report.json",
        "uncertainty": f"outputs/{detection_id}/uncertainty.tif"
    }
    
    if file_type not in file_paths:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = file_paths[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.post("/generate-synthetic")
async def generate_synthetic_scenario(
    n_spills: int = Query(1, ge=1, le=5),
    weather: str = Query("moderate", enum=["calm", "moderate", "rough", "storm"]),
    thickness_mm: float = Query(0.5, ge=0.1, le=10.0)
):
    """
    Generate synthetic SAR scenario for testing
    """
    simulator = RealisticSARSimulator(image_size=(512, 512))
    
    # Random spill locations
    centers = []
    radii = []
    for _ in range(n_spills):
        cy = np.random.randint(128, 384)
        cx = np.random.randint(128, 384)
        radius = np.random.randint(30, 60)
        centers.append((cy, cx))
        radii.append(radius)
    
    # Generate
    sar, mask, conf, meta = simulator.generate_oil_spill_scenario(
        spill_centers=centers,
        spill_radii=radii,
        weather_condition=weather,
        oil_properties={"spill_thickness_mm": thickness_mm}
    )
    
    # Save and return
    scenario_id = str(uuid.uuid4())
    output_dir = f"outputs/synthetic/{scenario_id}"
    files = simulator.save_scenario(sar, mask, conf, meta, output_dir, "scenario")
    
    return {
        "scenario_id": scenario_id,
        "files": files,
        "metadata": meta
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)