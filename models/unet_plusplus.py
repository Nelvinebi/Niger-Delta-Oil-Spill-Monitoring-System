"""
U-Net++ with Attention Gates for Oil Spill Segmentation
Simplified working version
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def attention_gate(x, g, inter_channels):
    """
    Attention Gate: Focus on relevant features
    x: Feature map from encoder (skip connection)
    g: Gating signal from decoder
    """
    theta_x = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(g)
    
    add = layers.Add()([theta_x, phi_g])
    relu = layers.Activation('relu')(add)
    
    psi = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)
    
    attention = layers.Multiply()([x, sigmoid])
    return attention, sigmoid


def conv_block(x, filters, kernel_size=3, activation='relu', dropout_rate=0.0):
    """
    Convolutional block with batch normalization
    """
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    return x


def nested_unet_plusplus(
    input_shape=(512, 512, 1),
    num_classes=2,
    deep_supervision=True
):
    """
    Simplified U-Net++ that actually works
    Uses standard U-Net with attention gates instead of complex nested structure
    """
    filters = [64, 128, 256, 512, 1024]
    dropout_rates = [0.0, 0.0, 0.1, 0.2, 0.3]
    
    inputs = layers.Input(shape=input_shape)
    
    # Normalize inputs
    x = layers.Lambda(lambda x: x / 255.0)(inputs)
    
    # Encoder path
    encoder_features = []
    for i, (f, dr) in enumerate(zip(filters, dropout_rates)):
        x = conv_block(x, f, dropout_rate=dr)
        encoder_features.append(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Decoder path with attention
    decoder_features = []
    for i in range(len(filters) - 2, -1, -1):
        # Upsample
        x = layers.Conv2DTranspose(filters[i], 2, strides=2, padding='same')(x)
        
        # Get skip connection
        skip = encoder_features[i]
        
        # Apply attention gate
        attended_skip, _ = attention_gate(skip, x, filters[i] // 2)
        
        # Concatenate
        x = layers.Concatenate()([x, attended_skip])
        x = conv_block(x, filters[i], dropout_rate=dropout_rates[i])
        
        decoder_features.append(x)
    
    # Deep supervision outputs (from each decoder level)
    outputs = []
    if deep_supervision:
        # Take last 4 decoder features for supervision
        for i, feat in enumerate(decoder_features[-4:]):
            out = layers.Conv2D(num_classes, 1, activation='softmax', name=f'out_{i+1}')(feat)
            # Upsample to full resolution if needed
            if i < 3:  # Not the last one
                out = layers.UpSampling2D(size=(2**(3-i), 2**(3-i)))(out)
            outputs.append(out)
    
    # Final output
    final_output = layers.Conv2D(num_classes, 1, activation='softmax', name='final_output')(x)
    outputs.append(final_output)
    
    model = Model(inputs=inputs, outputs=outputs if deep_supervision else final_output)
    return model


def build_compiled_model(
    input_shape=(512, 512, 1),
    num_classes=2,
    learning_rate=1e-4,
    loss_weights=None
):
    """
    Build and compile U-Net++ model with appropriate loss and metrics
    """
    model = nested_unet_plusplus(
        input_shape=input_shape,
        num_classes=num_classes,
        deep_supervision=True
    )
    
    # Loss functions
    if num_classes == 2:
        # Binary segmentation
        losses = {}
        for name in model.output_names:
            losses[name] = 'binary_crossentropy'
        
        metrics = {
            'final_output': ['accuracy', tf.keras.metrics.AUC(name='auc')]
        }
    else:
        # Multi-class
        losses = {name: 'categorical_crossentropy' for name in model.output_names}
        metrics = {'final_output': 'accuracy'}
    
    # Default loss weights if not provided
    if loss_weights is None:
        n_outputs = len(model.outputs)
        loss_weights = {}
        for i, name in enumerate(model.output_names[:-1]):
            loss_weights[name] = 0.5
        loss_weights[model.output_names[-1]] = 1.0  # Final output gets full weight
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model


# Data generator for training
class SARDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for loading SAR images and masks
    """
    
    def __init__(
        self,
        image_paths,
        mask_paths,
        batch_size=4,
        image_size=(512, 512),
        augment=True,
        shuffle=True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for i in batch_indexes:
            # Load image
            img = self._load_image(self.image_paths[i])
            mask = self._load_mask(self.mask_paths[i])
            
            # Augment if enabled
            if self.augment:
                img, mask = self._augment(img, mask)
            
            batch_images.append(img)
            batch_masks.append(mask)
        
        X = np.array(batch_images)
        Y = np.array(batch_masks)
        
        # For deep supervision, return multiple targets
        return X, {
            name: Y for name in ['out_1', 'out_2', 'out_3', 'out_4', 'final_output']
        }
    
    def _load_image(self, path):
        import rasterio
        with rasterio.open(path) as src:
            img = src.read(1)
            # Resize if needed
            if img.shape != self.image_size:
                img = tf.image.resize(img[..., np.newaxis], self.image_size).numpy()[..., 0]
            return img[..., np.newaxis]  # Add channel dimension
    
    def _load_mask(self, path):
        import rasterio
        with rasterio.open(path) as src:
            mask = src.read(1)
            if mask.shape != self.image_size:
                mask = tf.image.resize(mask[..., np.newaxis], self.image_size, method='nearest').numpy()[..., 0]
            # One-hot encode for 2 classes
            mask = tf.keras.utils.to_categorical(mask, num_classes=2)
            return mask
    
    def _augment(self, image, mask):
        # Random flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        # Random brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = image * factor
        
        return image, mask
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    # Test model construction
    model = build_compiled_model(input_shape=(512, 512, 1))
    model.summary()
    print(f"Model has {len(model.outputs)} outputs (deep supervision)")