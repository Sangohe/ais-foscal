import ml_collections


def get_unet_config():
    config = ml_collections.ConfigDict()
    config.model_module = "vanilla"
    config.input_shape = None  # Set later after figuring out the modalities.
    config.filters_per_level = "32,64,128,256,512,1024"
    config.blocks_depth = "2,2,2,2,2,2"
    config.num_classes = 1
    config.activation = "relu"
    config.out_activation = "sigmoid"
    config.kernel_size = 3
    config.strides = 1
    config.dilation_rate = 1
    config.padding = "same"
    config.norm_layer = "tf.keras.layers.LayerNormalization"
    config.pooling_layer = "tf.keras.layers.AveragePooling2D"
    config.upsample_layer = "tf.keras.layers.UpSampling2D"
    config.attention_layer = "models.conv_layers.AdditiveCrossAttention"
    config.dropout_rate = 0.4
    return config


def get_convnext_unet_config():
    config = ml_collections.ConfigDict()
    config.model_module = "convnext"
    config.input_shape = None  # Set later after figuring out the modalities.
    config.filters_per_level = "64,128,256,512,1024,1024"
    config.blocks_depth = "2,2,2,2,2,2"
    config.num_classes = 1
    config.out_activation = "sigmoid"
    config.layer_scale_init_value = 1e-6
    config.drop_path_rate = 0.0
    config.upsample_layer = "tf.keras.layers.UpSampling2D"
    config.attention_layer = "models.conv_layers.AdditiveCrossAttention"
    config.dropout_rate = 0.3
    return config
