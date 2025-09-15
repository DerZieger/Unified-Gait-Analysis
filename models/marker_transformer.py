import torch
import torch.nn as nn
from typing import Optional
import models.posenc
import models.mlp


class ClassificationTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        embedding_dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
        inner_layer_dim: int = 2048,
    ):
        """
        Initializes a ClassificationTransformerEncoder with specified parameters.

        Parameters:
        input_dim (int): Dimension of input features.
        embedding_dim (int): Dimension of the embedding space.
        num_attention_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dropout_rate (float): Dropout rate.
        inner_layer_dim (int): Dimension of the feedforward inner layer.
        """
        super(ClassificationTransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=inner_layer_dim,
            dropout=dropout_rate,
            activation=nn.functional.gelu,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.input_embedding = nn.Linear(
            input_dim, embedding_dim
        )  # Linear layer for input embedding
        self.positional_encoder = models.posenc.PositionalEncoding(
            embedding_dim, 0.0
        )  # Positional encoding

        # Learnable classification token
        self.classification_token = nn.Parameter(torch.rand(1, 1, embedding_dim))

    def forward(
        self,
        input_tensor,
        position_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass to process input tensor and produce encoded output.

        Parameters:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        position_indices (Optional[torch.Tensor], optional): Optional positional indices. Default is None.
        padding_mask (Optional[torch.Tensor], optional): Optional padding mask. Default is None.

        Returns:
        torch.Tensor: Encoded output.
        """
        input_data = input_tensor
        if len(input_tensor.shape) > 3:  # Reshape if necessary
            input_data = input_tensor.view(
                -1, input_tensor.shape[2], input_tensor.shape[3]
            )
        embedded_input = self.input_embedding(input_data)  # Embed input
        batch_size, seq_len, _ = input_data.shape

        # Expand and concat classification token
        cls_tokens = self.classification_token.expand([batch_size, -1, -1])
        embedded_input = torch.cat([cls_tokens, embedded_input], dim=1)

        # Prepare positional encoding and mask
        if position_indices is None:
            position_indices = torch.arange(0, seq_len)
            position_indices = (
                position_indices.expand(input_tensor.shape[0], -1)
                .to(input_tensor.device)
                .long()
            )
        if padding_mask is None:
            padding_mask = torch.zeros(input_tensor.shape[0], seq_len)
            padding_mask = padding_mask.to(input_tensor.device).to(torch.bool)

        position_indices = position_indices + 1
        position_indices = torch.cat(
            (
                torch.zeros([input_tensor.shape[0], 1], dtype=torch.int64).to(
                    position_indices.device
                ),
                position_indices,
            ),
            dim=1,
        )
        position_encoding = self.positional_encoder.enc(position_indices).view(
            input_tensor.shape[0], -1, embedded_input.shape[-1]
        )
        position_encoding = position_encoding.repeat_interleave(
            batch_size // input_tensor.shape[0], dim=0
        )
        transformer_input = embedded_input + position_encoding

        # Mask setup
        mask = torch.hstack(
            [
                torch.zeros(padding_mask.shape[0], 1)
                .to(padding_mask.device)
                .to(torch.bool),
                padding_mask,
            ]
        )
        mask = mask.repeat_interleave(batch_size // input_tensor.shape[0], dim=0)
        transformer_output = self.transformer_encoder(transformer_input, None, mask)
        return transformer_output

    def get_attnmaps(
        self,
        input_tensor,
        position_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Computes attention maps for each layer of the transformer encoder.

        Parameters:
        input_tensor (torch.Tensor): Input tensor.
        position_indices (Optional[torch.Tensor], optional): Positional indices. Default is None.
        padding_mask (Optional[torch.Tensor], optional): Padding mask. Default is None.

        Returns:
        torch.Tensor: Attention maps for each encoder layer.
        """
        input_data = input_tensor
        if len(input_tensor.shape) > 3:
            input_data = input_tensor.view(
                -1, input_tensor.shape[2], input_tensor.shape[3]
            )
        embedded_input = self.input_embedding(input_data)
        batch_size, seq_len, _ = input_data.shape

        # Add cls token to inputs
        cls_tokens = self.classification_token.expand([batch_size, -1, -1])
        embedded_input = torch.cat([cls_tokens, embedded_input], dim=1)

        # Prepare positional encoding and mask
        if position_indices is None:
            position_indices = torch.arange(0, seq_len)
            position_indices = (
                position_indices.expand(input_tensor.shape[0], -1)
                .to(input_tensor.device)
                .long()
            )
        if padding_mask is None:
            padding_mask = torch.zeros(input_tensor.shape[0], seq_len)
            padding_mask = padding_mask.to(input_tensor.device).to(torch.bool)

        position_indices = position_indices + 1
        position_indices = torch.cat(
            (
                torch.zeros([input_tensor.shape[0], 1], dtype=torch.int64).to(
                    position_indices.device
                ),
                position_indices,
            ),
            dim=1,
        )
        position_encoding = self.positional_encoder.enc(position_indices).view(
            input_tensor.shape[0], -1, embedded_input.shape[-1]
        )
        position_encoding = position_encoding.repeat_interleave(
            batch_size // input_tensor.shape[0], dim=0
        )
        transformer_input = embedded_input + position_encoding

        # Mask setup
        mask = torch.hstack(
            [
                torch.zeros(padding_mask.shape[0], 1)
                .to(padding_mask.device)
                .to(torch.bool),
                padding_mask,
            ]
        )
        mask = mask.repeat_interleave(batch_size // input_tensor.shape[0], dim=0)

        # Extract attention maps
        norm_applied = False
        first_layer = True
        attention_maps = torch.zeros(0)
        for layer in self.transformer_encoder.layers:
            encoder_input = transformer_input.clone()
            if layer.norm_first or norm_applied:
                encoder_input = layer.norm1(encoder_input)
                norm_applied = True
            attention_weights = layer.self_attn(
                encoder_input,
                encoder_input,
                encoder_input,
                attn_mask=None,
                key_padding_mask=mask,
                need_weights=True,
            )[1]
            assert attention_weights is not None
            if first_layer:
                first_layer = False
                attention_maps = torch.unsqueeze(attention_weights, 0)
            else:
                attention_maps = torch.cat(
                    [attention_maps, torch.unsqueeze(attention_weights, 0)]
                )
            transformer_input = layer(
                transformer_input, src_mask=None, src_key_padding_mask=mask
            )
        return attention_maps


class MarkerClassification(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 2,
        pose_feature_dim: int = 64,
        feature_classifier_dim: int = 512,
        num_classifier_heads: int = 8,
        num_feature_heads: int = 4,
        interval_length: int = 128,
    ):
        """
        Initializes a MarkerClassification model, which processes marker sequences for classification.

        Parameters:
        input_dim (int): Dimension of input features for each marker.
        output_dim (int): Dimension of output classes.
        feature_embedding_dim (int): Dimension of the feature embedding space.
        merged_embedding_dim (int): Dimension of the merged embedding space.
        num_merging_heads (int): Number of heads for merging features.
        num_feature_heads (int): Number of heads for feature extraction.
        """
        super(MarkerClassification, self).__init__()
        self.feature_extractor = ClassificationTransformerEncoder(
            input_dim, pose_feature_dim, num_feature_heads, 1
        )
        self.feature_merger = ClassificationTransformerEncoder(
            pose_feature_dim, feature_classifier_dim, num_classifier_heads, 1
        )
        self.classifier = models.mlp.MLPLayers(
            feature_classifier_dim, [feature_classifier_dim, output_dim]
        )

        self.network_type = "transformer"
        self.interval_length = interval_length

    def forward(
        self,
        marker_sequences,
        position_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Processes input through the marker classification pipeline.

        Parameters:
        marker_sequences (torch.Tensor): The input marker sequences.
        position_indices (Optional[torch.Tensor], optional): Optional positional indices. Default is None.
        padding_mask (Optional[torch.Tensor], optional): Optional padding mask. Default is None.

        Returns:
        torch.Tensor: The classification result for each input sequence.
        """
        batch_shape = marker_sequences.shape
        extracted_features = self.feature_extractor(
            marker_sequences, position_indices, padding_mask
        )
        feature_view = extracted_features[:, 0, :].view(
            batch_shape[0], batch_shape[1], -1
        )

        merged_features = self.feature_merger(feature_view)
        return self.classifier(merged_features[:, 0, :])

    @torch.jit.export
    def get_attnmaps(
        self,
        marker_sequences,
        position_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Extracts and returns attention maps from the feature extraction and merging processes.

        Parameters:
        marker_sequences (torch.Tensor): The input marker sequences.
        position_indices (Optional[torch.Tensor], optional): Optional positional indices. Default is None.
        padding_mask (Optional[torch.Tensor], optional): Optional padding mask. Default is None.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Feature extraction and merging attention maps.
        """
        feature_attention_maps = self.feature_extractor.get_attnmaps(
            marker_sequences, position_indices, padding_mask
        )

        batch_shape = marker_sequences.shape
        extracted_features = self.feature_extractor(
            marker_sequences, position_indices, padding_mask
        )
        feature_view = extracted_features[:, 0, :].view(
            batch_shape[0], batch_shape[1], -1
        )
        merge_attention_maps = self.feature_merger.get_attnmaps(feature_view)

        return feature_attention_maps, merge_attention_maps
