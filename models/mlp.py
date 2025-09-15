import torch
import torch.nn as nn
from typing import List, Optional


class MLPBlock(nn.Module):
    def __init__(self, input: int, features: List[int]):
        """
        Initializes an MLPBlock, which represents a sequence of linear transformations
        and activations.

        Parameters:
        input (int): The number of inputs for the first linear layer.
        features (List[int]): A list of feature sizes for the layers.
        """
        super(MLPBlock, self).__init__()
        self.downs = nn.ModuleList()

        # Initialize linear layers with intermixed activations
        self.downs.append(nn.Linear(input, features[0]))
        self.downs.append(nn.LeakyReLU())

        for i in range(len(features) - 1):
            self.downs.append(nn.Linear(features[i], features[i + 1]))
            self.downs.append(nn.ReLU())

    def forward(
        self,
        input: torch.Tensor,
        _: Optional[torch.Tensor] = None,
        __: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass by sequentially passing the input tensor through
        the defined layers.

        Parameters:
        input (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The tensor resulting from processing through the layers.
        """
        output = input.flatten(start_dim=1)
        for down in self.downs:
            output = down(output)

        return output


class MLPLayers(nn.Module):
    def __init__(
        self, input: int, features: List[int], activation: nn.Module = nn.ReLU()
    ):
        """
        Initializes a series of MLP layers with specified activation.

        Parameters:
        input (int): The number of inputs for the first linear layer.
        features (List[int]): A list of feature sizes for the layers.
        activation (nn.Module): The activation function applied between layers. Default is ReLU.
        """
        super(MLPLayers, self).__init__()
        self.downs = nn.ModuleList()

        # Initialize layers and activations
        self.downs.append(nn.Linear(input, features[0]))

        for i in range(len(features) - 1):
            self.downs.append(activation)
            self.downs.append(nn.Linear(features[i], features[i + 1]))

    def forward(
        self,
        input,
        _: Optional[torch.Tensor] = None,
        __: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass by sequentially passing the input tensor through
        the defined layers.

        Parameters:
        input (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The tensor resulting from processing through the layers.
        """
        output = input.flatten(start_dim=1)
        for down in self.downs:
            output = down(output)

        return output


class MultiMLP(nn.Module):
    def __init__(
        self,
        num_marker: int = 41,
        interval_length: int = 8,
        pfe_dims: List[int] = [256, 512, 256, 64],
        fc_dims: List[int] = [256, 512, 256, 64],
        output_features: int = 1,
        marker_length: int = 3,
    ):
        """
        Initializes a MultiMLP model for extracting and classifying features
        from marker data.

        Parameters:
        num_marker (int): Number of markers. Default is 41.
        interval_length (int): Length of interval. Default is 8.
        pfe_dims (List[int]): Dimensions of the encoder. Default is [256, 512, 256, 64].
        fc_dims (List[int]): Dimensions of the classifier. Default is [256, 512, 256, 64].
        output_features (int): Number of output features. Default is 1.
        marker_length (int): Number of features per marker. Default is 3.
        """
        super(MultiMLP, self).__init__()
        self.posefeatureextractor = MLPBlock(num_marker * marker_length, pfe_dims)
        self.featureclassifier = MLPBlock(pfe_dims[-1] * interval_length, fc_dims)
        self.last_layer = nn.Linear(fc_dims[-1], output_features)

        self.network_type = "multi_mlp"
        self.interval_length = interval_length

    def forward(
        self,
        input,
        _: Optional[torch.Tensor] = None,
        __: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass to extract and classify features from the input tensor.

        Parameters:
        input (torch.Tensor): The input tensor. Expected format is (batch, interval, marker, dims).

        Returns:
        torch.Tensor: The final classification output.
        """
        batch, interval, marker, dims = input.shape
        input = input.view(
            batch * interval, marker, dims
        )  # Reshape tensor for processing
        posefeature = self.posefeatureextractor(input)  # Extract pose features
        posefeature = posefeature.view(batch, interval, -1)  # Reshape for classifier
        mergedfeature = self.featureclassifier(posefeature)  # Classify features
        output = self.last_layer(mergedfeature)  # Final layer output

        return output


class LRPMLP(nn.Module):
    def __init__(
        self,
        num_marker: int = 41,
        interval_length: int = 8,
        pfe_dims: List[int] = [256, 512, 256, 64],
        fc_dims: List[int] = [256, 512, 256, 64],
        output_features: int = 1,
        marker_length: int = 3,
    ):
        """
        Initializes a MultiMLP model for extracting and classifying features
        from marker data.

        Parameters:
        num_marker (int): Number of markers. Default is 41.
        interval_length (int): Length of interval. Default is 8.
        encoder_dims (List[int]): Dimensions of the encoder. Default is [256, 512, 256, 64].
        fc_dims (List[int]): Dimensions of the classifier. Default is [256, 512, 256, 64].
        output_features (int): Number of output features. Default is 1.
        marker_length (int): Number of features per marker. Default is 3.
        """
        super(LRPMLP, self).__init__()

        modules = [
            nn.Flatten(2),
            nn.Linear(num_marker * marker_length, pfe_dims[0]),
            nn.LeakyReLU(),
        ]  # PFE starts
        for layer in range(len(pfe_dims) - 1):
            modules.append(nn.Linear(pfe_dims[layer], pfe_dims[layer + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Flatten())  # Now the FC starts
        modules.append(nn.Linear(interval_length * pfe_dims[-1], fc_dims[0]))
        modules.append(nn.LeakyReLU())
        for layer in range(len(fc_dims) - 1):
            modules.append(nn.Linear(fc_dims[layer], fc_dims[layer + 1]))
            modules.append(nn.ReLU())
        modules.append(
            nn.Linear(fc_dims[-1], output_features)
        )  # Final classification layer

        self.features = nn.Sequential(*modules)

        self.network_type = "lrpmlp"
        self.interval_length = interval_length

    def forward(self, input):
        """
        Performs a forward pass to extract and classify features from the input tensor.

        Parameters:
        input (torch.Tensor): The input tensor. Expected format is (batch, interval, marker, dims).

        Returns:
        torch.Tensor: The final classification output.
        """

        return self.features(input)
