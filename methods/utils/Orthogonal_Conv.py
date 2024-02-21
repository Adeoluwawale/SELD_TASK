import torch
import torch.nn as nn

# Define your custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# Function to perform SVD and split filters into two orthogonal sub-spaces
def split_filters_orthogonal(conv_layer):
    weight = conv_layer.weight
    in_channels, out_channels, kernel_size, _ = weight.size()

    # Reshape the weight tensor to perform SVD
    weight = weight.view(in_channels, out_channels * kernel_size * kernel_size)

    # Perform SVD
    U, S, V = torch.svd(weight, some=False)

    # Split into two orthogonal sub-spaces
    sub_space1 = U[:, :out_channels]
    sub_space2 = U[:, out_channels:]

    return sub_space1, sub_space2

# Initialize your CNN model
model = CustomCNN()

# Iterate through the blocks and apply SVD
for block in [model.block1, model.block2]:
    for layer in block:
        if isinstance(layer, nn.Conv2d):
            sub_space1, sub_space2 = split_filters_orthogonal(layer)

            # Create new convolutional layers with the decomposed weights
            new_layer1 = nn.Conv2d(layer.in_channels, sub_space1.size(1), kernel_size=layer.kernel_size, padding=layer.padding[0])
            new_layer2 = nn.Conv2d(layer.in_channels, sub_space2.size(1), kernel_size=layer.kernel_size, padding=layer.padding[0])

            # Initialize the new layers with the decomposed weights
            new_layer1.weight.data = sub_space1.view(new_layer1.weight.size())
            new_layer2.weight.data = sub_space2.view(new_layer2.weight.size())

            # Replace the old layer with the new layers
            layer = nn.Sequential(new_layer1, new_layer2)
            
# Print the updated model
print(model)
