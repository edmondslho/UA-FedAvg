import torch
import torch.nn as nn

#################################################################################
# This is the pytorch version developed base on the original Tensorflow official
# implmentation of the CNN netowrk proposed in Kulvicius et al., "Deep learning
# empowered sensor fusion boosts infant movement classification", Communications
# medicine, vol. 5, no. 1, p. 16, 2025.
#
# The best hyper-parameters reported by Kulvicius et al. were used, except
# adjusting the dropout rate.
#################################################################################

class InfantMovementCNN(nn.Module):

    def __init__(
        self,
        kn1=4,
        kw1=13,
        kn2=32,
        kw2=25,
        kn3=16,
        kw3=25,
        fc1=128,
        fc2=0,
        dropOutVal=0.3
    ):

        super().__init__()

        self.dropoutVal = dropOutVal#0.5#0.3#0.6#0.2

        layers = []

        # Conv1
        layers.extend([
            nn.Conv2d(
                in_channels=1,
                out_channels=kn1,
                kernel_size=(kw1, 1),
                padding=(kw1 // 2, 0)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(kn1),
            nn.Dropout(self.dropoutVal)
        ])

        in_ch = kn1

        # Conv2
        if kn2 > 0:
            layers.extend([
                nn.Conv2d(
                    in_ch,
                    kn2,
                    kernel_size=(kw2, 1),
                    padding=(kw2 // 2, 0)
                ),
                nn.ReLU(),
                nn.BatchNorm2d(kn2),
                nn.Dropout(self.dropoutVal)
            ])
            in_ch = kn2

        # Conv3
        if kn3 > 0:
            layers.extend([
                nn.Conv2d(
                    in_ch,
                    kn3,
                    kernel_size=(kw3, 1),
                    padding=(kw3 // 2, 0)
                ),
                nn.ReLU(),
                nn.BatchNorm2d(kn3),
                nn.Dropout(self.dropoutVal)
            ])
            in_ch = kn3

        self.conv_layers = nn.Sequential(*layers)

        # Equivalent to AveragePooling2D(pool_size=(250,1))
        self.global_pool = nn.AvgPool2d(
            kernel_size=(250, 1),
            stride=(250, 1)
        )

        self.fc1 = nn.Linear(in_ch * 60, fc1)

        self.bn_fc1 = nn.BatchNorm1d(fc1)   #   nn.LayerNorm(fc1)  #
        self.dropout_fc1 = nn.Dropout(self.dropoutVal)

        if fc2 > 0:
            self.fc2 = nn.Linear(fc1, fc2)
            self.bn_fc2 = nn.BatchNorm1d(fc2)   #       self.bn_fc2 = nn.LayerNorm(fc2)   #
            self.dropout_fc2 = nn.Dropout(self.dropoutVal)
            self.output = nn.Linear(fc2, 1)
        else:
            self.fc2 = None
            self.output = nn.Linear(fc1, 1)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.dropout_fc1(x)

        if self.fc2 is not None:
            x = torch.relu(self.fc2(x))
            x = self.bn_fc2(x)
            x = self.dropout_fc2(x)

        x = self.output(x)

        return x