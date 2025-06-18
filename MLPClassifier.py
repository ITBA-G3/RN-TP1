import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            # nn.BatchNorm1d(512), 
            # nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                # nn.init.uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)