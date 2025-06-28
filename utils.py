import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import mlflow
import torchvision.utils as vutils
from sklearn.preprocessing import LabelEncoder

def get_class(x):
    return str(x.parent).split("/")[-1]

def param_counter(model):
    """
    Cuenta el número de parámetros entrenables en un modelo PyTorch.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Función para loguear una figura matplotlib en TensorBoard
def plot_to_tensorboard(fig, writer, tag, step):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1) / 255.0
    writer.add_image(tag, image, global_step=step)
    plt.close(fig)
    
# Función para matriz de confusión y clasificación
def log_classification_report(model, 
                              device, 
                              train_dataset, 
                              loader, 
                              writer, 
                              step, 
                              prefix="val"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.label_encoder.classes_)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title(f'{prefix.title()} - Confusion Matrix')

    # Guardar localmente y subir a MLflow
    fig_path = f"confusion_matrix_{prefix}_epoch_{step}.png"
    fig_cm.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)

    plot_to_tensorboard(fig_cm, writer, f"{prefix}/confusion_matrix", step)

    cls_report = classification_report(all_labels, all_preds, target_names=train_dataset.label_encoder.classes_)
    writer.add_text(f"{prefix}/classification_report", f"<pre>{cls_report}</pre>", step)

    # También loguear texto del reporte
    with open(f"classification_report_{prefix}_epoch_{step}.txt", "w") as f:
        f.write(cls_report)
    mlflow.log_artifact(f.name)
    os.remove(f.name)

# Entrenamiento y validación
def evaluate(model, loader, writer, device, train_dataset, criterion, epoch=None, prefix="val"):
    log_classification_report(model, device, train_dataset, loader, writer, step=epoch, prefix="val")
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss_sum += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Loguear imágenes del primer batch
            if i == 0 and epoch is not None:
                img_grid = vutils.make_grid(images[:8].cpu(), normalize=True)
                writer.add_image(f"{prefix}/images", img_grid, global_step=epoch)

    acc = 100.0 * correct / total
    avg_loss = loss_sum / len(loader)

    if epoch is not None:
        writer.add_scalar(f"{prefix}/loss", avg_loss, epoch)
        writer.add_scalar(f"{prefix}/accuracy", acc, epoch)

    return avg_loss, acc


class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3, padding = 1, padding_mode = "reflect"),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3, padding = 1, padding_mode = "reflect"),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear((input_size//4)**2*32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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

class CNNClassifier(nn.Module):
    def __init__(self, input_size, dropout = 0.0, num_classes=10, out_channels=16, kernel_size=3):
        super().__init__()
        
        # Calcular tamaño después de 2 convoluciones + 2 maxpools
        def conv_out(size, kernel_size, padding = 1, stride = 1):
            return (size + 2*padding - kernel_size) // stride + 1
        
        aux_size = input_size
        aux_size = conv_out(aux_size, kernel_size)
        aux_size = aux_size // 2  # MaxPool 1
        aux_size = conv_out(aux_size, kernel_size)
        aux_size = aux_size // 2  # MaxPool 2

        linear_size = 32*(aux_size**2)
        
        self.model = nn.Sequential(
            nn.Conv2d(3,out_channels,kernel_size, padding = 1, padding_mode = "reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(out_channels,32,kernel_size, padding = 1, padding_mode = "reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(linear_size, 128), #Linear size sale de calcular las pasadas sucesivas por las conv
            nn.ReLU(),                   #Tuvimos que hacerlo para poder cambiar el tamaño del kernel libremente
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
