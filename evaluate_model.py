import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. ARCHITECTURE (Must match the trained model exactly)
# ==========================================
class CopNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(CopNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.conv4(x, edge_index))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

# ==========================================
# 2. EVALUATION SCRIPT
# ==========================================
def evaluate():
    # Force CPU to avoid the RTX 5070 driver issue we hit earlier
    device = torch.device('cpu')
    print(f"--- EVALUATING ON: {device} ---")

    # 1. Load the exact same Test Set
    print("Loading dataset...")
    dataset = torch.load("compiled_cops_and_robbers_v3.pt", weights_only=False)
    
    random.seed(42) # CRITICAL: This ensures we get the exact same 20% test split
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    test_dataset = dataset[split_idx:]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Load the trained weights
    print("Loading trained CopNet weights...")
    model = CopNet(num_node_features=3, hidden_channels=64, num_classes=3).to(device)
    model.load_state_dict(torch.load("copnet_weights_v4.pth", map_location=device, weights_only=True))
    model.eval()

    # 3. Gather Predictions
    print(f"Running inference on {len(test_dataset):,} unseen graphs...\n")
    all_preds = []
    all_truths = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_truths.extend(batch.y.view(-1).cpu().numpy())

    # 4. Print the Text Report
    class_names = ["1 Cop", "2 Cops", "2+ Cops"]
    print("-" * 50)
    print("CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(all_truths, all_preds, target_names=class_names))

    # 5. Plot the Visual Confusion Matrix
    cm = confusion_matrix(all_truths, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('CopNet Confusion Matrix (Unseen Test Data)', fontsize=16, pad=15)
    plt.ylabel('Actual Mathematical Truth', fontsize=12, fontweight='bold')
    plt.xlabel('AI Prediction', fontsize=12, fontweight='bold')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()