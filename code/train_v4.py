import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We reuse the highly successful V3 dataset
DATASET_FILE = "compiled_cops_and_robbers_v3.pt"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 64

# ==========================================
# 2. MODEL ARCHITECTURE
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
# 3. TRAINING LOOP (V4 - Best Checkpointing)
# ==========================================
def train():
    device = torch.device('cpu')
    print(f"\n--- USING DEVICE: {device} ---")

    print(f"Loading '{DATASET_FILE}' into memory...")
    dataset = torch.load(DATASET_FILE, weights_only=False)

    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    counts = [0, 0, 0]
    for data in train_dataset:
        counts[data.y.item()] += 1
    total_train = sum(counts)
    weights = [total_train / (3 * c) if c > 0 else 1.0 for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # Note: num_node_features=3 to match our V3 data
    model = CopNet(num_node_features=3, hidden_channels=HIDDEN_CHANNELS, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nStarting Training on {len(train_dataset):,} graphs (Testing on {len(test_dataset):,})...")
    print("-" * 65)
    
    # --- NEW: Track the high score ---
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct_train = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct_train += int((pred == batch.y.view(-1)).sum())
            
        train_acc = correct_train / len(train_dataset)
        avg_loss = total_loss / len(train_dataset)

        model.eval()
        correct_test = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct_test += int((pred == batch.y.view(-1)).sum())
                
        test_acc = correct_test / len(test_dataset)

        # --- NEW: Save dynamically on high score ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "copnet_weights_v4.pth")

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:>3} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    print("-" * 65)
    print(f"Training Complete!")
    print(f"Peak Model captured at Epoch {best_epoch} with Test Accuracy: {best_test_acc:.4f}")
    print("Weights saved to 'copnet_weights_v4.pth'")

if __name__ == "__main__":
    train()