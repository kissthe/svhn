import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm

def get_spectrum(matrix_2d_numpy):
    """计算2D矩阵的傅里叶变换频谱"""
    f_transform = np.fft.fft2(matrix_2d_numpy)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shifted)
    log_magnitude_spectrum = np.log(magnitude_spectrum + 1)
    return log_magnitude_spectrum

def decompose_weights_to_templates_for_layer(
    weight_tensor, name="", fixed_A_shape=None
):
    """
    对给定的权重张量进行 Kronecker 分解，得到模板 A 和 B。
    如果指定了 fixed_A_shape，则会尝试将权重矩阵分解为接近该形状的 A。
    """
    out_ch, in_ch, k_h, k_w = weight_tensor.shape
    target_shape_2d = (out_ch * k_h, in_ch * k_w)

    # 重塑为2D矩阵
    W_target = weight_tensor.permute(0, 2, 1, 3).reshape(target_shape_2d[0], target_shape_2d[1])
    print(f" - {name} 重塑后目标矩阵形状: {W_target.shape}")

    if fixed_A_shape is not None:
        p, q = fixed_A_shape
        print(f" - {name} 使用固定的 A 形状: {fixed_A_shape}")
        r = target_shape_2d[0] // p
        s = target_shape_2d[1] // q
        expected_kron_shape = (p * r, q * s)
        if expected_kron_shape != target_shape_2d:
            raise ValueError(f"固定的 A 形状 {fixed_A_shape} 导致 kron(A,B) 形状 {expected_kron_shape} 与目标形状 {target_shape_2d} 不匹配。")
    else:
        p, q = out_ch, in_ch
        r, s = k_h, k_w
        assert p * r == target_shape_2d[0] and q * s == target_shape_2d[1], f"维度不匹配！期望 {target_shape_2d}, 得到 {p*r}x{q*s}"

    print(f" - {name} 将分解为 A({p}x{q}) 和 B({r}x{s})")

    A = nn.Parameter(torch.randn(p, q) * 0.01, requires_grad=True)
    B = nn.Parameter(torch.randn(r, s) * 0.01, requires_grad=True)
    optimizer = optim.Adam([A, B], lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    weight_importance = torch.abs(W_target)
    weight_importance = weight_importance / torch.max(weight_importance)

    def weighted_mse_loss(W_pred, W_true, weights):
        diff = W_pred - W_true
        weighted_diff_sq = diff**2 * weights
        return torch.mean(weighted_diff_sq)

    print(f" - 开始对 {name} 进行 Kronecker 分解...")
    for epoch in range(2000):
        optimizer.zero_grad()
        W_approx = torch.kron(A, B)
        loss = weighted_mse_loss(W_approx, W_target, weight_importance)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A, B], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            rel_error = torch.norm(W_approx - W_target) / torch.norm(W_target)
            print(f" Epoch {epoch:4d} | Loss: {loss.item():.6e} | 相对误差: {rel_error:.4%}")

    with torch.no_grad():
        final_W_approx = torch.kron(A, B)
        final_rel_error = torch.norm(final_W_approx - W_target) / torch.norm(W_target)
        print(f" - {name} 最终相对重构误差: {final_rel_error:.4%}")

    return A.detach(), B.detach(), final_rel_error

class MatrixFeatureCNN(nn.Module):
    """
    用于从 A 矩阵预测性能的简单 CNN。
    输入: (batch_size, 1, A_p, A_q)
    输出: (batch_size, 1)
    """
    def __init__(self, input_height, input_width):
        super(MatrixFeatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Calculate size after first pooling
        pooled_h = input_height // 2
        pooled_w = input_width // 2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Calculate size after second pooling
        final_pooled_h = pooled_h // 2
        final_pooled_w = pooled_w // 2

        fc_input_dim = 64 * final_pooled_h * final_pooled_w
        if fc_input_dim <= 0:
             raise ValueError(f"Calculated fc_input_dim is non-positive: {fc_input_dim}. Check input dimensions or pooling layers.")
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension -> (B, 1, H, W)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten for FC layers
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1) # Remove last dimension to get scalar prediction per item

def train_cnn_model_with_matrix_features(A_list, targets, fixed_a_shape):
    """使用CNN训练模型，输入是 A 矩阵列表，输出是目标值"""
    if not A_list:
        raise ValueError("A_list 不能为空")
    A_stacked = torch.stack(A_list)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(A_stacked, targets_tensor, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    p, q = A_stacked.shape[1], A_stacked.shape[2]
    model = MatrixFeatureCNN(input_height=p, input_width=q)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    num_epochs = 200
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    print("\n--- 开始训练 CNN 模型 ---")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for A_batch, target_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(A_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for A_batch, target_batch in val_loader:
                outputs = model(A_batch)
                loss = criterion(outputs, target_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_cnn_model.pth'))
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train)
        val_preds = model(X_val)
        train_mse = criterion(train_preds, y_train).item()
        val_mse = criterion(val_preds, y_val).item()
        train_r2 = r2_score_pytorch(y_train, train_preds)
        val_r2 = r2_score_pytorch(y_val, val_preds)

    print("\n--- CNN 模型在第二层数据上的性能 ---")
    print(f"训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
    print(f"验证集 - MSE: {val_mse:.6f}, R²: {val_r2:.4f}")
    print(f"特征维度: {p} x {q} (来自 A 矩阵)")

    return model, fixed_a_shape

def r2_score_pytorch(y_true, y_pred):
    """PyTorch 版本的 R² 计算"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_on_layer_with_fixed_A(cnn_model, trained_a_shape, file_paths, layer_key, layer_shape_desc, performance_threshold):
    """在指定层上评估已训练的CNN模型，强制使用与训练时相同大小的 A 模板"""
    eval_A_matrices = []
    eval_targets = []
    print(f"\n--- 正在处理第三层 ({layer_shape_desc}) 以评估泛化能力 ---")
    print(f"--- 强制 A 的形状为与训练时一致: {trained_a_shape} ---")

    filtered_file_paths = [fp for fp in file_paths if filter_file_by_name(os.path.basename(fp))]
    
    for path in tqdm(filtered_file_paths, desc=f"Evaluating on {layer_key}"):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            val_acc = checkpoint.get('performance', {}).get('val_accuracy', -1)
            if val_acc < performance_threshold:
                continue

            weight_tensor = checkpoint['model_state_dict'][layer_key]
            A, B, _ = decompose_weights_to_templates_for_layer(
                weight_tensor,
                name=f"{os.path.basename(path)}_{layer_key}",
                fixed_A_shape=trained_a_shape
            )
            eval_A_matrices.append(A)
            eval_targets.append(val_acc)
        except Exception as e:
            print(f"处理文件 {path} 时出错: {e}. 跳过。")
            continue

    if not eval_A_matrices:
        print(f"没有找到满足条件的 {layer_key} 数据用于评估。")
        return

    A_stacked_eval = torch.stack(eval_A_matrices)
    eval_targets_tensor = torch.tensor(eval_targets, dtype=torch.float32)

    cnn_model.eval()
    with torch.no_grad():
        predictions_tensor = cnn_model(A_stacked_eval)
        predictions = predictions_tensor.numpy()
        eval_targets_np = eval_targets_tensor.numpy()

    mse_eval = np.mean((eval_targets_np - predictions) ** 2)
    r2_eval = r2_score_pytorch(eval_targets_tensor, predictions_tensor).item()

    print("\n--- CNN 模型在第三层数据上的泛化性能 ---")
    print(f"MSE: {mse_eval:.6f}, R²: {r2_eval:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(eval_targets_np, predictions, alpha=0.6)
    min_val = min(min(eval_targets_np), min(predictions))
    max_val = max(max(eval_targets_np), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Accuracy')
    plt.ylabel('Predicted Accuracy')
    plt.title(f'CNN Prediction vs True Accuracy\n(Layer: {layer_key}, $R^2$={r2_eval:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cnn_matrix_feature_prediction_{layer_key.replace('.', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


# --- 辅助函数：根据文件名判断是否包含在指定列表中 ---
def filter_file_by_name(filename):
    # 1. svhn_bad_data_issue_1001.pth 到 svhn_bad_data_issue_1400.pth
    if filename.startswith("svhn_bad_data_issue_"):
        try:
            num_str = filename.split('_')[4].split('.')[0] # e.g., "1001"
            num = int(num_str)
            return 1001 <= num <= 1400
        except (ValueError, IndexError):
            return False

    # 2. svhn_bad_undertrained_seed_1801_epoch_008/009/010 到 svhn_bad_undertrained_seed_2000_epoch_008/009/010
    if filename.startswith("svhn_bad_undertrained_seed_"):
        try:
            parts = filename.split('_')
            seed_num = int(parts[4]) # e.g., "1801"
            epoch_part = parts[6] # e.g., "008.pth"
            epoch_num = int(epoch_part.split('.')[0]) # e.g., "008"
            return 1801 <= seed_num <= 2000 and 8 <= epoch_num <= 10
        except (ValueError, IndexError):
            return False

    # 3. svhn_good_lr_variant_601.pth 到 svhn_good_lr_variant_800.pth
    if filename.startswith("svhn_good_lr_variant_"):
        try:
            num_str = filename.split('_')[4].split('.')[0] # e.g., "601"
            num = int(num_str)
            return 601 <= num <= 800
        except (ValueError, IndexError):
            return False

    # 4. svhn_good_snapshot_seed501_epoch007/008/009 到 svhn_good_snapshot_seed600_epoch007/008/009
    if filename.startswith("svhn_good_snapshot_seed"):
        try:
            seed_part = filename.split('_')[4] # e.g., "501"
            epoch_part = filename.split('_')[5] # e.g., "epoch007.pth"
            
            seed_num = int(seed_part)
            epoch_str = epoch_part.split('.')[0][5:] # remove 'epoch' prefix, e.g., "007"
            epoch_num = int(epoch_str)
            
            return 501 <= seed_num <= 600 and 7 <= epoch_num <= 9
        except (ValueError, IndexError):
            return False

    # 5. svhn_good_standard_001.pth 到 svhn_good_standard_500.pth
    if filename.startswith("svhn_good_standard_"):
        try:
            num_str = filename.split('_')[3].split('.')[0] # e.g., "001"
            num = int(num_str)
            return 1 <= num <= 500
        except (ValueError, IndexError):
            return False
            
    # 如果都不匹配，返回False
    return False


# --- 主流程 ---
# 1. 定义路径和过滤条件
print("开始训练！！！！！！！！！")
base_path = "/data/bowen/lintianze/svhn_weights_generation/weights/svhn"
# Use a more general pattern to catch all relevant files
file_pattern = os.path.join(base_path, "*.pth")

all_files = glob.glob(file_pattern)
print(f"Found {len(all_files)} total .pth files in {base_path}")

# Filter the list based on our specific criteria
filtered_all_files = [fp for fp in all_files if filter_file_by_name(os.path.basename(fp))]
print(f"Filtered to {len(filtered_all_files)} files matching specified patterns.")

# --- 调整性能阈值 ---
PERFORMANCE_THRESHOLD = 0.40  # 可根据需要调整
# --- 定义固定 A 的形状 ---
FIXED_A_SHAPE = (32, 32) # 可根据需要调整
# --- 定义用于训练的层和评估的层 ---
TRAINING_LAYER_KEY = 'conv2.weight' # 用于训练CNN的层
EVALUATION_LAYER_KEY = 'conv3.weight' # 用于评估泛化能力的层
EVALUATION_LAYER_SHAPE_DESC = '[64, 32, 3, 3]' # 评估层的描述


# 2. 加载满足条件的数据，提取 A 矩阵
train_A_matrices = []
train_targets = []

print(f"\n--- 正在处理训练层 ({TRAINING_LAYER_KEY}) 以训练 CNN 模型 (固定 A 形状为 {FIXED_A_SHAPE}) ---")
print(f"--- 当前性能阈值: {PERFORMANCE_THRESHOLD:.2%} ---")
print(f"--- 处理以下文件模式: svhn_bad_data_issue, svhn_bad_undertrained_seed, svhn_good_lr_variant, svhn_good_snapshot_seed, svhn_good_standard ---")

for path in tqdm(filtered_all_files, desc="Processing files for training"):
    try:
        checkpoint = torch.load(path, map_location='cpu')
        val_acc = checkpoint.get('performance', {}).get('val_accuracy', -1)
        
        if val_acc < PERFORMANCE_THRESHOLD:
            continue

        if TRAINING_LAYER_KEY not in checkpoint['model_state_dict']:
            print(f"Warning: {TRAINING_LAYER_KEY} not found in {path}. Skipping.")
            continue

        weight_tensor = checkpoint['model_state_dict'][TRAINING_LAYER_KEY]
        A, B, _ = decompose_weights_to_templates_for_layer(
            weight_tensor,
            name=os.path.basename(path),
            fixed_A_shape=FIXED_A_SHAPE
        )
        train_A_matrices.append(A)
        train_targets.append(val_acc)
    except Exception as e:
        print(f"处理文件 {path} 时出错: {e}. 跳过。")
        continue

if not train_A_matrices:
    raise ValueError("没有找到满足条件的训练数据。请检查路径和性能阈值。")

print(f"\n收集到 {len(train_A_matrices)} 个用于训练的 A 矩阵样本。")

# 3. 训练CNN模型
cnn_regressor, used_a_shape = train_cnn_model_with_matrix_features(train_A_matrices, train_targets, FIXED_A_SHAPE)

# 4. 在指定层上评估模型泛化能力
evaluate_on_layer_with_fixed_A(
    cnn_regressor,
    used_a_shape,
    filtered_all_files, # Pass the already filtered list
    EVALUATION_LAYER_KEY,
    EVALUATION_LAYER_SHAPE_DESC,
    PERFORMANCE_THRESHOLD
)

print("\n--- 流程结束 ---")