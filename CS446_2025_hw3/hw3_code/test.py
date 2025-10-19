import torch
import matplotlib.pyplot as plt
from hw3_q2 import svm_solver, svm_predictor  # import your implementation
from hw3_utils import poly, rbf  # import kernel helpers if in a different file

# ==== 1. Create a simple dataset ====
x_pos = torch.randn(20, 2) + torch.tensor([2.0, 2.0])
x_neg = torch.randn(20, 2) + torch.tensor([-2.0, -2.0])
x_train = torch.cat([x_pos, x_neg], dim=0)
y_train = torch.cat([torch.ones(20), -torch.ones(20)])

# Shuffle dataset
perm = torch.randperm(x_train.shape[0])
x_train, y_train = x_train[perm], y_train[perm]

# ==== 2. Kernel settings to test ====
kernels = {
    "Linear (degree=1)": poly(1),
    "Polynomial (degree=3)": poly(3),
    "RBF (sigma=1.0)": rbf(1.0),
}

lr = 0.01
num_iters = 2000
C = 1.0

# ==== 3. Iterate through kernels ====
for name, kernel in kernels.items():
    print(f"\n=== Training with {name} kernel ===")
    alpha = svm_solver(x_train, y_train, lr, num_iters, kernel=kernel, c=C)
    print(f"Non-zero alphas: {(alpha > 1e-5).sum().item()}/{len(alpha)}")

    # Create test points
    x_test = torch.tensor([[2.0, 2.0], [-2.0, -2.0], [0.0, 0.0], [3.0, -3.0]])
    y_pred = svm_predictor(alpha, x_train, y_train, x_test, kernel=kernel)

    print("Test predictions:")
    for i, pred in enumerate(y_pred):
        print(
            f"x_test[{i}] = {x_test[i].tolist()} → predicted label: {int(pred.item())}"
        )

    # ==== Visualization ====
    plt.figure(figsize=(6, 6))
    plt.title(f"SVM Decision Boundary — {name}")

    # Plot training data
    plt.scatter(
        x_train[y_train == 1, 0], x_train[y_train == 1, 1], color="blue", label="+1"
    )
    plt.scatter(
        x_train[y_train == -1, 0], x_train[y_train == -1, 1], color="red", label="-1"
    )

    # Mark support vectors
    support_idx = alpha > 1e-5
    plt.scatter(
        x_train[support_idx, 0],
        x_train[support_idx, 1],
        s=120,
        facecolors="none",
        edgecolors="k",
        label="Support Vectors",
    )

    # Decision boundary
    xx, yy = torch.meshgrid(
        torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing="ij"
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    pred_grid = svm_predictor(alpha, x_train, y_train, grid, kernel=kernel)
    pred_grid = pred_grid.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        pred_grid,
        levels=[-float("inf"), 0, float("inf")],
        colors=["#ffcccc", "#ccccff"],
        alpha=0.5,
    )
    plt.legend()
    plt.show()
