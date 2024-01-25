# Perceptron Learning Algorithm with Synthetic Dataset

#### 1. Synthetic Data Generation:

```python
from sklearn.datasets import make_classification
import numpy as np

# Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=15)
```

#### 2. Data Visualization:

```python
import matplotlib.pyplot as plt

# Visualize the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.show()
```

#### 3. Perceptron Implementation:

```python
def perceptron(X, y):
    w1 = w2 = b = 1
    lr = 0.1

    for j in range(1000):
        for i in range(X.shape[0]):
            z = w1 * X[i][0] + w2 * X[i][1] + b

            # Update weights and bias using the perceptron update rule
            if z * y[i] < 0:
                w1 = w1 + lr * y[i] * X[i][0]
                w2 = w2 + lr * y[i] * X[i][1]
                b = b + lr * y[i]

    return w1, w2, b
```

#### 4. Perceptron Working:

- **Initialization:**
  - Initialize weights (`w1` and `w2`) and bias (`b`) to 1.
  - Set learning rate (`lr`) to 0.1.

- **Perceptron Update Rule:**
  - For each training iteration:
    - Calculate the weighted sum \(z = w1 \cdot X[i][0] + w2 \cdot X[i][1] + b\).
    - If \(z \times y[i] < 0\) (misclassification), update weights and bias:
      - \(w1 = w1 + \text{{lr}} \times y[i] \times X[i][0]\)
      - \(w2 = w2 + \text{{lr}} \times y[i] \times X[i][1]\)
      - \(b = b + \text{{lr}} \times y[i]\)

#### 5. Decision Boundary Visualization:

```python
# Extract slope (m) and intercept (c) of the decision boundary
m = -(w1 / w2)
c = -(b / w2)

# Plot the decision boundary
x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + c

plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.show()
```

#### Notes:

- The perceptron aims to find a decision boundary that separates classes using a linear function \(z = w1 \cdot X[i][0] + w2 \cdot X[i][1] + b\).
- Misclassified points trigger updates to weights and bias, facilitating convergence.
- The decision boundary is visualized by plotting the line \(y = mx + c\).

By iteratively adjusting weights and bias based on misclassifications, the perceptron converges to a decision boundary that accurately separates the two classes in the synthetic dataset.

---
