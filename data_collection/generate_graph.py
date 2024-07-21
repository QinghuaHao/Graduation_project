import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os

def generate_circle_data(num_samples, radius=1, noise=0.1, num_points=100):
    data = []
    for _ in range(num_samples):
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta) + np.random.normal(0, noise, theta.shape)
        y = radius * np.sin(theta) + np.random.normal(0, noise, theta.shape)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        t = np.arange(num_points)
        data.append(np.column_stack((x, y, t)))
    return np.array(data)

def generate_square_data(num_samples, side_length=2, noise=0.1, num_points=100):
    data = []
    points_per_side = num_points // 4
    for _ in range(num_samples):
        x = np.linspace(-side_length / 2, side_length / 2, points_per_side)
        y = np.ones_like(x) * side_length / 2
        top = np.column_stack((x, y))
        bottom = np.column_stack((x, -y))
        x = np.ones_like(y) * side_length / 2
        y = np.linspace(-side_length / 2, side_length / 2, points_per_side)
        right = np.column_stack((x, y))
        left = np.column_stack((-x, y))
        square = np.vstack((top, right, bottom, left))
        square += np.random.normal(0, noise, square.shape)
        square[:, 0] = (square[:, 0] - np.min(square[:, 0])) / (np.max(square[:, 0]) - np.min(square[:, 0]))
        square[:, 1] = (square[:, 1] - np.min(square[:, 1])) / (np.max(square[:, 1]) - np.min(square[:, 1]))
        t = np.arange(num_points)
        data.append(np.column_stack((square, t)))
    return np.array(data)

def generate_triangle_data(num_samples, side_length=2, noise=0.1, num_points=99):
    data = []
    points_per_side = num_points // 3
    for _ in range(num_samples):
        x1 = np.linspace(-side_length / 2, side_length / 2, points_per_side)
        y1 = np.sqrt(3) / 2 * side_length - np.abs(x1)
        side1 = np.column_stack((x1, y1))

        x2 = np.linspace(side_length / 2, 0, points_per_side)
        y2 = np.linspace(0, -np.sqrt(3) / 2 * side_length, points_per_side)
        side2 = np.column_stack((x2, y2))

        x3 = np.linspace(0, -side_length / 2, num_points - 2 * points_per_side)
        y3 = np.linspace(-np.sqrt(3) / 2 * side_length, np.sqrt(3) / 2 * side_length, num_points - 2 * points_per_side)
        side3 = np.column_stack((x3, y3))

        triangle = np.vstack((side1, side2, side3))
        triangle += np.random.normal(0, noise, triangle.shape)
        triangle[:, 0] = (triangle[:, 0] - np.min(triangle[:, 0])) / (np.max(triangle[:, 0]) - np.min(triangle[:, 0]))
        triangle[:, 1] = (triangle[:, 1] - np.min(triangle[:, 1])) / (np.max(triangle[:, 1]) - np.min(triangle[:, 1]))
        t = np.arange(num_points)
        data.append(np.column_stack((triangle, t)))
    return np.array(data)

def generate_l_shape_data(num_samples, length=1, noise=0.1, num_points=100):
    data = []
    half_points = num_points // 2
    for _ in range(num_samples):
        x = np.concatenate([np.zeros(half_points), np.linspace(0, length, half_points)])
        y = np.concatenate([np.linspace(0, length, half_points), np.ones(half_points) * length])
        x += np.random.normal(0, noise, x.shape)
        y += np.random.normal(0, noise, y.shape)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        t = np.arange(num_points)
        data.append(np.column_stack((x, y, t)))
    return np.array(data)

def augment_data(data):
    augmented_data = []
    for trajectory in data:
        coords = trajectory[:, :2]
        t = trajectory[:, 2]
        
        # Random shift
        shift_x = np.random.uniform(-0.1, 0.1)
        shift_y = np.random.uniform(-0.1, 0.1)
        shifted = coords + np.array([shift_x, shift_y])
        
        # Random rotation
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = np.dot(shifted - np.mean(shifted, axis=0), rotation_matrix) + np.mean(shifted, axis=0)
        
        # Random scaling
        scale = np.random.uniform(0.5, 1.5)
        scaled = (rotated - np.mean(rotated, axis=0)) * scale + np.mean(rotated, axis=0)
        
        # Clip to [0, 1]
        scaled[:, 0] = np.clip(scaled[:, 0], 0, 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, 1)
        
        augmented_data.append(np.column_stack((scaled, t)))
    return np.array(augmented_data)


# 生成数据
num_points = 100
circle_data = generate_circle_data(500, num_points=num_points)
square_data = generate_square_data(500, num_points=num_points)
triangle_data = generate_triangle_data(500, num_points=num_points)
l_shape_data = generate_l_shape_data(500, num_points=num_points)

# 数据增强
augmented_circle_data = augment_data(circle_data)
augmented_square_data = augment_data(square_data)
augmented_triangle_data = augment_data(triangle_data)
augmented_l_shape_data = augment_data(l_shape_data)

# 创建标签
circle_labels = np.zeros(len(augmented_circle_data))
square_labels = np.ones(len(augmented_square_data))
triangle_labels = np.ones(len(augmented_triangle_data)) * 2
l_shape_labels = np.ones(len(augmented_l_shape_data)) * 3

# 合并数据
data = np.vstack((augmented_circle_data, augmented_square_data, augmented_triangle_data, augmented_l_shape_data))
labels = np.hstack((circle_labels, square_labels, triangle_labels, l_shape_labels))

# 随机打乱数据
data, labels = shuffle(data, labels)

# 保存数据
output_dir = '../data/standard'
os.makedirs(output_dir, exist_ok=True)

circle_df = pd.DataFrame(data[labels == 0].reshape(-1, 3), columns=['x', 'y', 'time'])
square_df = pd.DataFrame(data[labels == 1].reshape(-1, 3), columns=['x', 'y', 'time'])
triangle_df = pd.DataFrame(data[labels == 2].reshape(-1, 3), columns=['x', 'y', 'time'])
l_shape_df = pd.DataFrame(data[labels == 3].reshape(-1, 3), columns=['x', 'y', 'time'])

circle_df.to_csv('../data/standard/generated_circle.csv', index=False)
square_df.to_csv('../data/standard/generated_square.csv', index=False)
triangle_df.to_csv('../data/standard/generated_triangle.csv', index=False)
l_shape_df.to_csv('../data/standard/generated_l_shape.csv', index=False)

# 可视化数据
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.scatter(circle_df['x'], circle_df['y'], c=circle_df['time'], cmap='viridis', label='Circle')
plt.colorbar(label='Time')
plt.legend()
plt.subplot(2, 2, 2)
plt.scatter(square_df['x'], square_df['y'], c=square_df['time'], cmap='viridis', label='Square')
plt.colorbar(label='Time')
plt.legend()
plt.subplot(2, 2, 3)
plt.scatter(triangle_df['x'], triangle_df['y'], c=triangle_df['time'], cmap='viridis', label='Triangle')
plt.colorbar(label='Time')
plt.legend()
plt.subplot(2, 2, 4)
plt.scatter(l_shape_df['x'], l_shape_df['y'], c=l_shape_df['time'], cmap='viridis', label='L Shape')
plt.colorbar(label='Time')
plt.legend()
plt.show()
