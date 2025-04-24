import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from matplotlib import cm
from scipy.interpolate import griddata
import pandas as pd
import plotly.graph_objs as go


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 激活函数（Sigmoid和ReLU）及其导数
def relu(x):
    """
    ReLU激活函数
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    ReLU激活函数的导数
    """
    return np.where(x > 0, 1, 0)


# 读取数据
def load_data(filename):
    """
    从CSV文件中加载数据
    :param filename: 文件路径
    :return: 特征矩阵X和目标变量y
    """
    data = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        header = next(reader)  # 跳过表头
        for row in reader:
            data.append([float(x) for x in row])
    data = np.array(data)
    X = data[:, :-1]  # 所有特征：经度、纬度、房龄、房主收入
    Y = data[:, -1].reshape(-1, 1)  # 目标：房价
    return X, Y

# 矩阵标准化
def standardScaler(InputData):
    """
    标准化特征矩阵
    :param InputData: 特征矩阵
    :return: 均值，标准差，标准化后的特征矩阵
    """
    mean = np.mean(InputData, axis=0)  # 均值
    std = np.std(InputData, axis=0)  # 标准差
    return (mean, std, (InputData - mean) / std)

# 反标准化
def inverseStandardScaler(InputData, mean, std):
    """
    反标准化特征矩阵
    :param InputData: 特征矩阵
    :param mean: 均值
    :param std: 标准差
    :return: 反标准化后的特征矩阵
    """
    return InputData * std + mean

# 划分训练集和测试集
def trainTestSplit(X, Y, test_size=0.2, random_state=None):
    """
    划分训练集和测试集
    :param X: 特征矩阵
    :param Y: 目标变量
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: X_train, X_test, Y_train, Y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 获取样本数量
    n_samples = X.shape[0]
    
    # 计算测试集样本数
    n_test = int(n_samples * test_size)
    
    # 生成随机索引
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 划分数据集
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    
    return X_train, X_test, Y_train, Y_test

# mini-batch函数
def create_mini_batches(X, Y, batch_size):
    """
    mini-batch
    :param X: 特征矩阵
    :param Y: 目标变量
    :param batch_size: batch大小
    :return: mini-batch列表，每个元素为(X_batch, Y_batch)
    """
    mini_batches = []
    n_samples = X.shape[0]
    
    # 洗牌
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    # 创建mini-batches
    num_complete_batches = n_samples // batch_size
    for i in range(num_complete_batches):
        X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    # 处理剩余样本
    if n_samples % batch_size != 0:
        X_batch = X_shuffled[num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches

# 多层感知机（MLP）类
class MLP:
    def __init__(self, layers:list, activation='relu', learning_rate=0.01, max_iterations=1000, batch_size=32):
        """
        初始化多层感知机
        :param layers: 各层神经元数量
        :param activation: 激活函数类型（'sigmoid'或'relu'）
        :param activation_derivative: 激活函数导数
        :param learning_rate: 学习率
        :param max_iterations: 最大迭代次数
        :param batch_size: mini-batch大小
        """
        self.layers = layers
        self.activation = relu
        self.activation_derivative = relu_derivative
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size

        self.weights = {}  # 权重矩阵字典
        self.biases = {} # 偏置矩阵字典
        self.losses = []  # 损失列表
        

        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        初始化权重和偏置
        He初始
        """
        for i in range(1, len(self.layers)):
            # 权重矩阵初始化为随机值，偏置初始化为零
            self.weights[i] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2 / self.layers[i-1])
            self.biases[i] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 输出out和缓存cache
        """
        cache = {}  # 缓存字典，记录每层的权和和激活值
        out = X.T # X原来是[8000:4]，转置后是[4:8000]

        for i in range(1, len(self.layers)):
            net = np.dot(self.weights[i], out) + self.biases[i]
            cache['net' + str(i)] = net  # 记录每层的权和

            # 排除输出层
            if i == len(self.layers) - 1:
                out = net
            else:
                out = self.activation(net)
            cache['out' + str(i)] = out
    
        return out, cache # 返回的out：[1:8000]
    
    def backward(self, X, Y, cache):
        """
        反向传播
        :param X: 输入数据
        :param Y: 目标值
        :param cache: 前向传播的缓存
        :return: 梯度
        """
        results = {}
        L = len(self.layers)
        m = X.shape[0] # 样本数量
        
        # 输出层梯度
        dout = (cache['out' + str(L-1)] - Y.T) * (2/m)
        dW = np.dot(dout, cache['out' + str(L-2)].T) if L > 2 else 1/m * np.dot(dout, X)
        db = np.sum(dout, axis=1, keepdims=True)
        
        results[L - 1] = (dW, db) # 权重和偏置的梯度
        
        # 隐藏层梯度
        for i in reversed(range(1, L-1)):
            dout = np.dot(self.weights[i+1].T, dout)  * self.activation_derivative(cache['net' + str(i)])
            out_prev = cache['out' + str(i-1)] if i > 1 else X.T
            dW = np.dot(dout, out_prev.T)
            db = np.sum(dout, axis=1, keepdims=True)
            
            results[i] = (dW, db)
            
        return results

    def update(self, gradients):
        """
        更新权重和偏置
        :param gradients: 权重和偏置的梯度
        """
        for i in range(1, len(self.layers)):
            dW, db = gradients[i]
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def compute_loss(self, Y, Y_pred):
        """
        计算均方误差损失
        :param Y: 目标变量
        :param Y_pred: 预测值
        :return: 损失值
        """
        return np.mean((Y_pred.T - Y) ** 2)
    
    def run(self, X, Y):
        """
        :param X: 训练集特征
        :param Y: 训练集标签
        """
        for i in range(self.max_iterations):
            epoch_loss = 0
            # 创建mini-batches
            mini_batches = create_mini_batches(X, Y, self.batch_size)
            
            for X_batch, Y_batch in mini_batches:
                # 前向传播
                Y_pred, cache = self.forward(X_batch)
                
                # 计算损失
                batch_loss = np.mean((Y_pred.T - Y_batch) ** 2)
                epoch_loss += batch_loss * (X_batch.shape[0] / X.shape[0])
                
                # 反向传播
                gradients = self.backward(X_batch, Y_batch, cache)
                
                # 更新权重和偏置
                self.update(gradients)
            
            # 记录每个epoch的平均损失
            self.losses.append(epoch_loss)
            
            if i % 100 == 1:
                print()
            else:
                print(f"\rIteration {i}, Loss: {epoch_loss}", end="")
        
def main():
    X,Y = load_data("MLP_data.csv")

    # 数据标准化
    _,_,std_X_matrix = standardScaler(X)
    mean,std_Y,std_Y_matrix = standardScaler(Y)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = trainTestSplit(std_X_matrix, std_Y_matrix, test_size=0.2, random_state=42)

    # 创建MLP模型，添加batch_size参数
    mlp = MLP(layers=[X.shape[1], 64,32, 1], activation='relu', learning_rate=0.025, max_iterations=2000, batch_size=64)
    mlp.run(X_train, Y_train)

    # 预测和反标准化（将测试集数据输入）
    Y_pred, _ = mlp.forward(X_test)
    Y_pred = inverseStandardScaler(Y_pred.T, mean, std_Y)  # 反标准化，注意转置
    
    Y_original = inverseStandardScaler(Y_test, mean, std_Y)  # 反标准化

    # 计算均方误差
    mse = np.mean((Y_pred - Y_original) ** 2)
    print(f"\n测试集MSE: {mse}")

    # 可视化预测结果

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mlp.losses)), mlp.losses, label='Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.svg')
    plt.show()

    # 绘制预测值与真实值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_original, Y_pred, alpha=0.5)
    plt.plot([Y_original.min(), Y_original.max()], 
             [Y_original.min(), Y_original.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.savefig('prediction_results.svg')
    plt.grid(True)
    plt.show()
    
    # 使用PCA将特征从4维降至2维
    pca = PCA(n_components=2)
    X_test_original = inverseStandardScaler(X_test, *standardScaler(X)[:2])
    X_pca = pca.fit_transform(X_test_original)
    
    # 3D散点图：降维特征作为x,y轴，真实房价作为z轴
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], Y_original.flatten(), 
                         c=Y_original.flatten(), cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('PCA Feature 1')
    ax1.set_ylabel('PCA Feature 2')
    ax1.set_zlabel('Actual House Price')
    ax1.set_title('3D Scatter Plot of Actual House Prices')
    plt.colorbar(scatter, ax=ax1, label='House Price')
    plt.show()
    

    # 3D曲面图：降维特征作为x,y轴，预测房价作为z轴
    fig = plt.figure(figsize=(12, 10))
    ax2 = fig.add_subplot(111, projection='3d')
    
    # 创建网格以便绘制平滑曲面
    xi = np.linspace(min(X_pca[:, 0]), max(X_pca[:, 0]), 100)
    yi = np.linspace(min(X_pca[:, 1]), max(X_pca[:, 1]), 100)
    X1, Y1 = np.meshgrid(xi, yi)
    
    # 插值得到平滑的z值
    Z = griddata((X_pca[:, 0], X_pca[:, 1]), Y_pred.flatten(), 
                (X1, Y1), method='cubic', fill_value=Y_pred.mean())
    
    # 绘制曲面
    surf = ax2.plot_surface(X1, Y1, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # 添加原始散点
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], Y_pred.flatten(), c='black', s=10, alpha=0.5)
    
    ax2.set_xlabel('PCA Feature 1')
    ax2.set_ylabel('PCA Feature 2')
    ax2.set_zlabel('Predicted House Price')
    ax2.set_title('3D Surface Plot of Predicted House Prices')
    plt.colorbar(surf, ax=ax2, label='Predicted House Price')
    plt.show()

    # 查看原始特征对主成分的贡献
    components = pd.DataFrame(
        pca.components_,
        columns=['经度', '纬度', '房龄', '房主收入']
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(components, cmap='coolwarm')
    plt.xticks(range(4), ['经度', '纬度', '房龄', '房主收入'])
    plt.yticks(range(2), ['PC1', 'PC2'])
    plt.colorbar()
    plt.title('PCA 组成热力图')
    plt.savefig('pca_components.svg')
    plt.show()


    # 保存为html
    # 3D 散点图
    fig1 = go.Figure(data=[go.Scatter3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=Y_original.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=Y_original.flatten(),
        colorscale='Viridis',
        opacity=0.7,
        colorbar=dict(title='House Price')
    )
    )])

    fig1.update_layout(
        title='3D Scatter Plot of Actual House Prices',
        scene=dict(
            xaxis_title='PCA Feature 1',
            yaxis_title='PCA Feature 2',
            zaxis_title='Actual House Price'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig1.write_html("3D_scatter_actual_price.html")

    # 绘制3D曲面图
    # 网格创建
    xi = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
    yi = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
    X1, Y1 = np.meshgrid(xi, yi)

    # Z插值
    Z = griddata(
        (X_pca[:, 0], X_pca[:, 1]),
        Y_pred.flatten(),
        (X1, Y1),
        method='cubic',
        fill_value=np.mean(Y_pred)
    )

    fig2 = go.Figure()

    # 添加曲面
    fig2.add_trace(go.Surface(
        x=xi, y=yi, z=Z,
        colorscale='RdBu',
        opacity=0.8,
        colorbar=dict(title='Predicted Price')
    ))

    # 添加预测点（可选）
    fig2.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=Y_pred.flatten(),
        mode='markers',
        marker=dict(size=3, color='black', opacity=0.5),
        name='Predicted Points'
    ))

    fig2.update_layout(
        title='3D Surface Plot of Predicted House Prices',
        scene=dict(
            xaxis_title='PCA Feature 1',
            yaxis_title='PCA Feature 2',
            zaxis_title='Predicted House Price'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig2.write_html("3D_surface_predicted_price.html")
    
if __name__ == "__main__":
    main()