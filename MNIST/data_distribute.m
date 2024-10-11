
% 读取 CSV 数据
filename = '/home/isirl/anaconda3/bin/MNIST-DNN/20241010_220534_10000000000_client_data_distribution.csv'; % 替换为实际的CSV文件名
data = readtable(filename);

% 提取用于绘图的数据
client_data = data{:, 2:end}; % 客户端的数据（跳过第一列Client）
num_clients = size(client_data, 1); % 客户端数量
num_labels = size(client_data, 2); % 标签数量

% 创建图形
figure;

% 绘制热图
imagesc(client_data'); % 转置数据，使得X轴为客户端，Y轴为标签
colorbar; % 添加颜色条
xlabel('Client index'); % X轴标签
ylabel('Training data label'); % Y轴标签

% 设置颜色范围为0到5000
caxis([0 3000]);

% 自定义 X 和 Y 轴刻度
xticks(1:num_clients); % 客户端编号
yticks(1:num_labels); % 标签编号

% 设置标题并添加 caption
title('Client data distribution (α=10^{10})'); % 图表标题

% 添加网格线以提高清晰度（可选）
grid on;

% 显示图表
set(gca, 'YDir', 'normal'); % 确保Y轴标签从下到上

% 保存为 EPS 文件
print('-depsc', 'alpha_10**10_client_data_distribution.eps');