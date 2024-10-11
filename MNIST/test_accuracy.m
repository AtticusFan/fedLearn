file_names = {
    %{
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n1_k1_SGD_accuracy.csv',      % (1) SGD, N=1
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n1_k1_Adam_accuracy.csv',     % (2) Adam, N=1
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n40_k5_SGD_accuracy.csv', % (3) SGD, N=40, K=5
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n40_k5_Adam_accuracy.csv',% (4) Adam, N=40, K=5
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n80_k5_SGD_accuracy.csv', % (5) SGD, N=80, K=5
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n80_k5_Adam_accuracy.csv',% (6) Adam, N=80, K=5
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n160_k5_SGD_accuracy.csv',  % (7) SGD, N=160, K=5
    '/home/isirl/anaconda3/bin/MNIST-DNN/0.1_n160_k5_Adam_accuracy.csv'   % (8) Adam, N=160, K=5
    %}
    '/home/isirl/master/fedLearn/0.1_n160_k100_SGD_accuracy.csv'
};

labels = {
    'SGD, N=1', 
    'Adam, N=1', 
    'SGD, N=40, K=5', 
    'Adam, N=40, K=5', 
    'SGD, N=80, K=5', 
    'Adam, N=80, K=5', 
    'SGD, N=160, K=5', 
    'Adam, N=160, K=5'
};

% 创建图形
figure;

% 绘制每条线
for i = 1:length(file_names)
    accuracy_list = readmatrix(file_names{i}); % 读取每个 CSV 文件
    communication_rounds = 1:length(accuracy_list); % X轴：从1到N的通訊回合
    plot(communication_rounds, accuracy_list, 'LineWidth', 2); % 绘制线条
    hold on; % 保持当前图形
end

% 添加图例
legend(labels, 'Location', 'Best'); % 添加图例，'Best'会自动选择最佳位置

% 添加标注
xlabel('Communication rounds'); % X轴标注
ylabel('Test accuracy (%)');     % Y轴标注

% 设置 X 轴和 Y 轴刻度
set(gca, 'XTick', 0:10:1000);  % X轴从0到100，每10一刻度
set(gca, 'YTick', 0:10:100);  % Y轴从0到100，每10一刻度
ylim([0 100]);                % 固定 Y 轴范围为 0 到 100

% 标题及网
title('Test Accuracy for α=0.1');
grid on;

% 保存为 EPS 高质量图档
saveas(gcf, 'alpha0.1_test_accuracy.eps', 'epsc');