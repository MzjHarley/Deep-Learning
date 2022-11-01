class ScheduledOptim():
    def __init__(self, optimizer, lr, d_model, n_warm_steps):
        self._optimizer = optimizer
        self.d_model = d_model
        self.lr = lr
        self.n_warm_steps = n_warm_steps
        self.n_steps = 0

    def step_and_update(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()  # 清空过往梯度

    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** -0.5, self.n_steps * (self.n_warm_steps ** -1.5))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr * self._get_lr_scale()
        self._optimizer.param_groups[0]['lr'] = lr  # 调整训练过程中优化器对应的学习率
