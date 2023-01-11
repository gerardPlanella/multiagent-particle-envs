class SimpleTorusConfig: 
    __slots__ = ["world_size", "symmetric", "pred_colors", "n_preds", "pred_vel", "prey_vel", "discrete", "rew_shape"]

    def __init__(self, world_size = 0.05, symmetric = False, pred_colors = "regular", n_preds = 0, pred_vel = 0.0, prey_vel = 0.0, discrete = False, rew_shape = 0.0) -> None:
        self.world_size = world_size
        self.symmetric = symmetric
        self.pred_colors = pred_colors
        self.n_preds = n_preds
        self.pred_vel = pred_vel
        self.prey_vel = prey_vel
        self.discrete = discrete
        self.rew_shape = rew_shape

