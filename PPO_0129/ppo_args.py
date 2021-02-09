
class PPOArgs:

    def __init__(self):
        self.env_name = 'UAV_3_000'
        self.gamma = 0.99
        self.n_worker = 8
        self.samples_per_worker = 500
        self.n_updates = 10
        self.lr = 1e-3
        self.clip = 0.2
        self.std_begin = 0.3
        self.std_end = 0.3
        self.std_anneal = 1.0
        
        self.v_c = 1.0
        self.n_agents = 10
        self.log_dir = './logs/{}/{}'.format(self.env_name, self.n_agents)