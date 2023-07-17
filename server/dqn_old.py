class DQN(nn.Module):
	def __init__(self, obs_size, act_size, use_pooling = False, learning_rate = 0.0001, gamma = 0.99, replay_buffer_size = 1000000, model_params = None):
		super(DQN, self).__init__()
		self.obs_size = obs_size
		self.channels = obs_size[0]
		self.use_pooling = use_pooling

		self.alpha = learning_rate
		self.gamma = gamma

		self.memory = deque(maxlen=replay_buffer_size)

		self.replay_buffer_size = replay_buffer_size

		if model_params is not None:
			self.load_state_dict(model_params)

		if self.use_pooling:
			self.cnn = nn.Sequential(
				nn.Conv2d(self.channels, 32, kernel_size=8, stride=4, padding=2),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
				nn.ReLU(), 
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Flatten(),
			)
		else:
			self.cnn = nn.Sequential(
				nn.Conv2d(self.channels, 32, kernel_size=8, stride=4, padding=2),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Flatten(),
			)

		with torch.no_grad():
			outL = self.cnn(torch.empty(1, 1, 88, 80)).numel()

		# self.cnn = nn.Sequential(
		# 	nn.Linear(obs_size, 128),
		# 	nn.ReLU(),
		# 	nn.Linear(128, 128),
		# 	nn.ReLU(),
		# 	nn.Linear(128, act_size)
		# )

		# conv_output_size = self.conv_output_dim()

		self.linear = nn.Sequential(nn.Linear(outL, 512), nn.ReLU(), nn.Linear(512, act_size))

		self.Q = nn.Sequential(self.cnn, self.linear)
		self.target = copy.deepcopy(self.Q)

		self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

	def forward(self, x):
		x = self.Q(x)
		return x
	
	def synchronize_target(self, polyak_coefficent = 1.0):
		for target_param, param in zip(self.target.parameters(), self.Q.parameters()):
			target_param.data.copy_(polyak_coefficent*param.data + target_param.data*(1.0 - polyak_coefficent))

	def get_action(self, obs):
		with torch.no_grad():
			obs = torch.as_tensor(obs).float().unsqueeze(0)
			return self.Q(obs).max(dim=1)[1].item()
	
	def update(self, obs, act, rew, obs_next, done):
		obs = torch.as_tensor(obs).float()
		act = torch.as_tensor(act).long()
		rew = torch.as_tensor(rew).float()
		obs_next = torch.as_tensor(obs_next).float()
		done = torch.as_tensor(done).float()

		# compute loss
		# Torch.no_grad to avoid updating target networks with autograd
		with torch.no_grad():
			max_future_q = self.target(obs_next).max(dim=1)[0]
			target_q = rew + (1 - done) * self.gamma * max_future_q

		# get current q values
		current_q = self.Q(obs)[0, act]

		loss = F.mse_loss(current_q, target_q)

		# update
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss
	
	def save_experience(self, obs, act, rew, obs_next, done):
		self.memory.append((obs, act, rew, obs_next, done))

	def sample_experience(self, batch_size):
		batch = random.sample(self.memory, min(batch_size, len(self.memory)))
		obs, act, rew, obs_next, done = map(np.stack, zip(*batch))
		return obs, act, rew, obs_next, done
	


	