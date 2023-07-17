import gymnasium as gym
from dqn import DQN
from logger import Logger
import random
import numpy as np

np.object = object    

# Helpful preprocessing taken from github.com/ageron/tiny-dqn
def process_frame(frame):

	mspacman_color = np.array([210, 164, 74]).mean()
	img = frame[1:176:2, ::2]    # Crop and downsize
	img = img.mean(axis=2)       # Convert to greyscale
	img[img==mspacman_color] = 0 # Improve contrast by making pacman white
	img = (img - 128) / 128 - 1  # Normalize from -1 to 1.
	
	return np.expand_dims(img.reshape(88, 80), axis=0)  

# Averages images from the last few frame
def  blend_images (images, blend):
	avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

	for image in images:
		avg_image += image
		
	if len(images) < blend:
		return avg_image / len(images)
	else:
		return avg_image / blend

def main(
		initial_eplison=1,
		final_eplison=0.05,
		e_greedy_frames=20000 * 2500,
		lr=0.0001,
		gamma=0.99,
		training_episodes=5000,
		hardsync_target_every=10,
		gradient_updates_per_update=1,
		max_timesteps=20000,
		train_after=1000,
):
	env = gym.make('ALE/MsPacman-v5')
	print(env.action_space.n)
	agent = DQN((1, 88, 80), 9, learning_rate=lr, gamma=gamma, use_pooling=False)
	
	eplison = initial_eplison
	tt = 0

	for e in range(training_episodes):
		ep_rews = []
		ep_len = 0
		# state = process_frame(env.reset())
		obs, _ = env.reset()
		obs = process_frame(obs)
		
		for skip in range(90):
			env.step(0)
		
		for t in range(max_timesteps):
			if t % hardsync_target_every == 0:
				agent.synchronize_target()
			
			if random.random() > eplison:
				action = agent.get_action(obs)
			else:
				action = env.action_space.sample()
			
			next_obs, reward, done, _, _ = env.step(action)
			next_obs = process_frame(next_obs)

			ep_rews.append(reward)
			ep_len += 1
			tt += 1

			eplison = max(final_eplison, initial_eplison - (initial_eplison - final_eplison) * tt / e_greedy_frames)

			agent.save_experience(obs, action, reward, next_obs, done)
			obs = next_obs

			if done:
				print("Episode rew %f, len %d" % (sum(ep_rews), ep_len))
				break

			if tt > train_after and tt % 4 == 0:
				for _ in range(gradient_updates_per_update):
					# agent.train()
					s, a, r, n, d = agent.sample_experience(32)
					agent.update(s, a, r, n, d)
		


			 
		
	
	# for e in range(training_episodes):
		# rews = []
		
main()
