from denoising_diffusion_pytorch import Unet3D, GaussianDiffusion3D, Trainer3D


def main():
	model = Unet3D(
		dim = 64,
		dim_mults = (1, 2, 4, 8),
	)

	diffusion = GaussianDiffusion3D(
		model,
		image_size = 64,
		timesteps = 1000,           # number of steps
		sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
		loss_type = 'l1',           # L1 or L2
	)

	trainer = Trainer3D(
		diffusion,
		'C:/devel/mri-anomaly/data/normal',
		#train_batch_size = 32,
		train_batch_size = 2,
		train_lr = 8e-5,
		#train_num_steps = 700000,         # total training steps
		train_num_steps = 2,              # total training steps
		gradient_accumulate_every = 2,    # gradient accumulation steps
		ema_decay = 0.995,                # exponential moving average decay
		amp = True,                       # turn on mixed precision
		calculate_fid = False,            # whether to calculate fid during training
	)

	trainer.train()


if __name__ == '__main__':
	main()
