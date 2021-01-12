default_config = dict(
	content_dim=128,
	class_dim=256,

	content_std=1,
	content_decay=0.001,

	n_adain_layers=4,
	adain_dim=256,

	perceptual_loss=dict(
		layers=[2, 5, 8, 13, 18],
		weights=[1, 1, 1, 1, 1],
		scales=[64, ]
	),

	train=dict(
		batch_size=64,
		n_epochs=1000
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200
	)
)
