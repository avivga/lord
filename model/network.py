import os
import pickle
from collections import namedtuple

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers, losses, regularizers
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.applications import vgg16
from keras_lr_multiplier import LRMultiplier

from model.evaluation import EvaluationCallback, EncodersEvaluationCallback


Config = namedtuple(
	typename='Config',
	field_names=[
		'img_shape', 'n_imgs', 'n_classes',
		'content_dim', 'class_dim',
		'content_std', 'content_decay',
		'n_adain_layers', 'adain_dim',
		'perceptual_loss_layers', 'perceptual_loss_weights', 'perceptual_loss_scales'
	]
)


class Lord:

	@classmethod
	def build(cls, config):
		content_embedding = cls.__build_regularized_embedding(
			config.n_imgs, config.content_dim, config.content_std, config.content_decay, name='content'
		)

		class_embedding = cls.__build_embedding(config.n_classes, config.class_dim, name='class')
		class_modulation = cls.__build_class_modulation(config.class_dim, config.n_adain_layers, config.adain_dim)
		generator = cls.__build_generator(config.content_dim, config.n_adain_layers, config.adain_dim, config.img_shape)

		return Lord(config, content_embedding, class_embedding, class_modulation, generator)

	@classmethod
	def load(cls, model_dir, include_encoders=False):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		content_embedding = load_model(os.path.join(model_dir, 'content_embedding.h5py'))
		class_embedding = load_model(os.path.join(model_dir, 'class_embedding.h5py'))
		class_modulation = load_model(os.path.join(model_dir, 'class_modulation.h5py'))

		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		if not include_encoders:
			return Lord(config, content_embedding, class_embedding, class_modulation, generator)

		content_encoder = load_model(os.path.join(model_dir, 'content_encoder.h5py'))
		class_encoder = load_model(os.path.join(model_dir, 'class_encoder.h5py'))

		return Lord(config, content_embedding, class_embedding, class_modulation, generator, content_encoder, class_encoder)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.content_embedding.save(os.path.join(model_dir, 'content_embedding.h5py'))
		self.class_embedding.save(os.path.join(model_dir, 'class_embedding.h5py'))
		self.class_modulation.save(os.path.join(model_dir, 'class_modulation.h5py'))
		self.generator.save(os.path.join(model_dir, 'generator.h5py'))

		if self.content_encoder:
			self.content_encoder.save(os.path.join(model_dir, 'content_encoder.h5py'))

		if self.class_encoder:
			self.class_encoder.save(os.path.join(model_dir, 'class_encoder.h5py'))

	def __init__(self, config,
				 content_embedding, class_embedding,
				 class_modulation, generator,
				 content_encoder=None, class_encoder=None):

		self.config = config

		self.content_embedding = content_embedding
		self.class_embedding = class_embedding
		self.class_modulation = class_modulation
		self.generator = generator
		self.content_encoder = content_encoder
		self.class_encoder = class_encoder

		self.vgg = self.__build_vgg()

	def train(self, imgs, classes, batch_size, n_epochs, model_dir, tensorboard_dir):
		img_id = Input(shape=(1, ))
		class_id = Input(shape=(1, ))

		content_code = self.content_embedding(img_id)
		class_code = self.class_embedding(class_id)
		class_adain_params = self.class_modulation(class_code)
		generated_img = self.generator([content_code, class_adain_params])

		model = Model(inputs=[img_id, class_id], outputs=generated_img)

		model.compile(
			optimizer=LRMultiplier(
				optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
				multipliers={
					'content-embedding': 10,
					'class-embedding': 10
				}
			),

			loss=self.__perceptual_loss_multiscale
		)

		lr_scheduler = CosineLearningRateScheduler(max_lr=3e-4, min_lr=1e-5, total_epochs=n_epochs)
		early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=100, verbose=1)

		checkpoint = CustomModelCheckpoint(self, model_dir)
		tensorboard = EvaluationCallback(self, imgs, classes, tensorboard_dir)

		model.fit(
			x=[np.arange(imgs.shape[0]), classes], y=imgs,
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[lr_scheduler, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

	def train_encoders(self, imgs, classes, batch_size, n_epochs, model_dir, tensorboard_dir):
		self.content_encoder = self.__build_content_encoder(self.config.img_shape, self.config.content_dim)
		self.class_encoder = self.__build_class_encoder(self.config.img_shape, self.config.class_dim)

		img = Input(shape=self.config.img_shape)

		content_code = self.content_encoder(img)
		class_code = self.class_encoder(img)
		class_adain_params = self.class_modulation(class_code)
		generated_img = self.generator([content_code, class_adain_params])

		model = Model(inputs=img, outputs=[generated_img, content_code, class_code])
		model.compile(
			optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999),
			loss=[self.__perceptual_loss, losses.mean_squared_error, losses.mean_squared_error],
			loss_weights=[1, 1e4, 1e4]
		)

		reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min', min_delta=1, factor=0.5, patience=20, verbose=1)
		early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=40, verbose=1)

		checkpoint = CustomModelCheckpoint(self, model_dir)
		tensorboard = EncodersEvaluationCallback(self, imgs, tensorboard_dir)

		model.fit(
			x=imgs, y=[imgs, self.content_embedding.predict(np.arange(imgs.shape[0])), self.class_embedding.predict(classes)],
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

	def __perceptual_loss(self, y_true, y_pred):
		perceptual_codes_pred = self.vgg(y_pred)
		perceptual_codes_true = self.vgg(y_true)

		normalized_weights = self.config.perceptual_loss_weights / np.sum(self.config.perceptual_loss_weights)
		loss = 0

		for i, (p, t) in enumerate(zip(perceptual_codes_pred, perceptual_codes_true)):
			loss += normalized_weights[i] * K.mean(K.abs(p - t), axis=[1, 2, 3])

		loss = K.mean(loss)
		return loss

	def __perceptual_loss_multiscale(self, y_true, y_pred):
		loss = 0

		for scale in self.config.perceptual_loss_scales:
			y_true_resized = tf.image.resize_images(y_true, (scale, scale), method=tf.image.ResizeMethod.BILINEAR)
			y_pred_resized = tf.image.resize_images(y_pred, (scale, scale), method=tf.image.ResizeMethod.BILINEAR)

			loss += self.__perceptual_loss(y_true_resized, y_pred_resized)

		return loss / len(self.config.perceptual_loss_scales)

	@classmethod
	def __build_embedding(cls, input_dim, output_dim, name):
		idx = Input(shape=(1, ))

		embedding = Embedding(input_dim, output_dim)(idx)
		embedding = Reshape(target_shape=(output_dim, ))(embedding)

		model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

		print('%s-embedding:' % name)
		model.summary()

		return model

	@classmethod
	def __build_regularized_embedding(cls, input_dim, output_dim, std, decay, name):
		idx = Input(shape=(1, ))

		embedding = Embedding(input_dim, output_dim, activity_regularizer=regularizers.l2(decay))(idx)
		embedding = Reshape(target_shape=(output_dim, ))(embedding)

		embedding = GaussianNoise(stddev=std)(embedding)

		model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

		print('%s-embedding:' % name)
		model.summary()

		return model

	@classmethod
	def __build_class_modulation(cls, class_dim, n_adain_layers, adain_dim):
		class_code = Input(shape=(class_dim, ))

		adain_per_layer = [Dense(units=adain_dim * 2)(class_code) for _ in range(n_adain_layers)]
		adain_all = Concatenate(axis=-1)(adain_per_layer)
		class_adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_all)

		model = Model(inputs=[class_code], outputs=class_adain_params, name='class-modulation')

		print('class-modulation arch:')
		model.summary()

		return model

	@classmethod
	def __build_generator(cls, content_dim, n_adain_layers, adain_dim, img_shape):
		content_code = Input(shape=(content_dim, ))
		class_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		initial_height = img_shape[0] // (2 ** n_adain_layers)
		initial_width = img_shape[1] // (2 ** n_adain_layers)

		x = Dense(units=initial_height * initial_width * (adain_dim // 8))(content_code)
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * (adain_dim // 4))(x)
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * adain_dim)(x)
		x = LeakyReLU()(x)

		x = Reshape(target_shape=(initial_height, initial_width, adain_dim))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 2))(x)
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = LeakyReLU()(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, class_adain_params])

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=img_shape[-1], kernel_size=(7, 7), padding='same')(x)
		target_img = Activation('sigmoid')(x)

		model = Model(inputs=[content_code, class_adain_params], outputs=target_img, name='generator')

		print('generator arch:')
		model.summary()

		return model

	def __build_vgg(self):
		vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

		layer_outputs = [vgg.layers[layer_id].output for layer_id in self.config.perceptual_loss_layers]
		feature_extractor = Model(inputs=vgg.inputs, outputs=layer_outputs)

		img = Input(shape=self.config.img_shape)

		if self.config.img_shape[-1] == 1:
			x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(img)
		else:
			x = img

		x = VggNormalization()(x)
		features = feature_extractor(x)

		model = Model(inputs=img, outputs=features, name='vgg')

		print('vgg arch:')
		model.summary()

		return model

	@classmethod
	def __build_content_encoder(cls, img_shape, content_dim):
		img = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(img)
		x = LeakyReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = LeakyReLU()(x)

		content_code = Dense(units=content_dim)(x)

		model = Model(inputs=img, outputs=content_code, name='content-encoder')

		print('content-encoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_class_encoder(cls, img_shape, class_dim):
		img = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(img)
		x = LeakyReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = LeakyReLU()(x)

		class_code = Dense(units=class_dim)(x)

		model = Model(inputs=img, outputs=class_code, name='class-encoder')

		print('class-encoder arch:')
		model.summary()

		return model


class AdaptiveInstanceNormalization(Layer):

	def __init__(self, adain_layer_idx, **kwargs):
		super().__init__(**kwargs)
		self.adain_layer_idx = adain_layer_idx

	def call(self, inputs, **kwargs):
		assert isinstance(inputs, list)

		x, adain_params = inputs
		adain_offset = adain_params[:, self.adain_layer_idx, :, 0]
		adain_scale = adain_params[:, self.adain_layer_idx, :, 1]

		adain_dim = x.shape[-1]
		adain_offset = K.reshape(adain_offset, (-1, 1, 1, adain_dim))
		adain_scale = K.reshape(adain_scale, (-1, 1, 1, adain_dim))

		mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
		x_standard = (x - mean) / (tf.sqrt(var) + 1e-7)

		return (x_standard * adain_scale) + adain_offset

	def get_config(self):
		config = {
			'adain_layer_idx': self.adain_layer_idx
		}

		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))


class VggNormalization(Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, **kwargs):
		x = inputs * 255
		return vgg16.preprocess_input(x)


class CustomModelCheckpoint(Callback):

	def __init__(self, model, path):
		super().__init__()
		self.__model = model
		self.__path = path

	def on_epoch_end(self, epoch, logs=None):
		self.__model.save(self.__path)


class CosineLearningRateScheduler(Callback):

	def __init__(self, max_lr, min_lr, total_epochs):
		super().__init__()

		self.max_lr = max_lr
		self.min_lr = min_lr
		self.total_epochs = total_epochs

	def on_train_begin(self, logs=None):
		K.set_value(self.model.optimizer.lr, self.max_lr)

	def on_epoch_end(self, epoch, logs=None):
		fraction = epoch / self.total_epochs
		lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction * np.pi))

		K.set_value(self.model.optimizer.lr, lr)
		logs['lr'] = lr
