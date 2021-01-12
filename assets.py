import os
import shutil
from datetime import datetime


class AssetManager:

	def __init__(self, base_dir):
		self.__base_dir = base_dir

		self.__cache_dir = os.path.join(self.__base_dir, 'cache')
		if not os.path.exists(self.__cache_dir):
			os.mkdir(self.__cache_dir)

		self.__preprocess_dir = os.path.join(self.__cache_dir, 'preprocess')
		if not os.path.exists(self.__preprocess_dir):
			os.mkdir(self.__preprocess_dir)

		self.__models_dir = os.path.join(self.__cache_dir, 'models')
		if not os.path.exists(self.__models_dir):
			os.mkdir(self.__models_dir)

		self.__tensorboard_dir = os.path.join(self.__cache_dir, 'tensorboard')
		if not os.path.exists(self.__tensorboard_dir):
			os.mkdir(self.__tensorboard_dir)

		self.__out_dir = os.path.join(self.__base_dir, 'out')
		if not os.path.exists(self.__out_dir):
			os.mkdir(self.__out_dir)

	def get_preprocess_file_path(self, data_name):
		return os.path.join(self.__preprocess_dir, data_name + '.npz')

	def get_model_dir(self, model_name):
		return os.path.join(self.__models_dir, model_name)

	def recreate_model_dir(self, model_name):
		model_dir = self.get_model_dir(model_name)

		self.__recreate_dir(model_dir)
		return model_dir

	def get_tensorboard_dir(self, model_name):
		return os.path.join(self.__tensorboard_dir, model_name)

	def recreate_tensorboard_dir(self, model_name):
		tensorboard_dir = self.get_tensorboard_dir(model_name)

		self.__recreate_dir(tensorboard_dir)
		return tensorboard_dir

	def create_prediction_dir(self, model_name):
		prediction_dir = os.path.join(self.__out_dir, model_name, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
		if not os.path.exists(prediction_dir):
			os.makedirs(prediction_dir)

		return prediction_dir

	@staticmethod
	def __recreate_dir(path):
		if os.path.exists(path):
			shutil.rmtree(path)

		os.makedirs(path)

