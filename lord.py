import argparse
import os

import numpy as np

import dataset
from assets import AssetManager
from model.network import Lord, Config
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs, classes, contents = img_dataset.read_images()
	n_classes = np.unique(classes).size

	np.savez(
		file=assets.get_preprocess_file_path(args.data_name),
		imgs=imgs, classes=classes, contents=contents, n_classes=n_classes
	)


def split_classes(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, contents = data['imgs'], data['classes'], data['contents']

	n_classes = np.unique(classes).size
	test_classes = np.random.choice(n_classes, size=args.num_test_classes, replace=False)

	test_idx = np.isin(classes, test_classes)
	train_idx = ~np.isin(classes, test_classes)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
	)


def split_samples(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, contents = data['imgs'], data['classes'], data['contents']

	n_classes = np.unique(classes).size
	n_samples = imgs.shape[0]

	n_test_samples = int(n_samples * args.test_split)

	test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
	train_idx = ~np.isin(np.arange(n_samples), test_idx)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
	)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs'].astype(np.float32) / 255.0

	config = Config(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_classes=data['n_classes'].item(),

		content_dim=default_config['content_dim'],
		class_dim=default_config['class_dim'],

		content_std=default_config['content_std'],
		content_decay=default_config['content_decay'],

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim'],

		perceptual_loss_layers=default_config['perceptual_loss']['layers'],
		perceptual_loss_weights=default_config['perceptual_loss']['weights'],
		perceptual_loss_scales=default_config['perceptual_loss']['scales']
	)

	lord = Lord.build(config)
	lord.train(
		imgs=imgs,
		classes=data['classes'],

		batch_size=default_config['train']['batch_size'],
		n_epochs=default_config['train']['n_epochs'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	lord.save(model_dir)


def train_encoders(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	tensorboard_dir = assets.get_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs'].astype(np.float32) / 255.0

	backup_dir = os.path.join(model_dir, 'latent')
	if not os.path.exists(backup_dir):
		lord = Lord.load(model_dir, include_encoders=False)

		os.mkdir(backup_dir)
		lord.save(backup_dir)

	else:
		lord = Lord.load(backup_dir, include_encoders=False)

	lord.train_encoders(
		imgs=imgs,
		classes=data['classes'],

		batch_size=default_config['train_encoders']['batch_size'],
		n_epochs=default_config['train_encoders']['n_epochs'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	lord.save(model_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	split_classes_parser = action_parsers.add_parser('split-classes')
	split_classes_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_classes_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_classes_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_classes_parser.add_argument('-ntsi', '--num-test-classes', type=int, required=True)
	split_classes_parser.set_defaults(func=split_classes)

	split_samples_parser = action_parsers.add_parser('split-samples')
	split_samples_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_samples_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_samples_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_samples_parser.add_argument('-ts', '--test-split', type=float, required=True)
	split_samples_parser.set_defaults(func=split_samples)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.set_defaults(func=train)

	train_encoders_parser = action_parsers.add_parser('train-encoders')
	train_encoders_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_encoders_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_encoders_parser.set_defaults(func=train_encoders)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
