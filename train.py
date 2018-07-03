"""Обучаем модель"""

import argparse
import os
import tensorflow as tf
from model.input_fn import input_fn, input_fn2
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
# parser.add_argument('--final_train', default='N',
#                     help="Whether to train on a whole dataset")
parser.add_argument('--train_evaluate', default='N',
                    help="train and evaluate each epoch")
parser.add_argument('--hub', default='N')
parser.add_argument('--num_gpus', type=int, default=4,
                    help="Number of GPUs to train on")
parser.add_argument('--save_epoch', type=int, default=2,
                    help="Save checkpoints every N epochs")
parser.add_argument('--evaluate_every_epoch', type=int, default=5,
                    help="Evaluate every X epochs")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    if (args.num_gpus > 1):
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
    else:
        distribution = None

    # session_config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    # session_config.gpu_options.allow_growth = True  # starts out allocating very little memory
    checkpoint_every = int(((params.train_size / params.batch_size) * args.save_epoch) / args.num_gpus)

    print('Train size = {}'.format(params.train_size))
    print('Batch size = {}'.format(params.batch_size))
    print('Checkpoint every {} steps'.format(checkpoint_every))

    model_dir = args.model_dir

    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=model_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    train_distribute=distribution,
                                    # session_config=session_config,
                                    save_checkpoints_steps=checkpoint_every,
                                    keep_checkpoint_max=None)  # all checkpoint files are kept

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    if os.path.isfile(os.path.join(args.model_dir, 'checkpoint')):
        states = tf.train.get_checkpoint_state(model_dir)
        all_states = states.model_checkpoint_path
        global_step = int(all_states.split('-')[-1])
        print('GLOBAL STEP = {}'.format(global_step))
    else:
        global_step = 0

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    if args.train_evaluate == 'Y':
        if args.hub == 'Y':
            train_input_fn = input_fn2(args.data_dir, params, 'train')
            eval_input_fn = input_fn2(args.data_dir, params, 'eval', False)
        else:
            train_input_fn =lambda: input_fn(args.data_dir, params, 'train', True, args.evaluate_every_epoch)
            eval_input_fn =lambda: input_fn(args.data_dir, params, 'eval', False)

        max_steps = int(((params.train_size / params.batch_size) * params.num_epochs) / args.num_gpus) + global_step

        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                max_steps=max_steps),
            tf.estimator.EvalSpec(input_fn=eval_input_fn)
        )
    else:
        if args.hub == 'Y':
            train_input_fn = input_fn2(args.data_dir, params, 'train')
            eval_input_fn = input_fn2(args.data_dir, params, 'eval', False)
        else:
            train_input_fn =lambda: input_fn(args.data_dir, params, 'train')
            eval_input_fn =lambda: input_fn(args.data_dir, params, 'eval', False)

        estimator.train(lambda: input_fn(args.data_dir, params, 'train'))
        tf.logging.info("Evaluation on test set.")
        res = estimator.evaluate(lambda: input_fn(args.data_dir, params, 'eval', False))

        for key in res:
            print("{}: {}".format(key, res[key]))
