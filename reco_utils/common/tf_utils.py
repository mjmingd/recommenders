# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = "model_checkpoints"

"""Supported optimizers and their default arguments.
Common parameters and default arguments:
    learning_rate=0.001 (For SGD, the default is 0.01)
    epsilon=1e-07 (except Ftrl and SGD)
    decay
See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for details.
"""
OPTIMIZERS = dict(
    # rho=0.95, The decay rate
    adadelta=tf.keras.optimizers.Adadelta,
    # initial_accumulator_value=0.1
    adagrad=tf.keras.optimizers.Adagrad,
    # beta_1=0.9, The exponential decay rate for the 1st moment estimates
    # beta_2=0.999, The exponential decay rate for the 2nd moment estimates
    # amsgrad=False, Whether to apply AMSGrad variant
    adam=tf.keras.optimizers.Adam,
    # beta_1=0.9, The exponential decay rate for the 1st moment estimates
    # beta_2=0.999, The exponential decay rate for the exponentially weighted infinity norm
    adamax=tf.keras.optimizers.Adamax,
    # learning_rate_power=-0.5, Controls how the learning rate decreases during training.
    #                           Must be less or equal to zero. Use zero for a fixed learning rate.
    # initial_accumulator_value=0.1
    # l1_regularization_strength=0.0
    # l2_regularization_strength=0.0, Stabilization penalty
    # l2_shrinkage_regularization_strength=0.0, Magnitude penalty
    ftrl=tf.keras.optimizers.Ftrl,
    # beta_1=0.9
    # beta_2=0.999, The exponential decay rate for the exponentially weighted infinity norm
    nadam=tf.keras.optimizers.Nadam,
    # rho=0.9, Discounting factor for the history/coming gradient
    # momentum=0.0,
    # centered=False, If True, gradients are normalized by the estimated variance of the gradient;
    #                 if False, by the uncentered second moment
    rmsprop=tf.keras.optimizers.RMSprop,
    # momentum=0.0,
    # nesterov=False, Whether to apply Nesterov momentum
    sgd=tf.keras.optimizers.SGD,
)


def pandas_input_fn_for_saved_model(df, feat_name_type):
    """Pandas input function for TensorFlow SavedModel.
    
    Args:
        df (pd.DataFrame): Data containing features.
        feat_name_type (dict): Feature name and type spec. E.g.
            `{'userID': int, 'itemID': int, 'rating': float}`
        
    Returns:
        func: Input function
    """
    for feat_type in feat_name_type.values():
        assert feat_type in (int, float, list)

    def input_fn():
        examples = [None] * len(df)
        for i, sample in df.iterrows():
            ex = tf.train.Example()
            for feat_name, feat_type in feat_name_type.items():
                feat = ex.features.feature[feat_name]
                if feat_type == int:
                    feat.int64_list.value.extend([sample[feat_name]])
                elif feat_type == float:
                    feat.float_list.value.extend([sample[feat_name]])
                elif feat_type == list:
                    feat.float_list.value.extend(sample[feat_name])
            examples[i] = ex.SerializeToString()
        return {"inputs": tf.constant(examples)}

    return input_fn


def pandas_input_fn(
    df,
    y_col=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    seed=None,
    shuffle_buffer_size=1000,
):
    """Pandas input function for TensorFlow Estimator.
    This function returns a `tf.data.Dataset` function.

    Args:
        df (pd.DataFrame): Data containing features.
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function.
        num_epochs (int): Number of epochs to iterate over data. If None, will run forever.
        shuffle (bool): If True, shuffles the data queue.
        seed (int): Random seed for shuffle.
        shuffle_buffer_size (int): Buffer size for shuffling. If 1, will not be shuffled.

    Returns:
        tf.data.Dataset function
    """

    X_df = df.copy()
    if y_col is not None:
        y = X_df.pop(y_col).values
        if isinstance(y[0], np.float64):
            y = y.astype(np.float32)
    else:
        y = None

    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([l for l in values], dtype=np.float32)
        elif isinstance(values[0], np.float64):
            values = values.astype(np.float32)
        X[col] = values

    return lambda: _dataset(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )


def _dataset(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    seed=None,
    shuffle_buffer_size=1000,
):
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    elif seed is not None:
        import warnings

        warnings.warn("Seed was set but `shuffle=False`. Seed will be ignored.")

    return dataset.repeat(num_epochs).batch(batch_size)


def build_optimizer(name, lr=0.001, **kwargs):
    """Get an optimizer for TensorFlow Estimator.

    Args:
        name (str): Optimizer name. Note, to use 'Momentum', should specify
        lr (float): Learning rate
        kwargs: Optimizer arguments as key-value pairs

    Returns:
        tf.keras.optimizers.Optimizer
    """
    name = name.lower()

    try:
        optimizer_class = OPTIMIZERS[name]
    except KeyError:
        raise KeyError("Optimizer name should be one of: {}".format(list(OPTIMIZERS)))

    # Set parameters
    params = {"decay": kwargs.get("decay", 0.0)}
    if name == "ftrl":
        params["initial_accumulator_value"] = kwargs.get(
            "initial_accumulator_value", 0.1
        )
        params["learning_rate_power"] = kwargs.get(
            "learning_rate_power", -0.5
        )
        params["l1_regularization_strength"] = kwargs.get(
            "l1_regularization_strength", 0.0
        )
        params["l2_regularization_strength"] = kwargs.get(
            "l2_regularization_strength", 0.0
        )
    elif name == "sgd":
        params["nesterov"] = kwargs.get("nesterov", False)
        params["momentum"] = kwargs.get("momentum", 0.0)
    elif name == "rmsprop":
        params["centered"] = kwargs.get("centered", False)
        params["momentum"] = kwargs.get("momentum", 0.0)
    elif name == "adagrad":
        params["initial_accumulator_value"] = kwargs.get(
            "initial_accumulator_value", 0.1
        )
    elif name in ("adam", "adamax", "nadam"):
        params["beta_1"] = kwargs.get("beta_1", 0.9)
        params["beta_2"] = kwargs.get("beta_2", 0.999)
        if name == "adam":
            params["amsgrad"] = kwargs.get("amsgrad", False)

    return optimizer_class(learning_rate=lr, **params)


def export_model(model, tf_feat_cols, base_dir):
    """Export TensorFlow estimator model for serving.
    
    Args:
        model (tf.estimator.Estimator): Model to export.
        tf_feat_cols (list(tf.feature_column)): Feature columns.
        base_dir (str): Base directory to export the model.
    
    Returns:
        str: Exported model path
    """
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(tf_feat_cols)
    )

    exported_path = model.export_saved_model(base_dir, serving_input_fn)

    return exported_path.decode("utf-8")


def evaluation_log_listener(
    estimator,
    logger,
    true_df,
    y_col,
    eval_df,
    model_dir=None,
    batch_size=256,
    eval_fns=None,
    **eval_kwargs
):
    """Evaluation log listener for TensorFlow Estimator.
    For every checkpoints, evaluate the model by using the given evaluation functions.

    Args:
        estimator (tf.estimator.Estimator): Model to evaluate.
        logger (Logger): Custom logger to log the results.
            E.g., define a subclass of Logger for AzureML logging.
        true_df (pd.DataFrame): Ground-truth data.
        y_col (str): Label column name in true_df
        eval_df (pd.DataFrame): Evaluation data without label column.
        model_dir (str): Model directory to save the summaries to. If None, does not record.
        batch_size (int): Number of samples fed into the model at a time.
            Note, the batch size doesn't affect on evaluation results.
        eval_fns (iterable of functions): List of evaluation functions that have signature of
            (true_df, prediction_df, **eval_kwargs)->(float). If None, loss is calculated on true_df.
        **eval_kwargs: Keyword arguments for the evaluation functions.
            Note, prediction column name should be 'prediction'

    Returns:
        tf.train.SessionRunHook: Session run hook to evaluate the model while training.
    """

    return _EvaluationLogListener(
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        model_dir,
        batch_size,
        eval_fns,
        **eval_kwargs
    )


class _EvaluationLogListener(tf.estimator.CheckpointSaverListener):
    def __init__(
        self,
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        model_dir=None,
        batch_size=256,
        eval_fns=None,
        **eval_kwargs
    ):
        """Evaluation log hook class
        """
        self.model = estimator
        self.logger = logger
        self.true_df = true_df
        self.y_col = y_col
        self.eval_df = eval_df
        self.summary_dir = model_dir
        self.batch_size = batch_size
        self.eval_fns = eval_fns
        self.eval_kwargs = eval_kwargs
        self.summary_writer = None

    def begin(self):
        if self.summary_dir is None:
            self.summary_dir = "summary"
        self.summary_writer = tf.summary.create_file_writer(self.summary_dir)

    def after_save(self, session, global_step_value):
        logs = {}
        # By default, measure average loss
        result = self.model.evaluate(
            input_fn=pandas_input_fn(
                df=self.true_df, y_col=self.y_col, batch_size=self.batch_size
            )
        )["average_loss"]
        logs["validation_loss"] = result

        if self.eval_fns is not None:
            predictions = list(
                itertools.islice(
                    self.model.predict(
                        input_fn=pandas_input_fn(
                            df=self.eval_df, batch_size=self.batch_size
                        )
                    ),
                    len(self.eval_df),
                )
            )
            prediction_df = self.eval_df.copy()
            prediction_df["prediction"] = [p["predictions"][0] for p in predictions]
            for fn in self.eval_fns:
                result = fn(self.true_df, prediction_df, **self.eval_kwargs)
                logs[fn.__name__] = result

        self.logger.log("step", global_step_value)
        with self.summary_writer.as_default():
            for k, v in logs.items():
                tf.summary.scalar(k, v, step=global_step_value)
            self.summary_writer.flush()

    def end(self, session, global_step_value):
        self.summary_writer.close()


class MetricsLogger:
    """Metrics logger"""

    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.

        Args:
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)

    def get_log(self):
        """Getter
        
        Returns:
            dict: Log metrics.
        """
        return self._log
