import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

def cost_promise(y_true, y_pred, alpha1, alpha2, beta1, beta2):
    
    e = tf.math.abs(tf.math.subtract(y_true, y_pred))
    s = tf.math.sign(tf.math.subtract(y_true, y_pred))
    
    e1 = tf.math.truediv(tf.math.add(tf.math.subtract(e, beta1), tf.math.abs(tf.math.subtract(e, beta1))),
                         tf.constant(2.0, tf.float64))
    e2 = tf.math.truediv(tf.math.add(tf.math.subtract(e, beta2), tf.math.abs(tf.math.subtract(e, beta2))),
                        tf.constant(2.0, tf.float64))
    
    intermediate_loss = tf.math.add(tf.math.multiply(tf.math.multiply(tf.math.add(tf.constant(1.0, tf.float64), s), alpha1), 
                                    tf.math.multiply(e1, tf.math.exp(tf.math.truediv(y_true, tf.constant(60.0, tf.float64))))),
                                    tf.math.multiply(tf.math.multiply(tf.math.subtract(tf.constant(1.0, tf.float64), s), alpha2),
                                    tf.math.multiply(e2, tf.math.exp(tf.math.subtract(tf.constant(1.0, tf.float64), tf.math.truediv(y_true,
                                                                                                                                    tf.constant(60.0,tf.float64)))))))
    
    loss = tf.math.truediv(intermediate_loss,tf.constant(2.0, tf.float64))
    return tf.math.reduce_mean(loss, axis=-1)

def promise_loss(alpha1, alpha2, beta1, beta2):
    def cost(y_true, y_pred):
        return cost_promise(y_true, y_pred, alpha1, alpha2, beta1, beta2)
    return cost

def o2a_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 =2.0):
    
    def model_fn(features, labels, mode):
        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 32)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 16)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 8)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_o2a':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })
    
        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_o2a":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn

def fm_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 =2.0):
    def model_fn(features, labels, mode):
        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 32)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 16)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 8)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_fm':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_fm":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn

def wt_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 =2.0):
    def model_fn(features, labels, mode):
    
        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 32)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 16)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 8)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_wt':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_wt":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn

def o2p_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 2.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 = 2.0):
    def model_fn(features, labels, mode):

        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 64)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 32)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 16)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_O2P':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })
        
        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_O2P":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn

def lm_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 = 2.0):
    def model_fn(features, labels, mode):

        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        l1 = tf.compat.v1.layers.dense(input_layer, units = 64)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 32)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 16)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_last_mile':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })
        
        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_last_mile":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn


def o2d_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 0.0, BETA_2 = 0.0):
    def model_fn(features, labels, mode):
        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 256)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 128)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 64)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        #l4 = tf.compat.v1.layers.dense(r3, units = 32)
        #r4 = tf.nn.leaky_relu(l4, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_O2D_accurate':logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_O2D_accurate":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn


def o2d_beef_model_fn(features_regressor, LEAKY_RELU_ALPHA, LEARNING_RATE, LOSS='mae', 
                 ALPHA_1 = 1.0, ALPHA_2 = 1.0, BETA_1 = 2.0, BETA_2 = 2.0):
    def model_fn(features, labels, mode):
        
        # same feature set for beef
        features = {feat_name: tf.expand_dims(features[feat_name], -1) for feat_name in features_regressor}
        input_layer = tf.concat([features[feat_name] for feat_name in features], axis=-1)

        # feed forward neural network
        l1 = tf.compat.v1.layers.dense(input_layer, units = 256)
        r1 = tf.nn.leaky_relu(l1, alpha=LEAKY_RELU_ALPHA)
        l2 = tf.compat.v1.layers.dense(r1, units = 128)
        r2 = tf.nn.leaky_relu(l2, alpha=LEAKY_RELU_ALPHA)
        l3 = tf.compat.v1.layers.dense(r2, units = 128)
        r3 = tf.nn.leaky_relu(l3, alpha=LEAKY_RELU_ALPHA)
        logits = tf.compat.v1.layers.dense(r3, units=1)

        predictions = {'predicted_O2D': logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions = predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        if LOSS == 'custom':
            loss_fn = promise_loss(tf.constant(ALPHA_1, tf.float64),
                                  tf.constant(ALPHA_2, tf.float64),
                                  tf.constant(BETA_1, tf.float64),
                                  tf.constant(BETA_2, tf.float64))
        else:
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              predictions = predictions)
        # eval metrics
        eval_metric_ops = {"mae_O2D":tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions = logits)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
    return model_fn

