import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()


class CombinedEstimator(tf.estimator.Estimator):
    def __init__(self, estimators, label_names, MODEL_DIR):

        def model_fn(mode, features, labels):
            
            serving_dict = None
            predictions_dict = None
            train_op_list = None
            add_loss = None
            eval_metric_dict = None
            
            label_tensors = [features.pop(label_names[index], None) for index, _ in enumerate(label_names)]
            
            for index, estimator in enumerate(estimators):
                
                # call model function
                spec = estimator._call_model_fn(features, label_tensors[index], mode, estimator.config)
                
                #Mode='infer/predict', combine both export_outputs
                if spec.export_outputs:
                    if not serving_dict:
                        serving_dict = spec.export_outputs['predict'].outputs
                    else:
                        serving_dict = {**serving_dict, **spec.export_outputs['predict'].outputs}
                                    
                #Mode='all', combine multiple predictions - checkpointing purposes
                if spec.predictions:
                    if not predictions_dict:
                        predictions_dict = spec.predictions
                    else:
                        predictions_dict = {**predictions_dict, **spec.predictions}

                # Mode='train', combine train_op - actual purpose
                if tf.is_tensor(spec.train_op):
                    if not tf.is_tensor(train_op_list):
                        train_op_list = [spec.train_op]
                    else:
                        train_op_list += spec.train_op

                # mode='train', combine loss - checkpointing purpose
                if tf.is_tensor(spec.loss):
                    if not tf.is_tensor(add_loss):
                        add_loss = spec.loss
                    else:
                        add_loss += spec.loss
                
                if spec.eval_metric_ops:
                    if not eval_metric_dict:
                        eval_metric_dict = spec.eval_metric_ops
                    else:
                        eval_metric_dict = {**eval_metric_dict, **spec.eval_metric_ops}
            
            copy = list(spec)
            copy[1] = predictions_dict
            copy[2] = add_loss
            
            if train_op_list:
                train_op = tf.group(*train_op_list)
            else:
                train_op = None
            
            copy[3] = train_op
            copy[4] = eval_metric_dict

            if serving_dict:
                export_outputs = {
                    "serving_default": tf.estimator.export.PredictOutput(serving_dict),
                    "predict": tf.estimator.export.PredictOutput(serving_dict)
                }
            else:
                export_outputs = None
            copy[5] = export_outputs
            return tf.estimator.EstimatorSpec(*copy)

        super(CombinedEstimator, self).__init__(model_fn, MODEL_DIR, estimators[0].config)
