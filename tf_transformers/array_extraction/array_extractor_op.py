import tensorflow as tf
from ..abstract_operator import Abstract_Operator

class ArrayExtractorOp(Abstract_Operator):

    Max = lambda x: tf.expand_dims(tf.math.reduce_max(x), -1)
    Min = lambda x: tf.expand_dims(tf.math.reduce_min(x), -1)
    Sum = lambda x: tf.expand_dims(tf.math.reduce_sum(x), -1)
    Mean = lambda x: tf.expand_dims(tf.math.reduce_mean(x), -1)
    Length = lambda x: tf.expand_dims(tf.size(x), -1)

    First = lambda x: tf.slice(x, [0], [1])
    Last = lambda x: tf.slice(x, [tf.math.subtract(tf.size(x), 1)], [1])