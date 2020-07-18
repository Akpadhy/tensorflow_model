import tensorflow as tf
from ..abstract_operator import Abstract_Operator

class ModelRouterOp(Abstract_Operator):

    GreaterThan = tf.math.greater
    GreaterThanEqualTo = tf.math.greater_equal
    LessThan = tf.math.less
    LessThanEqualTo = tf.math.less_equal
    NotEqualTo = tf.math.not_equal
    EqualTo = tf.math.equal