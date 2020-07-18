import tensorflow as tf
from ..abstract_operator import Abstract_Operator


class BinaryOperation(Abstract_Operator):

    Multiply = tf.math.multiply
    Divide = tf.math.truediv
    Add = tf.math.add
    Subtract = tf.math.subtract
    Pow = tf.math.pow