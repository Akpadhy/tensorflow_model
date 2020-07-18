import tensorflow as tf
from ..abstract_operator import Abstract_Operator

class BinaryTensorOp(Abstract_Operator):

    Intersection = tf.sets.intersection
    Union = tf.sets.union
    Difference = tf.sets.difference