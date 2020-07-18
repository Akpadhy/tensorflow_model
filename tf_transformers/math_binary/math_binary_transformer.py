import tensorflow as tf
from ..abstract_transformer import AbstractTransformer
from ..math_binary.math_binary_op import BinaryOperation

class MathBinaryTransformer(AbstractTransformer):

    def __init__(self, operation: BinaryOperation, features, ia=None, ib=None):
        """Transformer which takes in BinaryOperation operation, performs an Operation on top of it

        :param operation:BinaryOperation an class with list of operations which are applicable
        :param features:dict a dictionary of features to transform
        :param ia:default if setInputA is not provided
        :param ib:default if setInputB is not provided

        .. note::
            The way `ib=Some(10)` is defined in Spark,`ib=tf.constant(10)` can be defined in tensorflow

        .. code-block:: python

            features = (MathBinaryTransformer(BinaryOperation.Add, features, ib = tf.constant(1, tf.float64)).
            setInputA("sepal_width").
            setOutputCol("extra_input"))

        """
        super().__init__(operation, features)

        self.input_a = ia
        self.input_b = ib

    def setInputA(self, input_a):
        """Sets the inputA to do transformations on. Alternate to `ia` as part of `__init__`.
        If both are set, it'll throw `ValueError`

        :param input_a:str feature name which will contain the inputA feature
        :return:self object itself to support pipelining of operations
        """
        self.check_feature(feature_type='input_a', feature_col=input_a)
        self.input_a = self.features[input_a]
        return self

    def setInputB(self, input_b):
        """Sets the inputB to do transformations on. Alternate to `ib` as part of `__init__`.
        If both are set, it'll throw `ValueError`

        :param input_b:str feature name which will contain the inputB feature
        :return:self object itself to support pipelining of operations
        """
        self.check_feature(feature_type='input_b', feature_col=input_b)
        self.input_b = self.features[input_b]
        return self

    def setOutputCol(self, outputCol):
        """ Transforms the input feature using tf.math.<>

        `check_types` is added considering all input_features should be of the same dtype

        :param outputCol:str feature name which will contain the transformed feature
        :return:dict `features` dict with `outputCol` as key and transformed feature as value
        """

        super().setOutputCol(outputCol)

        assert_op = self.check_types(inst_vars=self.__dict__.copy())
        with tf.control_dependencies([assert_op]):
            self.features[outputCol] = self.operation(
                    self.input_a,
                    self.input_b
                )
            return self.features