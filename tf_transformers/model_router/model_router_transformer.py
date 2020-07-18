import tensorflow as tf
from ..abstract_transformer import AbstractTransformer
from ..model_router.model_router_op import ModelRouterOp

class ModelRouterTransformer(AbstractTransformer):

    def __init__(self, operation: ModelRouterOp, features, ia=None, ib=None, oa=None, ob=None):
        """Transformer which takes in ModelRouterOp operation, performs an Operation on top of it

        :param operation:ModelRouterOP an class with list of operations which are applicable
        :param features:dict a dictionary of features to transform
        :param ia:default if setInputA is not provided
        :param ib:default if setInputB is not provided
        :param oa:default if setOutputA is not provided
        :param ob:default if setOutputB is not provided

        .. note::
            The way `ib=Some(10)` is defined in Spark,`ib=tf.constant(10)` can be defined in tensorflow

        .. code-block:: python

            features = (ModelRouterTransformer(ModelRouterOp.GreaterThan, features, oa = tf.constant(1.0,
                                   dtype=tf.float64), ob = tf.constant(0.0, dtype=tf.float64)).
             setInputA("sepal_width").
             setInputB("petal_length").
             setOutputCol("extra_input"))


        """
        super().__init__(operation, features)

        self.input_a = ia
        self.input_b = ib
        self.output_a = oa
        self.output_b = ob

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

    def setOutputA(self, output_a):
        """Sets the outputA to use when condition is `True`. Alternate to `oa` as part of `__init__`.
        If both are set, it'll throw `ValueError`

        :param output_a:str feature name which will contain the outputA feature
        :return:self object itself to support pipelining of operations
        """
        self.check_feature(feature_type='output_a', feature_col=output_a)
        self.output_a = self.features[output_a]
        return self

    def setOutputB(self, output_b):
        """Sets the outputB to use when condition is `True`. Alternate to `ob` as part of `__init__`.
        If both are set, it'll throw `ValueError`

        :param output_b:str feature name which will contain the outputB feature
        :return:self object itself to support pipelining of operations
        """
        self.check_feature(feature_type='output_b', feature_col=output_b)
        self.output_b = self.features[output_b]
        return self

    def setOutputCol(self, outputCol):
        """ Transforms the input feature using tf.where

        `check_types` is added considering all input_features should be of the same dtype

        :param outputCol:str feature name which will contain the transformed feature
        :return:dict `features` dict with `outputCol` as key and transformed feature as value
        """

        super().setOutputCol(outputCol)

        assert_op = self.check_types(inst_vars=self.__dict__.copy())
        with tf.control_dependencies([assert_op]):
            self.features[outputCol] = tf.where(
                self.operation(
                    self.input_a,
                    self.input_b
                ),
                self.output_a,
                self.output_b
            )

            return self.features
