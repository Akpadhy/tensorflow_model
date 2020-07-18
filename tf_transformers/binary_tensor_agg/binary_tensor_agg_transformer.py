import tensorflow as tf
from ..abstract_transformer import AbstractTransformer
from ..binary_tensor_agg.binary_tensor_agg_op import BinaryTensorAggOp
from ..binary_tensor_agg.binary_tensor_op import BinaryTensorOp

class BinaryTensorAggTransformer(AbstractTransformer):

    def __init__(self, op1: BinaryTensorOp, op2: BinaryTensorAggOp, features, cast=tf.float64):
        """Transformer which takes in BinaryTensorOp and BinaryTensorAgg operation, performs an Operation on top of it

        Doesn't take real-time constants.

        :param op1:BinaryTensorOp an class with list of operations which are applicable
        :param op2:BinaryTensorAggOp class with list of operations which are applicable.
        :param features:dict a dictionary of features to transform
        :param cast:tf.dtypes mention the datatype of the outputCol to be cast to, defaults to double (float64)

        .. code-block:: python

            features = (BinaryTensorAggTransformer(BinaryTensorOp.Intersection, BinaryTensorAggOp.Fraction, features).
            setInputA("item_ids").
            setInputB("o2p_highitems_res_1Mn").
            setOutputCol("high_item"))

        .. note::
            Caveat -

            1. intersection can only be performed on int64 arrays. Will work in case of item-ids
            2. Using cast of anything lesser than float64 would result in lossy conversion, due to division. defaults
            to tf.float64
            3. op2 is applied on the output of intersection and input_b. if in case you want to divide it by
            input_a, swap both while defining this transformer. The same is followed in MLEAP as well

            def apply(i1: Array[Double], i2: Array[Double]): Double = op match {
                case Fraction =>
                  i2.intersect(i1).length / (i2.length * 1.0)
              }
        """
        # super().__init__(operation, features) - doesn't support multiple operations
        self.op1 = op1
        self.op2 = op2
        self.features = features
        self.cast = cast

    def setInputA(self, input_a):
        """Sets the inputA to do transformations on. Alternate to `ia` as part of `__init__`
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
        """ Transforms the input feature using transformers

          :param outputCol:str feature name which will contain the transformed feature
          :return:dict `features` dict with `outputCol` as key and transformed feature as value
          """

        super().setOutputCol(outputCol)

        assert_op = self.check_types(inst_vars=self.__dict__.copy())

        with tf.control_dependencies([assert_op]):

            # casting to integer is necessary because intersection/union/set operations are applicable on int64
            cast_input_a = tf.cast(self.input_a, tf.int64)
            cast_input_b = tf.cast(self.input_b, tf.int64)

            # intersection works on 2D arrays
            expand_cast_input_a = tf.expand_dims(cast_input_a, 0)
            expand_cast_input_b = tf.expand_dims(cast_input_b, 0)

            interaction_a_b = self.op1(expand_cast_input_a, expand_cast_input_b)
            size_interaction_a_b = tf.size(interaction_a_b.values)
            size_num_split_elem_b = tf.size(cast_input_b)

            output_feature = tf.cast(
                self.op2(
                    size_interaction_a_b,
                    size_num_split_elem_b
                ),
                self.cast
            )
            self.features[outputCol] = tf.expand_dims(output_feature, -1)

            return self.features