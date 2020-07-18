import tensorflow as tf
from ..abstract_transformer import AbstractTransformer
from ..array_extraction.array_extractor_op import ArrayExtractorOp

class ArrayExtractorTransformer(AbstractTransformer):

    def __init__(self, operation: ArrayExtractorOp, features, cast=tf.float64):
        """Transformer which takes in ArrayExtractorOp operation, performs an Operation on top of it.
        Use setInputCol function which is inherited by the super class

        :param operation:ArrayExtractorOp an class with list of operations which are applicable
        :param features:dict a dictionary of features to transform
        :param cast:tf.dtypes mention the datatype of the outputCol to be cast to, defaults to double (float64)

        .. code-block:: python

            features = (ArrayExtractorTransformer(ArrayExtractorOp.Max, features).
            setInputCol("sepal_width").
            setOutputCol("extra_input"))

        """
        super().__init__(operation, features)
        self.cast = cast

    def setOutputCol(self, outputCol):
        """ Transforms the input feature using transformers

        :param outputCol:str feature name which will contain the transformed feature
        :return:dict `features` dict with `outputCol` as key and transformed feature as value
        """

        super().setOutputCol(outputCol)

        assert_op = self.check_types(inst_vars=self.__dict__.copy())
        with tf.control_dependencies([assert_op]):
            output_feature = tf.cast(self.operation(self.inputCol),
                                     self.cast)

            self.features[outputCol] = output_feature
            return self.features