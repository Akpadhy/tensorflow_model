from abc import ABC, abstractmethod
import tensorflow as tf
from .abstract_operator import Abstract_Operator

class AbstractTransformer(ABC):

    def __init__(self, operation: Abstract_Operator, features):
        """Every transformer must inherit from base transformer.

        :param operation:TFOperation tensorflow operation to be sent in as input
        :param features:dict dictionary of features to be transformed
        """
        self.operation = operation
        self.features = features

    def check_feature(self, feature_type: str, feature_col: str):
        """Performs some check on the features

        Two checks performed, one if the feature is already defined as real-time tensor. Second,
        if the `feature_column` doesn't exist in `features` dictionary.
        :param feature_type:str define if the feature is input/output/powerfactor/etc..
        :param feature_col:str the actual feature name which is present in `features` dictionary
        :return:
        """
        if (feature_type in self.__dict__) and (tf.is_tensor(self.__dict__[feature_type])):
            raise ValueError("Feature type {} already declared as real-time tensor"
                             " before".format(feature_type))
        if feature_col not in self.features:
            raise ValueError("Column {} not in feature dictionary".format(feature_col))

    def setInputCol(self, inputCol: str):
        """Sets the input column to do transformations on. Column of name `inputCol` should be present in feature
        and can't be accessed as part of `__init__`. Usage of this function is not mandatory in favour of
        custom inputs in respective transformers like input_a, input_b, etc..

        :param inputCol:str feature name which will have the input feature
        :return:self object itself to support pipelining of operations
        """

        self.check_feature(feature_type='inputCol', feature_col=inputCol)
        self.inputCol = self.features[inputCol]
        return self

    def check_types(self, inst_vars):
        """Checks whether all instance variables are of the same type.
        Some transformers like tf.Greater, tf.where need all input variables to be of the same type.

        :param inst_vars:dict dictionary with key tensor-name and value tensor-value
        :return: assert operation to be used with control dependencies.
        """
        for k, v in inst_vars.copy().items():
            if not tf.is_tensor(v):
                inst_vars.pop(k)

        inst_vars_list = list(inst_vars)
        type_var_list = [True]
        for i in range(len(inst_vars_list) - 1):
            type_var_list.append(
                inst_vars[inst_vars_list[i]].dtype.is_compatible_with(
                    inst_vars[inst_vars_list[i + 1]].dtype
                )
            )
        return tf.Assert(tf.reduce_all(type_var_list), type_var_list, name="Incompatible_Data_Type")

    @abstractmethod
    def setOutputCol(self, outputCol):
        """Takes the output column and does feature transformations.

        :param outputCol:str feature name which will contain the transformed feature
        :return:dict `features` dict with `outputCol` as key and transformed feature as value
        """

        if outputCol in self.features:
            raise ValueError("Output column {} already exists.".format(outputCol))
