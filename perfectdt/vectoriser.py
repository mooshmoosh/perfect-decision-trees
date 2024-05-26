import numpy as np
from .python_expressions import BooleanExpressionBuilder, ExpressionBuilder, NullableExpression


class Vectoriser:
    def __init__(self, include_constant=True):
        self.include_constant = int(include_constant)

    def fit(self, inputs):
        self.input_keys = {}
        self.null_keys = {}
        self.float_keys = {}
        self.ranges = {}
        for input_dict in inputs:
            for key, value in input_dict.items():
                if key not in self.input_keys:
                    self.input_keys[key] = len(self.input_keys)
                if value is None:
                    if ("null", key) not in self.input_keys:
                        null_idx = len(self.input_keys)
                        self.input_keys[("null", key)] = null_idx
                        self.null_keys[key] = null_idx
                elif not isinstance(value, (int, float)):
                    raise ValueError(f"Can't handle value: ({value})")
                else:
                    self.float_keys[key] = self.input_keys[key]
                    if key not in self.ranges:
                        self.ranges[key] = (value, value)
                    else:
                        old_min, old_max = self.ranges[key]
                        self.ranges[key] = (min(old_min, value), max(old_max, value))

    def to_vectors(self, inputs):
        result = np.zeros((len(inputs), len(self.input_keys) + self.include_constant))
        for row, value_dict in enumerate(inputs):
            self._update_vector(result[row], value_dict)
        return result

    def to_vector(self, inputs):
        result = np.zeros(len(self.input_keys) + self.include_constant)
        self._update_vector(result, inputs)
        return result

    def _update_vector(self, vector, value_dict):
        for key, idx in self.input_keys.items():
            match key:
                case ("null", str):
                    if value_dict.get(key[1]) is None:
                        vector[idx] = 1
                    else:
                        vector[idx] = 0
                case str:
                    value = value_dict.get(key)
                    if value is None:
                        vector[idx] = 0
                    else:
                        vector[idx] = self.to_scaled_value(key, value)
        if self.include_constant:
            vector[-1] = 1

    def _forward_scale_map(self, key):
        min_val, max_val = self.ranges[key]
        if min_val == max_val:
            return 0, min_val
        return 2 / (max_val - min_val), -(min_val + max_val) / (max_val - min_val)

    def to_scaled_value(self, key, value):
        a, b = self._forward_scale_map(key)
        return a * value + b

    def from_scaled_value(self, key, value):
        a, b = self._forward_scale_map(key)
        return (value - b) / a

    def from_vector(self, vector):
        result = {}
        for key, idx in self.null_keys.items():
            if vector[idx] > 0.5:
                result[key] = None
        for key, idx in self.float_keys.items():
            if key not in result:
                result[key] = self.from_scaled_value(key, vector[idx])
        return result

    def to_expression(self, vector):
        return self._to_expression_builder(vector, True).to_code()

    def _to_expression_builder(self, vector, is_boolean):
        if is_boolean:
            builder = BooleanExpressionBuilder()
        else:
            builder = ExpressionBuilder()
        builder.nullable_keys = self.null_keys

        for key, idx in self.float_keys.items():
            a, b = self._forward_scale_map(key)
            builder.add_coefficient(key, a * vector[idx])
            builder.add_constant_term(b * vector[idx])

        for key, idx in self.null_keys.items():
            builder.add_is_null_coefficient(key, vector[idx])

        if self.include_constant:
            builder.add_constant_term(vector[-1])

        return builder

    def to_mapped_expressions(self, matrix, input_vectoriser):
        result = {}

        for key, idx in self.null_keys.items():
            condition = input_vectoriser._to_expression_builder(matrix[idx], True)
            result[key] = NullableExpression(condition=condition)

        for key, idx in self.float_keys.items():
            result_expression = input_vectoriser._to_expression_builder(matrix[idx], False)
            a, b = self._forward_scale_map(key)
            result_expression.shift_by(-b)
            result_expression.scale_by(1 / a)
            if key in result:
                result[key].value = result_expression
            else:
                result[key] = result_expression

        return [(key, expression.to_code()) for key, expression in result.items()]

    def get_args(self):
        return ", ".join(self.float_keys.keys())
