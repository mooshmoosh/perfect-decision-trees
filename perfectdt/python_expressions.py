import itertools
from dataclasses import dataclass


class BaseLinearTerm:
    @property
    def sign(self):
        return "+" if self.coefficient >= 0 else "-"

    @property
    def display_coefficient(self):
        if self.coefficient in [1, -1]:
            return ""
        else:
            return f"{abs(self.coefficient)} * "

    def is_boolean(self):
        return False


@dataclass
class NullCheckTerm(BaseLinearTerm):
    var_name: str
    coefficient: float

    def to_code(self):
        if self.display_coefficient == "":
            return f"{self.var_name} is None"
        return f"{self.display_coefficient}({self.var_name} is None)"

    def as_negative_boolean(self):
        return f"{self.var_name} is not None"

    def is_boolean(self):
        return self.coefficient in [1, -1]


@dataclass
class LinearTerm(BaseLinearTerm):
    var_name: str
    coefficient: float

    def to_code(self):
        return f"{self.display_coefficient}{self.var_name}"


@dataclass
class NullableVariableTerm(BaseLinearTerm):
    var_name: str
    coefficient: float
    null_coefficient: float

    def is_boolean(self):
        return self.null_coefficient in [1, -1] and self.coefficient == 0

    def to_code(self):
        cf = self.display_coefficient
        if self.null_coefficient == 0:
            if cf == "":
                return f"{self.var_name} or 0"
            else:
                return f"{self.display_coefficient}({self.var_name} or 0)"
        elif self.null_coefficient == self.coefficient:
            if cf == "":
                return f"{self.var_name} or 1"
            else:
                return f"{self.display_coefficient}({self.var_name} or 1)"
        elif self.null_coefficient == -self.coefficient:
            if cf == "":
                return f"{self.var_name} or -1"
            else:
                return f"{self.display_coefficient}({self.var_name} or -1)"
        elif self.sign == "+":
            return (
                f"{self.display_coefficient}{self.var_name} if {self.var_name} is not None else {self.null_coefficient}"
            )
        else:
            return f"{self.display_coefficient}{self.var_name} if {self.var_name} is not None else {-self.null_coefficient}"


class ExpressionBuilder:
    def __init__(self):
        self.coefficients = {}
        self.null_coefficients = {}
        self.all_keys = set()
        self.constant = 0
        self.tolerance = 14

    def add_coefficient(self, key, value):
        if value != 0:
            self.coefficients[key] = round(value, self.tolerance)
            self.all_keys.add(key)

    def add_is_null_coefficient(self, key, value):
        if value != 0:
            self.null_coefficients[key] = round(value, self.tolerance)
            self.all_keys.add(key)

    def add_constant_term(self, value):
        self.constant += round(value, self.tolerance)

    def shift_by(self, value):
        self.constant += round(value, self.tolerance)

    def scale_by(self, value):
        if value == 0:
            self.coefficients = {}
            self.null_coefficients = {}
            self.all_keys = set()
            self.constant = 0
            return

        for key in self.coefficients.keys():
            self.coefficients[key] *= value
        for key in self.null_coefficients.keys():
            self.null_coefficients[key] *= value
        self.constant *= value

    def _terms(self):
        result = []
        for key in self.all_keys:
            if key in self.null_coefficients and key not in self.coefficients:
                # The expression only depends on whether the variable is null
                result.append(NullCheckTerm(key, self.null_coefficients[key]))
            elif key in self.null_coefficients and key in self.coefficients:
                # The expression depends on whether the variable is null, or the value when not
                result.append(NullableVariableTerm(key, self.coefficients[key], self.null_coefficients[key]))
            else:
                # We don't care about the variable being null, only its value
                if key in self.nullable_keys:
                    # But if the value can be null, we need to guard it in the final expression
                    result.append(NullableVariableTerm(key, self.coefficients[key], 0))
                else:
                    result.append(LinearTerm(key, self.coefficients[key]))
        return result

    def to_code(self):
        terms = self._terms()
        if len(terms) == 0:
            if self.constant == 0:
                return "0"
            else:
                return str(self.constant)

        result = []
        if terms[0].sign == "-":
            result.append("-")
        result.append(terms[0].to_code())
        for term in terms[1:]:
            result.append(f" {term.sign} {term.to_code()}")

        if self.constant > 0:
            result.append(f" + {self.constant}")
        elif self.constant < 0:
            result.append(f" - {-self.constant}")
        return "".join(result)


class BooleanExpressionBuilder(ExpressionBuilder):
    def as_boolean_expression(self, terms):
        positives = sum(term.sign == "+" for term in terms)
        negatives = len(terms) - positives
        # LHS = positive terms - negative terms + constant
        # each term is 1 or 0
        # so minimum value is when all positive terms are 0 and all negatives are 1 (-negative + constant)
        # maximum value is when all positives are 1 and all negatives are 0 (positive + constant)
        # if constant is such that its only true when all positives are 1 and all negatives are 0 we can turn it into an AND
        # if constant is such that its true when at least one positive is true or at least one negative is 0 we can turn it into an OR

        if -negatives + self.constant >= 0:
            # if the minimum possible value is already greater than or equal to zero, the condition will always be true
            return "True"

        if positives + self.constant < 0:
            # if the maximum possible value is already less than zero, the condition will never be true
            return "False"

        if -negatives + self.constant < 0 and -negatives + self.constant + 1 >= 0:
            # Minimum possible value + 1 or greater will evaluate to true.
            # This is the same as any of the positive terms or at least one negative term being 0
            return " or ".join([term.to_code() if term.sign == "+" else term.as_negative_boolean() for term in terms])

        if positives + self.constant >= 0 and positives + self.constant - 1 < 0:
            # only maximum possible value will evaluate to true.
            # This is the same as and of the positive terms and none of the negative terms
            return " and ".join([term.to_code() if term.sign == "+" else term.as_negative_boolean() for term in terms])

        result = [
            " + ".join(term.to_code() for term in terms if term.sign == "+"),
            " >= ",
            " + ".join(term.as_negative_boolean() for term in terms if term.sign == "-"),
        ]

        # edge case where one of the sides of the inequality is empty, it should display as 0, not an empty string.
        if result[0] == "":
            result[0] = f"{self.constant}"
        else:
            result[0] = f"{self.constant} + {result[0]}"

        if result[2] == "":
            result[2] = "0"

        return "".join(result)

    def to_code(self):
        if len(self.coefficients) + len(self.null_coefficients) == 0:
            return "True"
        coef_counts = {}
        for coef in itertools.chain(self.coefficients.values(), self.null_coefficients.values()):
            abs_coef = abs(coef)
            if abs_coef not in coef_counts:
                coef_counts[abs_coef] = 1
            else:
                coef_counts[abs_coef] += 1

        self.scale_by(1 / sorted(coef_counts.items(), key=lambda x: -x[1])[0][0])

        terms = self._terms()
        if all(term.is_boolean() for term in terms):
            return self.as_boolean_expression(terms)

        lhs = []
        rhs = []
        for term in terms:
            if term.sign == "-":
                rhs.append(term.to_code())
            elif term.sign == "+":
                lhs.append(term.to_code())
        if self.constant == 0:
            # This condition is true if self.constant == -0.0, the second zero in floating point.
            # This means we always display 0 as 0, and never as 0.0 or -0.0
            self.constant = 0
        if len(lhs) == 0:
            lhs.append(str(self.constant))
        elif len(rhs) == 0:
            rhs.append(str(-self.constant))
        elif self.constant > 0:
            lhs.append(str(self.constant))
        elif self.constant < 0:
            rhs.append(str(self.constant))

        return " >= ".join([" + ".join(lhs), " + ".join(rhs)])


class NullableExpression:
    def __init__(self, condition):
        self.condition = condition
        self.value = ExpressionBuilder()
        self._condition_inverted = False

    def to_code(self):
        if not self._condition_inverted:
            self._condition_inverted = True
            self.condition.scale_by(-1)
        condition_code = self.condition.to_code()
        if condition_code == "True":
            return f"{self.value.to_code()}"
        else:
            return f"({self.value.to_code()} if {condition_code} else None)"
