import itertools


class ExpressionBuilder:
    def __init__(self):
        self.coefficients = {}
        self.null_coefficients = {}
        self.all_keys = set()
        self.constant = 0

    def add_coefficient(self, key, value):
        if value != 0:
            self.coefficients[key] = value
            self.all_keys.add(key)

    def add_is_null_coefficient(self, key, value):
        if value != 0:
            self.null_coefficients[key] = value
            self.all_keys.add(key)

    def add_constant_term(self, value):
        self.constant += value

    def shift_by(self, value):
        self.constant += value

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
                if self.null_coefficients[key] == 1:
                    result.append(("+", f"int({key} is None)"))
                elif self.null_coefficients[key] == -1:
                    result.append(("-", f"int({key} is None)"))
                elif self.null_coefficients[key] > 0:
                    result.append(("+", f"({self.null_coefficients[key]} if {key} is None else 0)"))
                else:
                    result.append(("-", f"({-self.null_coefficients[key]} if {key} is None else 0)"))
            elif key in self.null_coefficients and key in self.coefficients:
                # The expression depends on whether the variable is null, or the value when not
                if self.null_coefficients[key] > 0:
                    result.append(
                        ("+", f"({self.null_coefficients[key]} if {key} is None else {self.coefficients[key]})")
                    )
                else:
                    result.append(
                        ("-", f"({-self.null_coefficients[key]} if {key} is None else {-self.coefficients[key]})")
                    )
            else:
                # We don't care about the variable being null, only its value
                if self.coefficients[key] == 1:
                    if key in self.nullable_keys:
                        # But if the value can be null, we need to guard it with an or
                        result.append(("+", f"({key} or 0)"))
                    else:
                        result.append(("+", key))
                elif self.coefficients[key] == -1:
                    if key in self.nullable_keys:
                        # But if the value can be null, we need to guard it with an or
                        result.append(("-", f"({key} or 0)"))
                    else:
                        result.append(("-", f"{key}"))
                else:
                    if key in self.nullable_keys:
                        if self.coefficients[key] > 0:
                            result.append(("+", f"({self.coefficients[key]} * {key} if {key} is not None else 0)"))
                        else:
                            result.append(("-", f"({-self.coefficients[key]} * {key} if {key} is not None else 0)"))
                    else:
                        if self.coefficients[key] > 0:
                            result.append(("+", f"{self.coefficients[key]} * {key}"))
                        else:
                            result.append(("-", f"{-self.coefficients[key]} * {key}"))
        return result

    def to_code(self):
        terms = self._terms()
        if len(terms) == 0:
            return "0"

        result = []
        if terms[0][0] == "-":
            result.append("-")
        result.append(terms[0][1])
        for sign, term in terms[1:]:
            result.append(f" {sign} {term}")

        if self.constant > 0:
            result.append(f" + {self.constant}")
        elif self.constant < 0:
            result.append(f" - {-self.constant}")
        return "".join(result)


class BooleanExpressionBuilder(ExpressionBuilder):
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

        lhs = []
        rhs = []
        for sign, term in terms:
            if sign == "-":
                rhs.append(term)
            elif sign == "+":
                lhs.append(term)
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
        self.condition_inverted = False

    def to_code(self):
        if not self.condition_inverted:
            self.condition_inverted = True
            self.condition.scale_by(-1)
        condition_code = self.condition.to_code()
        if condition_code == "True":
            return f"{self.value.to_code()}"
        else:
            return f"({self.value.to_code()} if {condition_code} else None)"
