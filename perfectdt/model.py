from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
import pulp
from .vectoriser import Vectoriser


class Model:
    def fit(self, inputs, outputs, regularise="l1"):
        self.regularise = regularise
        self._vectorise_data(inputs, outputs)
        self._train_model()

    def _vectorise_data(self, inputs, outputs):
        self.input_vectoriser = Vectoriser()
        self.output_vectoriser = Vectoriser(include_constant=False)
        self.input_vectoriser.fit(inputs)
        self.output_vectoriser.fit(outputs)
        self.trained_inputs = self.input_vectoriser.to_vectors(inputs)
        self.trained_outputs = self.output_vectoriser.to_vectors(outputs)

    def predict(self, inputs):
        result = []
        inputs = self.input_vectoriser.to_vector(inputs)
        node = self.root_node
        while getattr(node, "linear_map", None) is None:
            if (node.condition_map @ inputs).sum() >= 0:
                node = node.greater
            else:
                node = node.less
        return self.output_vectoriser.from_vector(node.linear_map @ inputs)

    def _train_model(self):
        depth = 1
        while not self._train_model_at_depth(depth):
            depth += 1

    def _make_problem(self, depth):
        problem = pulp.LpProblem()
        self.root_node = self._build_tree(depth, "root")
        self.root_node.gather_constraints(problem, self.trained_inputs, self.trained_outputs)
        if self.regularise == "l1":
            constraints, objective = self.root_node.gather_objective()
            for constraint in constraints:
                problem += constraint
            problem += objective
        return problem

    def _train_model_at_depth(self, depth):
        problem = self._make_problem(depth)
        if problem.solve() > 0:
            self.root_node.make_maps()
            return True
        return False

    def _build_tree(self, depth, name):
        input_width = self.trained_inputs.shape[1]
        input_length = self.trained_inputs.shape[0]
        output_width = self.trained_outputs.shape[1]
        if depth == 1:
            return LeafNode(
                name,
                np.zeros((output_width, input_width)),
                [[pulp.LpVariable(f"{name}-{i}:{j}") for j in range(input_width)] for i in range(output_width)],
            )
        else:
            greater = self._build_tree(depth - 1, f"{name}-greater")
            less = self._build_tree(depth - 1, f"{name}-less")
            return InternalNode(
                name,
                greater,
                less,
                np.zeros(input_width),
                [pulp.LpVariable(f"{name}-{j}") for j in range(input_width)],
                [
                    pulp.LpVariable(f"{name}-choice({i})", cat="Integer", lowBound=0, upBound=1)
                    for i in range(input_length)
                ],
            )

    def to_python_code(self, function_name, indent=0, node=None):
        if node == None:
            args = self.input_vectoriser.get_args()
            return "\n".join(
                [f"def {function_name}({args}):"]
                + self.to_python_code(function_name, indent=indent + 1, node=self.root_node)
            )
        elif hasattr(node, "linear_map"):
            result = ["  " * indent + "return {"]
            output_expressions = self.output_vectoriser.to_mapped_expressions(node.linear_map, self.input_vectoriser)
            for var_name, expression in output_expressions:
                result.append("  " * (indent + 1) + f'"{var_name}": {expression},')
            result.append("  " * indent + "}")
        else:
            condition_expression = self.input_vectoriser.to_expression(node.condition_map)
            result = ["  " * indent + f"if {condition_expression}:"]
            result += self.to_python_code(function_name, indent=indent + 1, node=node.greater)
            result += ["  " * indent + f"else:"]
            result += self.to_python_code(function_name, indent=indent + 1, node=node.less)
        return result


@dataclass
class LeafNode:
    name: str
    linear_map: np.ndarray
    map_variables: list[list[pulp.LpVariable]]

    def gather_constraints(self, problem, inputs, outputs, choice_vars=None):
        if choice_vars is None:
            choice_vars = []

        # for all input/output pairs make sure the map times input - output is between lower_bound and upper bound
        for choice_idx, (input_row, output_row) in enumerate(zip(inputs, outputs)):
            # if all the choice vars are 0 then upper_bound = lower_bound = 0
            # if any are non zero then upper_bound >= 2 and lower_bound <= -2
            upper_bound = 0
            lower_bound = 0
            for choice_var_list in choice_vars:
                upper_bound += 2 * choice_var_list[choice_idx]
                lower_bound += -2 * choice_var_list[choice_idx]
            for output_map, output_value in zip(self.map_variables, output_row):
                mapped_output = 0
                for input_coeficient, input_value in zip(output_map, input_row):
                    mapped_output += input_coeficient * input_value
                constraint = mapped_output - output_value >= lower_bound
                problem += constraint
                constraint = mapped_output - output_value <= upper_bound
                problem += constraint

    def gather_objective(self):
        constraints = []
        result = 0
        for i, var_list in enumerate(self.map_variables):
            for j, var in enumerate(var_list):
                abs_value_var = pulp.LpVariable(f"{self.name}-obj-{i}:{j}")
                # creating a variable that must be greater than a map_variable and the negative of that map
                # variable means we've created a variable that is always greater than the absolute value
                # of the map variable. Minimizing the sum of these extra variables means we're minimizing the
                # L1 norm of our coefficients.
                constraints.append(var <= abs_value_var)
                constraints.append(-var <= abs_value_var)
                result += abs_value_var
        return constraints, result

    def make_maps(self):
        for i, row in enumerate(self.map_variables):
            for j, v in enumerate(row):
                self.linear_map[i, j] = v.varValue


@dataclass
class InternalNode:
    name: str
    greater: Union["InternalNode", LeafNode]
    less: Union["InternalNode", LeafNode]
    condition_map: np.ndarray
    map_variables: list[pulp.LpVariable]
    choice_variables: Optional[list[pulp.LpVariable]]

    def gather_constraints(self, problem, inputs, outputs, choice_vars=None):
        if choice_vars is None:
            choice_vars = []
        self.greater.gather_constraints(problem, inputs, outputs, choice_vars + [self.choice_variables])
        self.less.gather_constraints(problem, inputs, outputs, choice_vars + [[1 - x for x in self.choice_variables]])
        for input_row, choice_var in zip(inputs, self.choice_variables):
            map_sum = 0
            for x, map_var in zip(input_row, self.map_variables):
                map_sum += x * map_var
            # choice_variable being 0 <=> map_sum is between 0 and 2
            # choice_variable being 1 <=> map_sum is between -2 and -0.1
            constraint = map_sum <= (2 - 2.1 * choice_var)
            problem += constraint
            constraint = map_sum >= (-2 * choice_var)
            problem += constraint

    def gather_objective(self):
        constraints, result = self.greater.gather_objective()
        sub_constraints, sub_objective = self.less.gather_objective()
        constraints += sub_constraints
        result += sub_objective
        for idx, var in enumerate(self.map_variables):
            abs_value_var = pulp.LpVariable(f"{self.name}-obj-{idx}")
            # creating a variable that must be greater than a map_variable and the negative of that map
            # variable means we've created a variable that is always greater than the absolute value
            # of the map variable. Minimizing the sum of these extra variables means we're minimizing the
            # L1 norm of our coefficients.
            constraints.append(var <= abs_value_var)
            constraints.append(-var <= abs_value_var)
            result += abs_value_var
        return constraints, result

    def make_maps(self):
        self.greater.make_maps()
        self.less.make_maps()
        for i, v in enumerate(self.map_variables):
            self.condition_map[i] = v.varValue
