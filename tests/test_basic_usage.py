import pytest
from perfectdt import Model


def test_find_relu():
    inputs = [{"x": -2}, {"x": -1}, {"x": 0}, {"x": 1}, {"x": 2}]
    outputs = [{"y": max(0, value["x"])} for value in inputs]

    model = Model()
    model.fit(inputs, outputs)

    assert (
        "\n" + model.to_python_code("relu") + "\n"
        == """
def relu(x):
  if x >= 0:
    return {
      "y": x,
    }
  else:
    return {
      "y": 0,
    }
"""
    )


def test_find_max():
    inputs = [
        {"x": -3, "y": -3},
        {"x": -3, "y": -2},
        {"x": -3, "y": -1},
        {"x": -3, "y": 0},
        {"x": -3, "y": 1},
        {"x": -3, "y": 2},
        {"x": -2, "y": -3},
        {"x": -2, "y": -2},
        {"x": -2, "y": -1},
        {"x": -2, "y": 0},
        {"x": -2, "y": 1},
        {"x": -2, "y": 2},
        {"x": -1, "y": -3},
        {"x": -1, "y": -2},
        {"x": -1, "y": -1},
        {"x": -1, "y": 0},
        {"x": -1, "y": 1},
        {"x": -1, "y": 2},
        {"x": 0, "y": -3},
        {"x": 0, "y": -2},
        {"x": 0, "y": -1},
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 0, "y": 2},
        {"x": 1, "y": -3},
        {"x": 1, "y": -2},
        {"x": 1, "y": -1},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 2, "y": -3},
        {"x": 2, "y": -2},
        {"x": 2, "y": -1},
        {"x": 2, "y": 0},
        {"x": 2, "y": 1},
        {"x": 2, "y": 2},
    ]
    outputs = [{"z": max(value["y"], value["x"])} for value in inputs]

    model = Model()
    model.fit(inputs, outputs)

    assert (
        "\n" + model.to_python_code("new_max") + "\n"
        == """
def new_max(x, y):
  if x >= y:
    return {
      "z": x,
    }
  else:
    return {
      "z": y,
    }
"""
    )


def test_piecewise_constant_function():
    """
    This test highlights two bugs discovered and fixed.

    When the leaf node of a model was a constant the python code generated would be "0", not the
    constant.

    Also previously some coefficients would be 0.99999999999 in the solution. We have a tolerance
    now to round these to 1.
    """
    inputs = [{"x": i} for i in [0, 1, 2, 3]]
    outputs = [{"y": i} for i in [-1, -1, 1, 1]]

    model = Model()
    model.fit(inputs, outputs)

    assert (
        "\n" + model.to_python_code("step") + "\n"
        == """
def step(x):
  if x >= 2.0:
    return {
      "y": 1.0,
    }
  else:
    return {
      "y": -1.0,
    }
"""
    )


def test_find_the_one_thats_not_null():
    """
    This one shows that we need to improve expressions underneath boolean expressions.

    x or 0 is a guard to convert x being None into 0

    but in this example the expression is only evaluated when the if statement "x is not None" is
    True, meaning the guard is unnecessary.

    (y or 0 if y is not None else None)

    is the same as y on its own

    This example is correct, but could be improved.
    """
    options = [0, 1, 2, None]
    inputs = [{"x": i, "y": j} for i in options for j in options]
    outputs = [{"z": d["x"] if d["x"] is not None else d["y"]} for d in inputs]

    model = Model()
    model.fit(inputs, outputs)

    assert (
        "\n" + model.to_python_code("x_or_y") + "\n"
        == """
def x_or_y(x, y):
  if x is not None:
    return {
      "z": x or 0,
    }
  else:
    return {
      "z": (y or 0 if y is not None else None),
    }
"""
    )


def test_if_both_inputs_are_none():
    """
    This deomstrates an improvement to handling is None checks.

    The model that is found is ultimately a bunch of linear functions, turning these into boolean
    expressions involves figuring out if all the expressions have to be true or if at least one of
    them has to be true.
    """
    options = [0, 1, None]
    inputs = [{"x": i, "y": j} for i in options for j in options]
    outputs = [{"z": 10 if d["x"] is None and d["y"] is None else 0} for d in inputs]

    model = Model()
    model.fit(inputs, outputs)

    assert "\n" + model.to_python_code("x_and_y_are_None") + "\n" in [
        """
def x_and_y_are_None(x, y):
  if y is not None or x is not None:
    return {
      "z": 0,
    }
  else:
    return {
      "z": 10.0,
    }
""",
        """
def x_and_y_are_None(x, y):
  if x is not None or y is not None:
    return {
      "z": 0,
    }
  else:
    return {
      "z": 10.0,
    }
""",
    ]
