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


def test_find_the_one_thats_not_null():
    """
    This one shows that we need to improve turning expressions involving "is None" into nicer
    expressions. Its correct, but not easy to read.

        x is not None

    is becoming

        0 >= int(y is None)

    Correct, but not nice.
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
  if 0 >= int(x is None):
    return {
      "z": (x or 0),
    }
  else:
    return {
      "z": ((y or 0) if 0 >= int(y is None) else None),
    }
"""
    )
