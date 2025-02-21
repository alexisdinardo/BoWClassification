#!/bin/sh
python convert_A06.py

echo Test01: compute_SPM_repr  TEST01
pytest Autograder_a06.py::test01

echo Test02: compute_SPM_repr  TEST02
pytest Autograder_a06.py::test02

echo Test03: compute_SPM_repr  TEST03
pytest Autograder_a06.py::test03

echo Test04: compute_SPM_repr  TEST04
pytest Autograder_a06.py::test04
