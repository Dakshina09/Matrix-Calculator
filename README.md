# Matrix Calculator

## Overview

This is a C program that functions as a comprehensive matrix calculator. It supports a wide range of matrix operations, including addition, subtraction, multiplication, scalar multiplication, transpose, determinant, inverse, adjoint, vector multiplication, and checks for various matrix properties. The program features a menu-driven interface and uses colored terminal output for better readability.

## Features

- **Add, Subtract, and Multiply Matrices**
- **Scalar Multiplication**
- **Transpose of a Matrix**
- **Determinant, Inverse, and Adjoint Calculation**
- **Vector Multiplication**
- **Check Matrix Properties:**
  - Symmetric
  - Skew Symmetric
  - Idempotent
  - Orthogonal
  - Involutary
- **Eigenvalues and Eigenvectors Calculation** (using power iteration method)
- **User-friendly menu with colored output**

## How to Compile

Use GCC to compile the program:
```sh
gcc "Matrix Calculator.c" -o matrix_calculator -lm
```
The `-lm` flag links the math library.

## How to Run

```sh
./matrix_calculator
```
or on Windows:
```sh
matrix_calculator.exe
```

## Usage

1. Run the program.
2. Select an operation from the menu by entering its number.
3. Enter the required matrix/matrices or scalar value as prompted.
4. View the result and continue with further operations or exit.

## Example Menu

```
Matrix Calculator
-----------------
1. Add Matrices
2. Subtract Matrices
3. Multiply Matrix by Scalar
4. Multiply Matrices
5. Transpose Matrix
6. Calculate Determinant
7. Calculate Inverse
8. Calculate Adjoint
9. Perform Vector Multiplication
10. Check Symmetry
11. Check Skew Symmetry
12. Check Idempotence
13. Check Orthogonality
14. Check Involutiveness
15. Calculate Eigen Value and Eigen Vector
16. Exit
Enter your choice:
```

## Notes

- Only square matrices are valid for determinant, inverse, adjoint, and property checks.
- Eigenvalue/eigenvector calculation uses a simple power iteration method and may not work for all matrices.
- The program uses ANSI escape codes for colored output, which may not display correctly on all terminals.

---
