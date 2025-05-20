#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include<math.h>

// Structure to represent a matrix
typedef struct
{
    int rows;
    int cols;
    double **data;
} Matrix;

// Function to initialize a matrix with given dimensions
Matrix* createMatrix(int rows, int cols)
{
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    if (mat == NULL) 
    {
        printf("\033[1;31mMemory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    mat->rows = rows;
    mat->cols = cols;
    // Allocate memory for matrix data
    mat->data = (double**)malloc(rows * sizeof(double*));
    if (mat->data == NULL) 
    {
        printf("\033[1;31mMemory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) 
    {
        mat->data[i] = (double*)malloc(cols * sizeof(double));
        if (mat->data[i] == NULL)
        {
            printf("\033[1;31mMemory allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }
    
    return mat;
}
// Function to free memory allocated for a matrix
void freeMatrix(Matrix *mat) 
{
    for (int i = 0; i < mat->rows; i++) 
    {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}
// Function to print a matrix
void printMatrix(Matrix *mat) 
{
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            printf("\033[0;35m");
            printf("%.0f\t", mat->data[i][j]);
        }
        printf("\n");
        printf("\033[0m");
    }
}
// Function to add two matrices
Matrix* addMatrices(Matrix **matrices, int numMatrices) 
{
    // Check if all matrices have the same dimensions
    int rows = matrices[0]->rows;
    int cols = matrices[0]->cols;
    for (int i = 1; i < numMatrices; i++) 
    {
        if (matrices[i]->rows != rows || matrices[i]->cols != cols) 
        {
            printf("\033[1;31mMatrices are not compatible for addition\n");
            return NULL;
        }
    }
    Matrix *result = createMatrix(rows, cols);
    
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
        {
            result->data[i][j] = 0;
            for (int k = 0; k < numMatrices; k++) 
            {
                result->data[i][j] += matrices[k]->data[i][j];
            }
        }
    }
    
    return result;
}
// Function to subtract two matrices
Matrix* subtract(Matrix **matrices, int numMatrices) 
{
    int rows = matrices[0]->rows;
    int cols = matrices[0]->cols;
    for (int i = 1; i < numMatrices; i++)
    {
        if (matrices[i]->rows != rows || matrices[i]->cols != cols) 
        {
            printf("\033[1;31mMatrices are not compatible for subtraction\n");
            return NULL;
        }
    }
    Matrix *result = createMatrix(rows, cols);
    
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
        {
            result->data[i][j] = matrices[0]->data[i][j];
            for (int k = 1; k < numMatrices; k++) 
            {
                result->data[i][j] -= matrices[k]->data[i][j];
            }
        }
    }
    
    return result;
}
// Function to multiply a matrix by a scalar
Matrix* scalarMultiply(Matrix *mat, double scalar)
{
    Matrix *result = createMatrix(mat->rows, mat->cols);
    
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            result->data[i][j] = mat->data[i][j] * scalar;
        }
    }
    
    return result;
}
// Function to multiply two matrices
Matrix* multiply(Matrix *mat1, Matrix *mat2) 
{
    if (mat1->cols != mat2->rows) 
    {
        printf("\033[1;31mMatrices are not compatible for multiplication\n");
        return NULL;
    }
    
    Matrix *result = createMatrix(mat1->rows, mat2->cols);
    
    for (int i = 0; i < mat1->rows; i++) 
    {
        for (int j = 0; j < mat2->cols; j++)
        {
            result->data[i][j] = 0;
            for (int k = 0; k < mat1->cols; k++) 
            {
                result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
            }
        }
    }
    
    return result;
}
// Function to transpose a matrix
Matrix* transpose(Matrix *mat) 
{
    Matrix *result = createMatrix(mat->cols, mat->rows);
    
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            result->data[j][i] = mat->data[i][j];
        }
    }
    
    return result;
}
// Function to calculate the determinant of a matrix
double determinant(Matrix *mat) 
{
    if (mat->rows != mat->cols) 
    {
        printf("\033[1;31mDeterminant can only be calculated for square matrices\n");
        return 0.0;
    }
    int n = mat->rows;
    double det = 0;
    Matrix *subMat = createMatrix(n - 1, n - 1);
    if (n == 1) 
    {
        det = mat->data[0][0];
    } 
    else if (n == 2) 
    {
        det = mat->data[0][0] * mat->data[1][1] - mat->data[0][1] * mat->data[1][0];
    } 
    else
    {
        for (int k = 0; k < n; k++) 
        {
            int subi = 0;
            for (int i = 1; i < n; i++) 
            {
                int subj = 0;
                for (int j = 0; j < n; j++) 
                {
                    if (j == k)
                        continue;
                    subMat->data[subi][subj] = mat->data[i][j];
                    subj++;
                }
                subi++;
            }
            det += (k % 2 == 0 ? 1 : -1) * mat->data[0][k] * determinant(subMat);
        }
    }
    
    freeMatrix(subMat);
    return det;
}
// Function to calculate the inverse of a matrix
Matrix* inverse(Matrix *mat) 
{
    double det = determinant(mat);
    if (det == 0) 
    {
        printf("\033[1;31mInverse does not exist as the determinant is zero\n");
        return NULL;
    }
    
    int n = mat->rows;
    Matrix *invMat = createMatrix(n, n);
    Matrix *subMat = createMatrix(n - 1, n - 1);
    
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            int subi = 0;
            for (int k = 0; k < n; k++) 
            {
                if (k == i)
                    continue;
                int subj = 0;
                for (int l = 0; l < n; l++) 
                {
                    if (l == j)
                        continue;
                    subMat->data[subi][subj] = mat->data[k][l];
                    subj++;
                }
                subi++;
            }
            invMat->data[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * determinant(subMat) / det;
        }
    }
    
    freeMatrix(subMat);
    return invMat;
}
// Function to calculate the adjoint of a matrix
Matrix* adjoint(Matrix *mat) 
{
    int n = mat->rows;
    Matrix *adjMat = createMatrix(n, n);
    Matrix *subMat = createMatrix(n - 1, n - 1);
    
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            int subi = 0;
            for (int k = 0; k < n; k++)
            {
                if (k == i)
                    continue;
                int subj = 0;
                for (int l = 0; l < n; l++) 
                {
                    if (l == j)
                        continue;
                    subMat->data[subi][subj] = mat->data[k][l];
                    subj++;
                }
                subi++;
            }
            adjMat->data[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * determinant(subMat);
        }
    }
    
    freeMatrix(subMat);
    return adjMat;
}

// Function to perform vector multiplication
double vectorMultiply(Matrix *vec1, Matrix *vec2) 
{
    if (vec1->rows != vec2->rows || vec1->cols != 1 || vec2->cols != 1)
    {
        printf("\033[1;31mVectors are not compatible for multiplication\n");
        return 0.0;
    }
    
    double result = 0.0;
    
    for (int i = 0; i < vec1->rows; i++) 
    {
        result += vec1->data[i][0] * vec2->data[i][0];
    }
    
    return result;
}

// Function to check if a matrix is symmetric
bool isSymmetric(Matrix *mat)
{
    if (mat->rows != mat->cols) 
    {
        return false; // Non-square matrices cannot be symmetric
    }
    
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = i + 1; j < mat->cols; j++) 
        {
            if (mat->data[i][j] != mat->data[j][i]) {
                return false;
            }
        }
    }
    
    return true;
}

// Function to check if a matrix is skew symmetric
bool isSkewSymmetric(Matrix *mat) 
{
    if (mat->rows != mat->cols) 
    {
        return false; // Non-square matrices cannot be skew symmetric
    }
    
    for (int i = 0; i < mat->rows; i++) 
    {
        for (int j = i + 1; j < mat->cols; j++)
        {
            if (mat->data[i][j] != -mat->data[j][i]) 
            {
                return false;
            }
        }
    }
    
    return true;
}
// Function to check if a matrix is idempotent
bool isIdempotent(Matrix *mat) 
{
    // Calculate the result of multiplying the matrix by itself
    Matrix *result = multiply(mat, mat); // Pass the address of the matrix and the number of matrices

    bool isIdemp = true;

    // Check if the result matches the original matrix
    for (int i = 0; i < mat->rows; i++) 
    {
        for (int j = 0; j < mat->cols; j++) 
        {
            if (mat->data[i][j] != result->data[i][j]) 
            {
                isIdemp = false;
                break;
            }
        }
    }

    // Free the memory allocated for the result matrix
    freeMatrix(result);

    return isIdemp;
}
// Function to check if a matrix is orthogonal
bool isOrthogonal(Matrix *mat) 
{
    if (mat->rows != mat->cols) 
    {
        return false; // Non-square matrices cannot be orthogonal
    }

    Matrix *transposeMat = transpose(mat);
    Matrix *result = multiply(mat, transposeMat); // Pass the address of the matrix and the number of matrices

    bool isOrtho = true;

    for (int i = 0; i < mat->rows; i++) 
    {
        for (int j = 0; j < mat->cols; j++) 
        {
            if (i == j && result->data[i][j] != 1.0) 
            {
                isOrtho = false;
                break;
            } 
            else if (i != j && result->data[i][j] != 0.0) 
            {
                isOrtho = false;
                break;
            }
        }
    }

    freeMatrix(transposeMat);
    freeMatrix(result);

    return isOrtho;
}
// Function to check if a matrix is involuntary
bool isInvolutary(Matrix *mat) 
{
    Matrix *result = multiply(mat, mat); // Pass the address of the matrix and the number of matrices

    bool isInv = true;

    for (int i = 0; i < mat->rows; i++) 
    {
        for (int j = 0; j < mat->cols; j++) 
        {
            if ((i == j && result->data[i][j] != 1.0) || (i != j && result->data[i][j] != 0.0)) 
            {
                isInv = false;
                break;
            }
        }
    }

    freeMatrix(result);

    return isInv;
}
// Function to calculate eigenvalues and eigenvectors
void eigen(Matrix *mat) 
{
    int n = mat->rows;
    // Initialize the eigenvector matrix with the identity matrix
    Matrix *eigenvectors = createMatrix(n, n);
    for (int i = 0; i < n; i++)
    {
        eigenvectors->data[i][i] = 1.0;
    }
    // Initialize the eigenvalues matrix
    Matrix *eigenvalues = createMatrix(n, 1);
    for (int i = 0; i < n; i++)
    {
        eigenvalues->data[i][0] = 0.0;
    }
    // Set the convergence criteria
    double epsilon = 1e-9;
    int maxIterations = 1000;
    // Power iteration method to find eigenvalues and eigenvectors
    for (int i = 0; i < n; i++) 
    {
        Matrix *x = createMatrix(n, 1); // Initial guess for eigenvector
        x->data[i][0] = 1.0; // Arbitrarily set one component to 1
        double lambda_old = 0.0;
        double lambda_new = 1.0;
        int iterations = 0;
        while (fabs(lambda_new - lambda_old) > epsilon && iterations < maxIterations) 
        {
            lambda_old = lambda_new;
            // Perform matrix-vector multiplication
            Matrix *Ax = multiply(mat, x);
            // Find the norm of the resulting vector
            double norm = 0.0;
            for (int j = 0; j < n; j++) 
            {
                norm += Ax->data[j][0] * Ax->data[j][0];
            }
            norm = sqrt(norm);
            // Normalize the vector
            for (int j = 0; j < n; j++) 
            {
                x->data[j][0] = Ax->data[j][0] / norm;
            }
            // Calculate the eigenvalue
            Matrix *Ax_norm = multiply(mat, x);
            lambda_new = vectorMultiply(x, Ax_norm);
            // Update the eigenvector
            for (int j = 0; j < n; j++) 
            {
                eigenvectors->data[j][i] = x->data[j][0];
            }
            freeMatrix(Ax);
            freeMatrix(Ax_norm);
            iterations++;
        }
        eigenvalues->data[i][0] = lambda_new;
        freeMatrix(x);
    }
    // Print the final results
    printf("Eigenvalues:\n");
    printMatrix(eigenvalues);
    printf("Eigenvectors:\n");
    printMatrix(eigenvectors);

    // Free allocated memory
    freeMatrix(eigenvalues);
    freeMatrix(eigenvectors);
}
// Function to display the menu and get user choice
int getMenuChoice()
{
    int choice;
    printf("\033[1;36m");
    printf("\nMatrix Calculator\n");
    printf("-----------------\n");
    printf("1. Add Matrices\n");
    printf("2. Subtract Matrices\n");
    printf("3. Multiply Matrix by Scalar\n");
    printf("4. Multiply Matrices\n");
    printf("5. Transpose Matrix\n");
    printf("6. Calculate Determinant\n");
    printf("7. Calculate Inverse\n");
    printf("8. Calculate Adjoint\n");
    printf("9. Perform Vector Multiplication\n");
    printf("10. Check Symmetry\n");
    printf("11. Check Skew Symmetry\n");
    printf("12. Check Idempotence\n");
    printf("13. Check Orthogonality\n");
    printf("14. Check Involutiveness\n");
    printf("15. Calculate Eigen Value and Eigen Vector\n");
    printf("16. Exit\n");
    printf("Enter your choice: ");
    printf("\033[0m");
    scanf("%d", &choice);
    return choice;
}
// Function to input a matrix from user
Matrix* inputMatrix()
{
    printf("\033[1;33m");
    int rows, cols;
    printf("Enter number of rows: ");
    scanf("%d", &rows);
    printf("Enter number of columns: ");
    scanf("%d", &cols);
    
    Matrix *mat = createMatrix(rows, cols);
    
    printf("Enter matrix elements:\n");
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
        {
            scanf("%lf", &mat->data[i][j]);
        }
    }
    printf("\033[0m");
    return mat;
}
// Function to handle matrix operations based on user choice
void performOperation(int choice) 
{
    printf("\033[1;33m");
    int numMatrices;
    Matrix **matrices,*mat1,*mat2;
    double scalar;
    
    switch (choice) 
    {
        case 1:
        // Add Matrices
            printf("\nEnter the number of matrices to add: ");
            scanf("%d", &numMatrices);
            matrices = (Matrix**)malloc(numMatrices * sizeof(Matrix*));
            if (matrices == NULL) 
            {
                printf("Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            printf("\nEnter matrices:\n");
            for (int i = 0; i < numMatrices; i++) 
            {
                printf("\nEnter matrix %d:\n", i + 1);
                matrices[i] = inputMatrix();
            }
            Matrix *addResult = addMatrices(matrices, numMatrices);
            if (addResult != NULL) 
            {
                printf("\nResult:\n");
                printMatrix(addResult);
                freeMatrix(addResult);
            }
            for (int i = 0; i < numMatrices; i++) 
            {
                freeMatrix(matrices[i]);
            }
            free(matrices);
            break;
            
        case 2:
        // Subtract Matrices
            printf("\nEnter the number of matrices to subtract: ");
            scanf("%d", &numMatrices);
            matrices = (Matrix**)malloc(numMatrices * sizeof(Matrix*));
            if (matrices == NULL) 
            {
                printf("Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            printf("\nEnter matrices:\n");
            for (int i = 0; i < numMatrices; i++) 
            {
                printf("\nEnter matrix %d:\n", i + 1);
                matrices[i] = inputMatrix();
            }
            Matrix *subResult = subtract(matrices, numMatrices);
            if (subResult != NULL) 
            {
                printf("\nResult:\n");
                printMatrix(subResult);
                freeMatrix(subResult);
            }
            for (int i = 0; i < numMatrices; i++) 
            {
                freeMatrix(matrices[i]);
            }
            free(matrices);
            break;
            
        case 3: 
        // Multiply Matrix by Scalar
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nEnter scalar value: ");
            scanf("%lf", &scalar);
            Matrix* result = scalarMultiply(mat1, scalar);
            printf("\nResult:\n");
            printMatrix(result);
            freeMatrix(mat1);
            freeMatrix(result);
            break;
            
        case 4:
        // Multiply Matrices
            printf("\nEnter first matrix:\n");
            mat1 = inputMatrix();
            printf("\nEnter second matrix:\n");
            mat2 = inputMatrix();
            result = multiply(mat1, mat2);
            if (result != NULL) {
                printf("\nResult:\n");
                printMatrix(result);
                freeMatrix(result);
            }
            freeMatrix(mat1);
            freeMatrix(mat2);
            break;
            
        case 5:
        // Transpose Matrix
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            result = transpose(mat1);
            printf("\nResult:\n");
            printMatrix(result);
            freeMatrix(mat1);
            freeMatrix(result);
            break;
            
        case 6: 
        // Calculate Determinant
            // Implementation of determinant function
            mat1 = inputMatrix();
            printf("\nDeterminant: %.2f\n", determinant(mat1));
            freeMatrix(mat1);
            break;
            
        case 7: 
        // Calculate Inverse
            // Implementation of inverse function
            mat1 = inputMatrix();
            result = inverse(mat1);
            if (result != NULL) {
                printf("\nInverse:\n");
                printMatrix(result);
                freeMatrix(result);
            }
            freeMatrix(mat1);
            break;
            
        case 8: 
        // Calculate Adjoint
            // Implementation of adjoint function
            
            mat1 = inputMatrix();
            result = adjoint(mat1);
            if (result != NULL) {
                printf("\nAdjoint:\n");
                printMatrix(result);
                freeMatrix(result);
            }
            freeMatrix(mat1);
            break;
            
        case 9: 
        // Perform Vector Multiplication
            printf("\nEnter first vector:\n");
            mat1 = inputMatrix();
            printf("\nEnter second vector:\n");
            mat2 = inputMatrix();
            printf("\nVector multiplication result: %.2f\n", vectorMultiply(mat1, mat2));
            freeMatrix(mat1);
            freeMatrix(mat2);
            break;
            
        case 10: 
        // Check Symmetry
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nMatrix is %s\n", isSymmetric(mat1) ? "\033[0;32msymmetric" : "\033[1;31mnot symmetric");
            freeMatrix(mat1);
            break;
            
        case 11: 
        // Check Skew Symmetry
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nMatrix is %s\n", isSkewSymmetric(mat1) ? "\033[0;32mskew symmetric" : "\033[1;31mnot skew symmetric");
            freeMatrix(mat1);
            break;
            
        case 12: 
        // Check Idempotence
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nMatrix is %s\n", isIdempotent(mat1) ? "\033[0;32midempotent" : "\033[1;31mnot idempotent");
            freeMatrix(mat1);
            break;
            
        case 13: 
        // Check Orthogonality
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nMatrix is %s\n", isOrthogonal(mat1) ? "\033[0;32morthogonal" : "\033[1;31mnot orthogonal");
            freeMatrix(mat1);
            break;
            
        case 14: 
        // Check Involutiveness
            printf("\nEnter matrix:\n");
            mat1 = inputMatrix();
            printf("\nMatrix is %s\n", isInvolutary(mat1) ? "\033[0;32minvolutary" : "\033[1;31mnot involutary");
            freeMatrix(mat1);
            break;
        case 15:
        // Calculate Eigenvalues and Eigenvectors
        printf("\nEnter matrix:\n");
        mat1 = inputMatrix();
        eigen(mat1);
        freeMatrix(mat1);
        break;
            
        case 16: 
        // Exit
            printf("\n\033[1;31mExiting program\n");
            exit(EXIT_SUCCESS);
            
        default:
            printf("\n\033[1;31mInvalid choice\n");
    }
}
int main() {
    while (1) {
        int choice = getMenuChoice();
        performOperation(choice);
    }
    
    return 0;
}
