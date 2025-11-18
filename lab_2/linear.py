import argparse
import sys
import numpy as np
import tensorflow as tf

def rotate_point(point: list, angle_degrees: float) -> tf.Tensor:
    """
    Rotates 2D point (vector) around (0,0) by given angle
    """
    angle_radians = tf.cast(angle_degrees * (np.pi / 180.00), dtype=tf.float32)
    
    # Fixed: applied radians to cos as well
    cos_a = tf.cos(angle_radians)
    sin_a = tf.sin(angle_radians)
    
    rotation_matrix = tf.convert_to_tensor([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=tf.float32)
    
    point_tensor = tf.convert_to_tensor(point, dtype=tf.float32)
    if len(point_tensor.shape) == 1:
        point_tensor = tf.expand_dims(point_tensor, 1)

    new_point = tf.matmul(rotation_matrix, point_tensor)

    return tf.squeeze(new_point)

@tf.function
def solve_linear_tf_func(A_tf: tf.Tensor, b_tf: tf.Tensor) -> tf.Tensor:
    solution = tf.linalg.solve(A_tf, b_tf)
    return tf.squeeze(solution)

def check_dim_and_solve(matrix_a: np.ndarray, vector_b: np.ndarray) -> tf.Tensor | None:
    A = tf.convert_to_tensor(matrix_a, dtype=tf.float32)
    b = tf.convert_to_tensor(vector_b, dtype=tf.float32)
    
    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        print(
            f"Error: Mismatched dimensions. A: {A.shape}, b: {b.shape}",
            file=sys.stderr
        )
        return None

    det_A = tf.linalg.det(A)
    
    if tf.abs(det_A) < 1e-6:
        print(
            f"Error: Matrix A is singular (determinant = {det_A:.2f}).",
            file=sys.stderr
        )
        return None
    
    b_exp = tf.expand_dims(b, 1)
    solution = solve_linear_tf_func(A, b_exp)
    return solution

def parse_system_arguments(data: list) -> tuple[np.ndarray, np.ndarray] | None:
    n_args = len(data)
    
    delta = 1 + 4 * n_args
    if delta < 0:
        return None 

    sqrt_delta = np.sqrt(delta)
    if sqrt_delta % 1 != 0:
        return None 

    n = int((-1 + sqrt_delta) / 2)
    
    if n * (n + 1) != n_args:
        return None

    print(f"Detected a {n}x{n} equation system.")

    try:
        elements_A = data[:n*n]
        elements_b = data[n*n:]
        
        matrix_A = np.array(elements_A).reshape((n, n))
        vector_b = np.array(elements_b)
        
        return matrix_A, vector_b
    except ValueError as e:
        print(f"Error during data reshaping: {e}", file=sys.stderr)
        return None

def main():
    tf.get_logger().setLevel('ERROR')

    parser = argparse.ArgumentParser(description="Lab 2 - Tensor Operations.")
    
    subparsers = parser.add_subparsers(
        dest='task', 
        required=True, 
        help='Select the task to run: "rotate" or "solve"'
    )

    parser_rotate = subparsers.add_parser('rotate', help='Rotates a point.')
    parser_rotate.add_argument('--point', nargs=2, type=float, required=True)
    parser_rotate.add_argument('--angle', type=float, required=True)

    parser_solve = subparsers.add_parser('solve', help='Solves Ax = b.')
    parser_solve.add_argument('--data', nargs='+', type=float, required=True)

    args = parser.parse_args()

    if args.task == 'rotate':
        start_point = args.point
        angle = args.angle
        
        print(f"Rotating point: {start_point}")
        print(f"By angle: {angle} degrees")
        
        new_point = rotate_point(start_point, angle)
        
        print("--- Result ---")
        print(f"New point: {new_point.numpy()}")
        
        print("\n--- Automatic Test ---")
        test_p = [1.0, 0.0]
        test_a = 90.0
        test_result = rotate_point(test_p, test_a)
        expected_result = np.array([0.0, 1.0])
        
        try:
            assert np.allclose(test_result.numpy(), expected_result, atol=1e-6)
            print("Test (rotate [1,0] by 90 deg -> [0,1]) PASSED.")
        except AssertionError:
            print(f"Test FAILED. Expected {expected_result}, got {test_result.numpy()}")

    elif args.task == 'solve':
        parsing_result = parse_system_arguments(args.data)
        
        if parsing_result is None:
            print("Error: Invalid number of arguments.", file=sys.stderr)
            sys.exit(1)
            
        matrix_A, vector_b = parsing_result

        print("Matrix A:")
        print(matrix_A)
        print("\nVector b:")
        print(vector_b)

        # Fixed: Correct function name used here
        solution = check_dim_and_solve(matrix_A, vector_b)
        
        if solution is not None:
            print("\n--- Result ---")
            print(f"Solution 'x': {solution.numpy()}")
            
            print("\n--- Automatic Test (Verifying A*x = b) ---")
            try:
                test_result = np.dot(matrix_A, solution.numpy())
                assert np.allclose(test_result, vector_b, atol=1e-5)
                print(f"Verification (A * x): {test_result}")
                print("Test (A*x = b) PASSED.")
            except AssertionError:
                print("Test (A*x = b) FAILED.")

if __name__ == "__main__":
    main()