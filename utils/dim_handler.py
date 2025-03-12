import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dim_mismatch_handler')


class DimensionHandler:
    """
    A utility class for handling dimension mismatches in array operations.
    Provides functions to verify and fix dimensions across multi-civilization simulations.
    """

    def __init__(self, verbose=True, auto_fix=True, log_level=logging.INFO):
        """
        Initialize the dimension handler.

        Args:
            verbose (bool): Whether to print dimension mismatch messages
            auto_fix (bool): Whether to automatically fix dimension mismatches
            log_level (int): Logging level for messages
        """
        self.verbose = verbose
        self.auto_fix = auto_fix
        self.fixed_count = 0
        self.warning_count = 0
        self.error_count = 0

        # Set logger level
        logger.setLevel(log_level)

    def verify_dimensions(self, arrays, expected_shapes, context=""):
        """
        Verify that arrays have the expected shapes.

        Args:
            arrays (dict): Dictionary of arrays to verify
            expected_shapes (dict): Dictionary of expected shapes
            context (str): Context information for error messages

        Returns:
            bool: True if all dimensions match expectations, False otherwise
        """
        all_match = True

        for name, array in arrays.items():
            if name not in expected_shapes:
                continue

            expected = expected_shapes[name]

            # Check if shape matches
            if isinstance(array, np.ndarray):
                actual = array.shape

                # Compare shapes
                if len(actual) != len(expected):
                    all_match = False
                    self.warning_count += 1
                    if self.verbose:
                        logger.warning(f"{context} - {name} has wrong dimensions: expected {expected}, got {actual}")
                else:
                    # Check each dimension
                    for i, (a, e) in enumerate(zip(actual, expected)):
                        if e is not None and a != e:
                            all_match = False
                            self.warning_count += 1
                            if self.verbose:
                                logger.warning(f"{context} - {name} dimension {i} mismatch: expected {e}, got {a}")
            else:
                all_match = False
                self.warning_count += 1
                if self.verbose:
                    logger.warning(f"{context} - {name} is not a numpy array")

        return all_match

    def fix_dimensions(self, arrays, expected_shapes, context=""):
        """
        Fix array dimensions to match expected shapes.

        Args:
            arrays (dict): Dictionary of arrays to fix (will be modified in place)
            expected_shapes (dict): Dictionary of expected shapes
            context (str): Context information for error messages

        Returns:
            dict: Dictionary of fixed arrays
        """
        fixed_arrays = {}

        for name, array in arrays.items():
            if name not in expected_shapes:
                fixed_arrays[name] = array
                continue

            expected = expected_shapes[name]

            # Only process numpy arrays
            if not isinstance(array, np.ndarray):
                try:
                    # Try to convert to numpy array
                    array = np.array(array)
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Converted {name} to numpy array")
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Could not convert {name} to numpy array: {e}")
                    fixed_arrays[name] = array
                    continue

            actual = array.shape
            needs_fixing = False

            # Check if dimensions match
            if len(actual) != len(expected):
                needs_fixing = True
            else:
                # Check each dimension
                for i, (a, e) in enumerate(zip(actual, expected)):
                    if e is not None and a != e:
                        needs_fixing = True
                        break

            # Fix dimensions if needed
            if needs_fixing:
                try:
                    fixed_array = self._fix_array_dimensions(array, expected, name, context)
                    fixed_arrays[name] = fixed_array
                    self.fixed_count += 1
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Could not fix {name} dimensions: {e}")
                    fixed_arrays[name] = array
            else:
                fixed_arrays[name] = array

        return fixed_arrays

    def _fix_array_dimensions(self, array, expected_shape, name, context):
        """
        Fix a single array's dimensions to match expected shape.

        Args:
            array (np.ndarray): Array to fix
            expected_shape (tuple): Expected shape
            name (str): Array name for logging
            context (str): Context information for error messages

        Returns:
            np.ndarray: Fixed array
        """
        actual_shape = array.shape

        # If array is completely wrong shape, create a new one
        if len(actual_shape) != len(expected_shape):
            # Special case: 1D to 2D conversion
            if len(actual_shape) == 1 and len(expected_shape) == 2:
                if expected_shape[1] == 1:
                    # Convert to column vector
                    fixed = array.reshape(-1, 1)
                    if self.verbose:
                        logger.info(f"{context} - Reshaped {name} from {actual_shape} to {fixed.shape}")
                    return fixed
                elif expected_shape[0] == 1:
                    # Convert to row vector
                    fixed = array.reshape(1, -1)
                    if self.verbose:
                        logger.info(f"{context} - Reshaped {name} from {actual_shape} to {fixed.shape}")
                    return fixed
                elif actual_shape[0] == expected_shape[0]:
                    # Expand to 2D (assume each value is a row)
                    fixed = np.zeros(expected_shape)
                    for i in range(min(actual_shape[0], expected_shape[0])):
                        fixed[i] = np.full(expected_shape[1], array[i])
                    if self.verbose:
                        logger.info(f"{context} - Expanded {name} from {actual_shape} to {fixed.shape}")
                    return fixed
                else:
                    # Create new array filled with zeros
                    fixed = np.zeros(expected_shape)
                    # Copy values where possible
                    for i in range(min(actual_shape[0], expected_shape[0])):
                        fixed[i, 0] = array[i]
                    if self.verbose:
                        logger.info(f"{context} - Created new array for {name}: {actual_shape} -> {fixed.shape}")
                    return fixed

            # Special case: 2D to 1D conversion
            elif len(actual_shape) == 2 and len(expected_shape) == 1:
                if actual_shape[1] == 1:
                    # Convert from column vector
                    fixed = array.flatten()
                    if self.verbose:
                        logger.info(f"{context} - Flattened {name} from {actual_shape} to {fixed.shape}")
                    return fixed
                elif actual_shape[0] == 1:
                    # Convert from row vector
                    fixed = array.flatten()
                    if self.verbose:
                        logger.info(f"{context} - Flattened {name} from {actual_shape} to {fixed.shape}")
                    return fixed
                else:
                    # Take first row or sum/average all rows
                    fixed = array.mean(axis=1)
                    if self.verbose:
                        logger.info(f"{context} - Reduced {name} from {actual_shape} to {fixed.shape} using mean")
                    return fixed

            # General case: Create a new array with the right dimensions
            fixed = np.zeros(expected_shape)
            if self.verbose:
                logger.info(f"{context} - Created new array for {name}: {actual_shape} -> {expected_shape}")
            return fixed

        # Same number of dimensions, but different sizes
        fixed = array
        new_shape = list(actual_shape)

        # Fix each dimension
        for i, (a, e) in enumerate(zip(actual_shape, expected_shape)):
            if e is not None and a != e:
                if a < e:
                    # Expand dimension
                    padding = [(0, 0)] * len(actual_shape)
                    padding[i] = (0, e - a)
                    fixed = np.pad(fixed, padding, mode='constant')
                    new_shape[i] = e
                    if self.verbose:
                        logger.info(f"{context} - Expanded {name} dimension {i}: {a} -> {e}")
                else:
                    # Shrink dimension
                    slices = [slice(None)] * len(actual_shape)
                    slices[i] = slice(0, e)
                    fixed = fixed[tuple(slices)]
                    new_shape[i] = e
                    if self.verbose:
                        logger.info(f"{context} - Truncated {name} dimension {i}: {a} -> {e}")

        return fixed

    def verify_and_fix_if_needed(self, arrays, expected_shapes, context=""):
        """
        Verify array dimensions and fix if needed.

        Args:
            arrays (dict): Dictionary of arrays to verify and fix
            expected_shapes (dict): Dictionary of expected shapes
            context (str): Context information for error messages

        Returns:
            dict: Dictionary of verified and fixed arrays
        """
        if not self.verify_dimensions(arrays, expected_shapes, context):
            if self.auto_fix:
                return self.fix_dimensions(arrays, expected_shapes, context)
            elif self.verbose:
                logger.warning(f"{context} - Dimension mismatch detected but auto-fix is disabled")

        return arrays

    def verify_matrix_vector_dimensions(self, matrix, vector, context=""):
        """
        Verify that matrix and vector dimensions are compatible for matrix-vector operations.

        Args:
            matrix (np.ndarray): Matrix of shape (m, n)
            vector (np.ndarray): Vector of shape (n,) or (n, 1)
            context (str): Context information for error messages

        Returns:
            bool: True if dimensions are compatible, False otherwise
        """
        if not isinstance(matrix, np.ndarray) or not isinstance(vector, np.ndarray):
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Matrix or vector is not a numpy array")
            return False

        # Check matrix dimensions
        if len(matrix.shape) != 2:
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Matrix is not 2D: {matrix.shape}")
            return False

        # Check vector dimensions
        if len(vector.shape) not in [1, 2]:
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Vector is not 1D or 2D: {vector.shape}")
            return False

        # Check compatibility
        if len(vector.shape) == 1:
            if matrix.shape[1] != vector.shape[0]:
                self.warning_count += 1
                if self.verbose:
                    logger.warning(f"{context} - Matrix-vector dimension mismatch: {matrix.shape} and {vector.shape}")
                return False
        else:  # len(vector.shape) == 2
            if vector.shape[1] != 1 or matrix.shape[1] != vector.shape[0]:
                self.warning_count += 1
                if self.verbose:
                    logger.warning(f"{context} - Matrix-vector dimension mismatch: {matrix.shape} and {vector.shape}")
                return False

        return True

    def fix_matrix_vector_dimensions(self, matrix, vector, context=""):
        """
        Fix matrix and vector dimensions to be compatible for matrix-vector operations.

        Args:
            matrix (np.ndarray): Matrix to fix
            vector (np.ndarray): Vector to fix
            context (str): Context information for error messages

        Returns:
            tuple: Fixed (matrix, vector)
        """
        # Convert to numpy arrays if needed
        if not isinstance(matrix, np.ndarray):
            try:
                matrix = np.array(matrix)
                self.fixed_count += 1
                if self.verbose:
                    logger.info(f"{context} - Converted matrix to numpy array")
            except Exception as e:
                self.error_count += 1
                if self.verbose:
                    logger.error(f"{context} - Could not convert matrix to numpy array: {e}")

        if not isinstance(vector, np.ndarray):
            try:
                vector = np.array(vector)
                self.fixed_count += 1
                if self.verbose:
                    logger.info(f"{context} - Converted vector to numpy array")
            except Exception as e:
                self.error_count += 1
                if self.verbose:
                    logger.error(f"{context} - Could not convert vector to numpy array: {e}")
                return matrix, vector

        # Ensure matrix is 2D
        if len(matrix.shape) != 2:
            if len(matrix.shape) == 1:
                # Convert 1D to 2D (as row vector)
                matrix = matrix.reshape(1, -1)
                self.fixed_count += 1
                if self.verbose:
                    logger.info(f"{context} - Reshaped matrix from 1D to 2D: {matrix.shape}")
            else:
                # More complex case, create a new 2D matrix
                self.error_count += 1
                if self.verbose:
                    logger.error(f"{context} - Cannot automatically fix matrix with shape {matrix.shape}")
                return matrix, vector

        # Ensure vector is 1D or 2D column
        if len(vector.shape) > 2:
            # More complex case, try to flatten
            try:
                vector = vector.flatten()
                self.fixed_count += 1
                if self.verbose:
                    logger.info(f"{context} - Flattened vector from {vector.shape} to 1D")
            except Exception as e:
                self.error_count += 1
                if self.verbose:
                    logger.error(f"{context} - Could not flatten vector: {e}")
                return matrix, vector

        # Ensure dimensions are compatible
        if len(vector.shape) == 1:
            if matrix.shape[1] != vector.shape[0]:
                # Resize vector or matrix
                if matrix.shape[1] < vector.shape[0]:
                    # Truncate vector
                    vector = vector[:matrix.shape[1]]
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Truncated vector to match matrix: {vector.shape}")
                else:
                    # Expand vector
                    new_vector = np.zeros(matrix.shape[1])
                    new_vector[:vector.shape[0]] = vector
                    vector = new_vector
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Expanded vector to match matrix: {vector.shape}")
        else:  # len(vector.shape) == 2
            if vector.shape[1] != 1:
                # Convert to column vector if multi-column
                if vector.shape[0] == 1:
                    # Convert row vector to column vector
                    vector = vector.T
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Converted row vector to column vector: {vector.shape}")
                else:
                    # Take first column
                    vector = vector[:, 0:1]
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Extracted first column of vector: {vector.shape}")

            if matrix.shape[1] != vector.shape[0]:
                # Resize vector or matrix
                if matrix.shape[1] < vector.shape[0]:
                    # Truncate vector
                    vector = vector[:matrix.shape[1], :]
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Truncated vector to match matrix: {vector.shape}")
                else:
                    # Expand vector
                    new_vector = np.zeros((matrix.shape[1], 1))
                    new_vector[:vector.shape[0], :] = vector
                    vector = new_vector
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Expanded vector to match matrix: {vector.shape}")

        return matrix, vector

    def verify_array_index_bounds(self, array, index, context=""):
        """
        Verify that an index is within bounds for an array.

        Args:
            array (np.ndarray): Array to check
            index (int or tuple): Index to verify
            context (str): Context information for error messages

        Returns:
            bool: True if index is within bounds, False otherwise
        """
        if not isinstance(array, np.ndarray):
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Array is not a numpy array")
            return False

        try:
            if isinstance(index, (int, np.integer)):
                # Single index
                if index < 0 or index >= array.shape[0]:
                    self.warning_count += 1
                    if self.verbose:
                        logger.warning(f"{context} - Index {index} out of bounds for array with shape {array.shape}")
                    return False
            elif isinstance(index, tuple):
                # Multi-dimensional index
                if len(index) != len(array.shape):
                    self.warning_count += 1
                    if self.verbose:
                        logger.warning(
                            f"{context} - Index {index} has wrong dimensionality for array with shape {array.shape}")
                    return False

                for i, idx in enumerate(index):
                    if idx < 0 or idx >= array.shape[i]:
                        self.warning_count += 1
                        if self.verbose:
                            logger.warning(
                                f"{context} - Index {index} out of bounds for array with shape {array.shape}")
                        return False
            else:
                # Unsupported index type
                self.warning_count += 1
                if self.verbose:
                    logger.warning(f"{context} - Unsupported index type: {type(index)}")
                return False
        except Exception as e:
            self.error_count += 1
            if self.verbose:
                logger.error(f"{context} - Error checking index bounds: {e}")
            return False

        return True

    def fix_array_index_bounds(self, array, index, context=""):
        """
        Fix an index to be within bounds for an array.

        Args:
            array (np.ndarray): Array to check
            index (int or tuple): Index to fix
            context (str): Context information for error messages

        Returns:
            tuple: Fixed (array, index) - array may be expanded if needed
        """
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
                self.fixed_count += 1
                if self.verbose:
                    logger.info(f"{context} - Converted array to numpy array")
            except Exception as e:
                self.error_count += 1
                if self.verbose:
                    logger.error(f"{context} - Could not convert array to numpy array: {e}")
                return array, index

        try:
            if isinstance(index, (int, np.integer)):
                # Single index
                if index < 0:
                    # Convert negative index to positive
                    index = array.shape[0] + index
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Converted negative index to positive: {index}")

                if index < 0 or index >= array.shape[0]:
                    # Index still out of bounds, expand array or fix index
                    if index < 0:
                        # Index too negative, clip to 0
                        index = 0
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Clipped negative index to 0")
                    elif self.auto_fix:
                        # Expand array
                        new_array = np.zeros((index + 1,) + array.shape[1:])
                        new_array[:array.shape[0]] = array
                        array = new_array
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Expanded array from shape {array.shape} to {new_array.shape}")
                    else:
                        # Clip index to bounds
                        index = array.shape[0] - 1
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Clipped index to array bounds: {index}")

            elif isinstance(index, tuple):
                # Multi-dimensional index
                if len(index) != len(array.shape):
                    # Dimensionality mismatch
                    if len(index) < len(array.shape):
                        # Add missing dimensions
                        new_index = list(index)
                        for _ in range(len(array.shape) - len(index)):
                            new_index.append(0)
                        index = tuple(new_index)
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Added missing dimensions to index: {index}")
                    else:
                        # Too many dimensions, truncate
                        index = index[:len(array.shape)]
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Truncated index to match array dimensions: {index}")

                # Check each dimension
                fixed_index = list(index)
                expand_needed = False
                expand_dims = list(array.shape)

                for i, idx in enumerate(index):
                    if idx < 0:
                        # Convert negative index to positive
                        fixed_index[i] = array.shape[i] + idx
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(
                                f"{context} - Converted negative index to positive: dim {i}, {idx} -> {fixed_index[i]}")

                    if fixed_index[i] < 0 or fixed_index[i] >= array.shape[i]:
                        if fixed_index[i] < 0:
                            # Index too negative, clip to 0
                            fixed_index[i] = 0
                            self.fixed_count += 1
                            if self.verbose:
                                logger.info(f"{context} - Clipped negative index to 0: dim {i}")
                        elif self.auto_fix:
                            # Mark for expansion
                            expand_needed = True
                            expand_dims[i] = max(expand_dims[i], fixed_index[i] + 1)
                        else:
                            # Clip index to bounds
                            fixed_index[i] = array.shape[i] - 1
                            self.fixed_count += 1
                            if self.verbose:
                                logger.info(f"{context} - Clipped index to array bounds: dim {i}, {fixed_index[i]}")

                if expand_needed:
                    # Expand array
                    new_array = np.zeros(tuple(expand_dims))
                    slices = tuple(slice(0, s) for s in array.shape)
                    new_array[slices] = array
                    array = new_array
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Expanded array to shape {array.shape}")

                index = tuple(fixed_index)
            else:
                # Unsupported index type, convert to int if possible
                try:
                    index = int(index)
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Converted index to int: {index}")

                    # Now check bounds with the converted index
                    if index < 0 or index >= array.shape[0]:
                        if index < 0:
                            index = 0
                        elif self.auto_fix:
                            # Expand array
                            new_array = np.zeros((index + 1,) + array.shape[1:])
                            new_array[:array.shape[0]] = array
                            array = new_array
                        else:
                            index = array.shape[0] - 1

                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Fixed converted index bounds: {index}")
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Could not convert index to int: {e}")

        except Exception as e:
            self.error_count += 1
            if self.verbose:
                logger.error(f"{context} - Error fixing index bounds: {e}")

        return array, index

    def verify_and_fix_array_index(self, array, index, context=""):
        """
        Verify array index bounds and fix if needed.

        Args:
            array (np.ndarray): Array to check
            index (int or tuple): Index to verify and fix
            context (str): Context information for error messages

        Returns:
            tuple: Verified and fixed (array, index)
        """
        if not self.verify_array_index_bounds(array, index, context):
            if self.auto_fix:
                return self.fix_array_index_bounds(array, index, context)
            elif self.verbose:
                logger.warning(f"{context} - Index out of bounds but auto-fix is disabled")

        return array, index

    def verify_and_fix_boolean_indexing(self, array, boolean_array, context=""):
        """
        Verify and fix boolean array for indexing.

        Args:
            array (np.ndarray): Array to be indexed
            boolean_array (np.ndarray): Boolean array for indexing
            context (str): Context information for error messages

        Returns:
            np.ndarray: Fixed boolean array
        """
        if not isinstance(array, np.ndarray) or not isinstance(boolean_array, np.ndarray):
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Array or boolean index is not a numpy array")

            if self.auto_fix:
                # Try to convert to numpy arrays
                if not isinstance(array, np.ndarray):
                    try:
                        array = np.array(array)
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Converted array to numpy array")
                    except Exception as e:
                        self.error_count += 1
                        if self.verbose:
                            logger.error(f"{context} - Could not convert array to numpy array: {e}")

                if not isinstance(boolean_array, np.ndarray):
                    try:
                        boolean_array = np.array(boolean_array, dtype=bool)
                        self.fixed_count += 1
                        if self.verbose:
                            logger.info(f"{context} - Converted boolean array to numpy array")
                    except Exception as e:
                        self.error_count += 1
                        if self.verbose:
                            logger.error(f"{context} - Could not convert boolean array to numpy array: {e}")
                        return boolean_array
            else:
                return boolean_array

        # Check if boolean_array is actually boolean
        if boolean_array.dtype != bool:
            self.warning_count += 1
            if self.verbose:
                logger.warning(f"{context} - Boolean array has non-boolean dtype: {boolean_array.dtype}")

            if self.auto_fix:
                try:
                    boolean_array = boolean_array.astype(bool)
                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Converted array to boolean dtype")
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Could not convert array to boolean dtype: {e}")
                    return boolean_array

        # Check if boolean_array shape matches array shape in the first dimension
        if len(boolean_array.shape) != 1 or boolean_array.shape[0] != array.shape[0]:
            self.warning_count += 1
            if self.verbose:
                logger.warning(
                    f"{context} - Boolean array shape {boolean_array.shape} doesn't match array shape {array.shape}")

            if self.auto_fix:
                try:
                    # Resize boolean_array to match the first dimension of array
                    if boolean_array.shape[0] < array.shape[0]:
                        # Expand
                        new_boolean = np.zeros(array.shape[0], dtype=bool)
                        new_boolean[:boolean_array.shape[0]] = boolean_array
                        boolean_array = new_boolean
                    else:
                        # Truncate
                        boolean_array = boolean_array[:array.shape[0]]

                    self.fixed_count += 1
                    if self.verbose:
                        logger.info(f"{context} - Resized boolean array to match array shape: {boolean_array.shape}")
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Could not resize boolean array: {e}")

        return boolean_array

    def safe_multi_civilization_update(self, arrays, num_civs, context=""):
        """
        Safely update multi-civilization arrays to match the current number of civilizations.

        Args:
            arrays (dict): Dictionary of arrays to update
            num_civs (int): Current number of civilizations
            context (str): Context information for error messages

        Returns:
            dict: Updated arrays with correct dimensions
        """
        updated_arrays = {}

        for name, array in arrays.items():
            # Skip non-array items
            if not isinstance(array, np.ndarray):
                updated_arrays[name] = array
                continue

            # Determine expected shape based on array structure
            if len(array.shape) == 1:
                # 1D array (usually per-civilization values)
                expected_shape = (num_civs,)
            elif len(array.shape) == 2:
                # 2D array (usually per-civilization with additional dimension)
                if array.shape[0] == num_civs:
                    # Already correct first dimension
                    updated_arrays[name] = array
                    continue

                expected_shape = (num_civs, array.shape[1])
            elif len(array.shape) == 3:
                # 3D array (usually civilization-to-civilization matrices)
                expected_shape = (num_civs, num_civs, array.shape[2] if len(array.shape) > 2 else 1)
            else:
                # More complex case, leave as is
                updated_arrays[name] = array
                continue

            # Check if resize needed
            if array.shape != expected_shape:
                try:
                    if self.auto_fix:
                        if self.verbose:
                            logger.warning(f"{context} - {name} array resized to match {num_civs} civilizations")

                        # Create new array with expected shape
                        new_array = np.zeros(expected_shape)

                        # Copy data from old array where possible
                        if len(expected_shape) == 1:
                            # 1D array
                            copy_length = min(array.shape[0], expected_shape[0])
                            new_array[:copy_length] = array[:copy_length]
                        elif len(expected_shape) == 2:
                            # 2D array
                            copy_rows = min(array.shape[0], expected_shape[0])
                            copy_cols = min(array.shape[1], expected_shape[1])
                            new_array[:copy_rows, :copy_cols] = array[:copy_rows, :copy_cols]
                        elif len(expected_shape) == 3:
                            # 3D array
                            copy_i = min(array.shape[0], expected_shape[0])
                            copy_j = min(array.shape[1], expected_shape[1])
                            copy_k = min(array.shape[2], expected_shape[2])
                            new_array[:copy_i, :copy_j, :copy_k] = array[:copy_i, :copy_j, :copy_k]

                        updated_arrays[name] = new_array
                        self.fixed_count += 1
                    else:
                        updated_arrays[name] = array
                        if self.verbose:
                            logger.warning(
                                f"{context} - {name} array has wrong shape {array.shape}, expected {expected_shape}")
                except Exception as e:
                    updated_arrays[name] = array
                    self.error_count += 1
                    if self.verbose:
                        logger.error(f"{context} - Error resizing {name} array: {e}")
            else:
                updated_arrays[name] = array

        return updated_arrays

    def get_status_report(self):
        """
        Get a status report of dimension fixes.

        Returns:
            dict: Status report
        """
        return {
            "fixed_count": self.fixed_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count
        }


# Module-level instance for easy import
dim_handler = DimensionHandler()


def safe_calculate_distance_matrix(positions, num_civs=None):
    """
    Safely calculate distance matrix between civilization positions.

    Args:
        positions (np.ndarray): Array of civilization positions
        num_civs (int): Number of civilizations (optional)

    Returns:
        np.ndarray: Distance matrix
    """
    context = "calculate_distance_matrix"

    # Ensure positions is a numpy array
    if not isinstance(positions, np.ndarray):
        try:
            positions = np.array(positions)
            if dim_handler.verbose:
                logger.info(f"{context} - Converted positions to numpy array")
        except Exception as e:
            if dim_handler.verbose:
                logger.error(f"{context} - Could not convert positions to numpy array: {e}")
            return np.zeros((0, 0))

    # Check dimensions
    if len(positions.shape) != 2:
        if dim_handler.verbose:
            logger.warning(f"{context} - Positions array has wrong dimensionality: {positions.shape}")

        if dim_handler.auto_fix:
            if len(positions.shape) == 1:
                # Assume 1D positions, reshape to 2D with 1 position dimension
                positions = positions.reshape(-1, 1)
                if dim_handler.verbose:
                    logger.info(f"{context} - Reshaped positions to 2D: {positions.shape}")
            else:
                if dim_handler.verbose:
                    logger.error(f"{context} - Cannot fix positions with shape {positions.shape}")
                return np.zeros((positions.shape[0], positions.shape[0]))

    # Get number of civilizations
    if num_civs is None:
        num_civs = positions.shape[0]

    # Ensure positions has the right first dimension
    if positions.shape[0] != num_civs:
        if dim_handler.verbose:
            logger.warning(f"{context} - Positions has {positions.shape[0]} rows, expected {num_civs}")

        if dim_handler.auto_fix:
            if positions.shape[0] < num_civs:
                # Expand
                new_positions = np.zeros((num_civs, positions.shape[1]))
                new_positions[:positions.shape[0]] = positions
                positions = new_positions
            else:
                # Truncate
                positions = positions[:num_civs]

            if dim_handler.verbose:
                logger.info(f"{context} - Resized positions to {positions.shape}")

    # Calculate distance matrix
    try:
        # Get dimensions
        n_civs = positions.shape[0]
        n_dims = positions.shape[1]

        # Create matrix
        distance_matrix = np.zeros((n_civs, n_civs))

        # Calculate Euclidean distances
        for i in range(n_civs):
            for j in range(n_civs):
                if i != j:
                    # Use numpy for faster calculation
                    diff = positions[i] - positions[j]
                    distance_matrix[i, j] = np.sqrt(np.sum(diff * diff))

        return distance_matrix

    except Exception as e:
        if dim_handler.verbose:
            logger.error(f"{context} - Error calculating distance matrix: {e}")

        # Return empty matrix on error
        return np.zeros((num_civs, num_civs))


def safe_process_civilization_interactions(civs_data, interaction_func, num_civs=None):
    """
    Safely process interactions between civilizations.

    Args:
        civs_data (dict): Dictionary of civilization data arrays
        interaction_func (function): Function to process interactions
        num_civs (int): Number of civilizations (optional)

    Returns:
        dict: Updated civilization data
    """
    context = "process_civilization_interactions"

    # Determine number of civilizations
    if num_civs is None:
        # Try to infer from data
        for key, value in civs_data.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                num_civs = value.shape[0]
                break

        if num_civs is None:
            if dim_handler.verbose:
                logger.warning(f"{context} - Could not determine number of civilizations")
            return civs_data

    # Ensure all arrays have the right dimensions
    safe_data = dim_handler.safe_multi_civilization_update(civs_data, num_civs, context)

    # Process interactions
    try:
        return interaction_func(safe_data, num_civs)
    except Exception as e:
        if dim_handler.verbose:
            logger.error(f"{context} - Error processing interactions: {e}")
        return safe_data


def example_usage():
    """Example of how to use the dimension handler."""
    # Create some sample arrays with dimension issues
    positions = np.random.random((3, 2))  # 3 civilizations, 2D positions

    # Shortened array that will cause issues
    influence = np.array([1.0, 2.0])  # Only 2 values, should have 3

    # Mismatched sizes array
    resources = np.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2x2, should be 3x2

    # Create a handler
    handler = DimensionHandler(verbose=True, auto_fix=True)

    # Define expected shapes
    expected_shapes = {
        'positions': (3, 2),
        'influence': (3,),
        'resources': (3, 2)
    }

    # Check and fix dimensions
    arrays = {
        'positions': positions,
        'influence': influence,
        'resources': resources
    }

    fixed_arrays = handler.verify_and_fix_if_needed(arrays, expected_shapes, "example")

    print("\nFixed arrays:")
    for name, array in fixed_arrays.items():
        print(f"{name}: {array.shape} - {array}")

    # Example of safely calculating distance matrix
    distance_matrix = safe_calculate_distance_matrix(fixed_arrays['positions'])

    print("\nDistance matrix:")
    print(distance_matrix)

    # Example of handling out-of-bounds index
    array = np.array([1, 2, 3])
    index = 5  # Out of bounds

    fixed_array, fixed_index = handler.verify_and_fix_array_index(array, index, "index_example")

    print("\nFixed array index:")
    print(f"Original: {array}, index: {index}")
    print(f"Fixed: {fixed_array}, index: {fixed_index}")

    # Example of handling boolean indexing
    array = np.array([1, 2, 3, 4, 5])
    boolean_array = np.array([True, False, True])  # Too short

    fixed_boolean = handler.verify_and_fix_boolean_indexing(array, boolean_array, "boolean_example")

    print("\nFixed boolean indexing:")
    print(f"Original: {array}, boolean: {boolean_array}")
    print(f"Fixed boolean: {fixed_boolean}")

    # Show status report
    report = handler.get_status_report()
    print("\nStatus report:")
    print(f"Fixed: {report['fixed_count']}, Warnings: {report['warning_count']}, Errors: {report['error_count']}")

    return handler


if __name__ == "__main__":
    example_usage()