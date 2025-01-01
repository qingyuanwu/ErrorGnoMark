class chip:
    """
    Represents a quantum chip with a 2D grid of qubits.
    """
    def __init__(self, rows=10, columns=10):
        self.rows = rows
        self.columns = columns


class qubit_selection:
    def __init__(self, chip, qubit_index_max=50, qubit_number=9, option=None):
        self.chip = chip
        self.qubit_index_max = qubit_index_max
        self.qubit_number = int(qubit_number)  # Ensure this is an integer
        self.option = option if option is not None else {}
        self.rows = getattr(chip, 'rows', 10)      # Default to 10 rows if not specified
        self.columns = getattr(chip, 'columns', 10)  # Default to 10 columns if not specified


    def quselected(self):
        """
        Selects qubits and their connectivity based on the specified constraints.

        Returns:
            dict: 
                - "qubit_index_list" (list): Indices of selected qubits.
                - "qubit_connectivity" (list): Connectivity data as pairs of qubits.
        """
        # Retrieve constraints from options
        max_qubits_per_row = self.option.get('max_qubits_per_row', self.columns)  # Default to all qubits in a row
        min_qubit_index = self.option.get('min_qubit_index', 0)
        max_qubit_index = self.option.get('max_qubit_index', self.qubit_index_max)

        # Ensure qubit_number does not exceed the total qubits in the grid
        total_qubits = self.rows * self.columns
        qubit_number = min(self.qubit_number, total_qubits)

        best_selection = {
            "qubit_index_list": [],
            "qubit_connectivity": []
        }

        # Iterate over all possible rectangle sizes and positions
        for height in range(1, self.rows + 1):
            for width in range(1, self.columns + 1):
                # Check if the rectangle can fit within the max qubit number
                if height * width > qubit_number:
                    continue

                # Iterate over all possible starting positions
                for row_start in range(self.rows - height + 1):
                    for col_start in range(self.columns - width + 1):
                        # Collect qubit indices in the rectangle
                        qubit_indices = []
                        valid = True
                        for r in range(row_start, row_start + height):
                            if width > max_qubits_per_row:
                                valid = False
                                break
                            for c in range(col_start, col_start + width):
                                q = r * self.columns + c
                                if q < min_qubit_index or q > max_qubit_index:
                                    valid = False
                                    break
                                qubit_indices.append(q)
                            if not valid:
                                break
                        if not valid:
                            continue

                        # Check if this selection is better (more qubits) than the previous best
                        if len(qubit_indices) > len(best_selection["qubit_index_list"]):
                            best_selection["qubit_index_list"] = qubit_indices

                            # Create horizontal connectivity within each row
                            qubit_connectivity = []
                            for r in range(row_start, row_start + height):
                                for c in range(col_start, col_start + width - 1):
                                    q1 = r * self.columns + c 
                                    q2 = r * self.columns + (c + 1)
                                    qubit_connectivity.append([q1, q2])
                            best_selection["qubit_connectivity"] = qubit_connectivity

        # If no selection was found, return empty lists
        if not best_selection["qubit_index_list"]:
            return best_selection

        # Print chip structure
        print(f"Chip Structure: {self.rows} rows x {self.columns} columns")
        print("Selected Qubits and Connectivity:")

        # Create a grid to visualize selected qubits and connectivity
        grid = [["." for _ in range(self.columns)] for _ in range(self.rows)]

        for q in best_selection["qubit_index_list"]:
            row = q // self.columns
            col = q % self.columns
            grid[row][col] = "Q"

        for conn in best_selection["qubit_connectivity"]:
            q1, q2 = conn
            r1, c1 = q1 // self.columns, q1 % self.columns
            r2, c2 = q2 // self.columns, q2 % self.columns
            if r1 == r2:  # Horizontal connection
                grid[r1][c1] = "-"
            elif c1 == c2:  # Vertical connection
                grid[r1][c1] = "|"

        for row in grid:
            print(" ".join(row))

        return best_selection
