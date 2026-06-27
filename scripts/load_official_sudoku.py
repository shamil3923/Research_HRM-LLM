"""
Load official Sudoku-Extreme dataset from HuggingFace
"""

import csv
import numpy as np
from typing import List, Tuple
import random

def load_official_sudoku_data(
    csv_path: str, 
    max_samples: int = 1000,
    min_difficulty: int = 0,
    shuffle_augment: bool = True,
    num_augmentations: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load Sudoku data from official CSV format
    
    CSV format: source,question,answer,rating
    - question: 81-char string with '.' for blanks, digits 1-9 for givens
    - answer: 81-char string with digits 1-9 for complete solution
    - rating: difficulty rating (higher = harder)
    """
    
    puzzles = []
    solutions = []
    
    print(f"Loading Sudoku data from {csv_path}")
    print(f"Max samples: {max_samples}, Min difficulty: {min_difficulty}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        loaded_count = 0
        total_count = 0
        
        for row in reader:
            total_count += 1
            
            # Check difficulty filter
            rating = int(row['rating'])
            if rating < min_difficulty:
                continue
            
            # Parse puzzle and solution
            question = row['question']
            answer = row['answer']
            
            # Validate format
            if len(question) != 81 or len(answer) != 81:
                continue
            
            # Convert to numpy arrays
            puzzle = np.array([int(c) if c.isdigit() else 0 for c in question])
            solution = np.array([int(c) for c in answer])
            
            # Validate solution
            if not np.all((solution >= 1) & (solution <= 9)):
                continue
            
            # Add original puzzle
            puzzles.append(puzzle)
            solutions.append(solution)
            loaded_count += 1
            
            # Add augmentations if requested
            if shuffle_augment and num_augmentations > 0:
                for _ in range(num_augmentations):
                    aug_puzzle, aug_solution = shuffle_sudoku(
                        puzzle.reshape(9, 9), 
                        solution.reshape(9, 9)
                    )
                    puzzles.append(aug_puzzle.flatten())
                    solutions.append(aug_solution.flatten())
                    loaded_count += 1
            
            # Stop if we have enough samples
            if loaded_count >= max_samples * (1 + num_augmentations):
                break
            
            if loaded_count % 1000 == 0:
                print(f"  Loaded {loaded_count} samples...")
    
    print(f"Loaded {len(puzzles)} puzzles from {total_count} total")
    print(f"Difficulty range: {min_difficulty}+ (rating)")
    
    return puzzles, solutions

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply valid Sudoku transformations that preserve correctness
    Based on official HRM implementation
    """
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks)
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def analyze_dataset(csv_path: str, sample_size: int = 10000):
    """Analyze the dataset to understand difficulty distribution"""
    
    difficulties = []
    
    print(f"Analyzing {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        count = 0
        for row in reader:
            difficulties.append(int(row['rating']))
            count += 1
            
            if count >= sample_size:
                break
    
    difficulties = np.array(difficulties)
    
    print(f"Analyzed {len(difficulties)} puzzles:")
    print(f"  Difficulty range: {difficulties.min()} - {difficulties.max()}")
    print(f"  Mean difficulty: {difficulties.mean():.1f}")
    print(f"  Median difficulty: {np.median(difficulties):.1f}")
    print(f"  Std difficulty: {difficulties.std():.1f}")
    
    # Show percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("  Percentiles:")
    for p in percentiles:
        print(f"    {p}%: {np.percentile(difficulties, p):.1f}")
    
    return difficulties

if __name__ == "__main__":
    # Analyze the dataset
    print("=" * 60)
    print("ANALYZING OFFICIAL SUDOKU-EXTREME DATASET")
    print("=" * 60)
    
    train_difficulties = analyze_dataset("data/sudoku-extreme/train.csv")
    test_difficulties = analyze_dataset("data/sudoku-extreme/test.csv")
    
    print("\n" + "=" * 60)
    print("LOADING SAMPLE DATA")
    print("=" * 60)
    
    # Load a small sample for testing
    train_puzzles, train_solutions = load_official_sudoku_data(
        "data/sudoku-extreme/train.csv",
        max_samples=100,
        min_difficulty=20,  # Medium difficulty
        num_augmentations=2
    )
    
    print(f"\nLoaded {len(train_puzzles)} training examples")
    
    # Show a sample puzzle
    print("\nSample puzzle:")
    puzzle = train_puzzles[0].reshape(9, 9)
    solution = train_solutions[0].reshape(9, 9)
    
    print("Puzzle:")
    for row in puzzle:
        print(" ".join(str(x) if x != 0 else "." for x in row))
    
    print("\nSolution:")
    for row in solution:
        print(" ".join(str(x) for x in row))