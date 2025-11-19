# test_calculator.py
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    """Tests for the Calculator class."""

    def setUp(self):
        """Set up a new Calculator instance before each test."""
        self.calc = Calculator()

    def test_add_positive(self):
        """Test addition with two positive numbers."""
        self.assertEqual(self.calc.add(5, 3), 8)

    def test_add_negative(self):
        """Test addition with positive and negative numbers."""
        self.assertEqual(self.calc.add(-10, 5), -5)

    def test_subtract_basic(self):
        """Test basic subtraction."""
        self.assertEqual(self.calc.subtract(10, 4), 6)

    def test_subtract_zero(self):
        """Test subtraction resulting in zero."""
        self.assertEqual(self.calc.subtract(5, 5), 0)

if __name__ == '__main__':
    unittest.main()