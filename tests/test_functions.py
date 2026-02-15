"""Basic unit tests for functions.py examples."""

import math
import unittest

import functions


class TestFunctionsModule(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(functions.greet("Rem"), "Hello, Rem!")

    def test_calculate_area(self):
        self.assertEqual(functions.calculate_area(4, 5), 20)
        self.assertAlmostEqual(functions.calculate_area(2.5, 4.0), 10.0)

    def test_celsius_to_fahrenheit(self):
        self.assertEqual(functions.celsius_to_fahrenheit(0), 32)
        self.assertAlmostEqual(functions.celsius_to_fahrenheit(25), 77)

    def test_factorial(self):
        self.assertEqual(functions.factorial(0), 1)
        self.assertEqual(functions.factorial(5), math.factorial(5))

    def test_is_even(self):
        self.assertTrue(functions.is_even(10))
        self.assertFalse(functions.is_even(7))

    def test_calculate_volume(self):
        self.assertEqual(functions.calculate_volume(2, 3, 4), 24)

    def test_create_user_defaults(self):
        user = functions.create_user("alice")
        self.assertEqual(user["username"], "alice")
        self.assertFalse(user["admin"])
        self.assertTrue(user["active"])

    def test_create_user_custom(self):
        user = functions.create_user("bob", is_admin=True, is_active=False)
        self.assertTrue(user["admin"])
        self.assertFalse(user["active"])


if __name__ == "__main__":
    unittest.main()
