## An example of module documentation to go into workflow

"""
main.py
====================================
The core module of my example project
"""

def about_me(your_name):
    """
    Return the most important thing about a person.
    Parameters
    ----------
    your_name
        A string indicating the name of the person.
    """
    return "The wise {} loves Python.".format(your_name)


class ExampleClass:
    """An example docstring for a class definition."""

    def __init__(self, name):
        """
        Blah blah blah.
        Parameters
        ---------
        name
            A string to assign to the `name` instance attribute.
        """
        self.name = name

    def about_self(self):
        """
        Return information about an instance created from ExampleClass.
        """
