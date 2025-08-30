def say_hello(name: str) -> str:
    ## basic function to return a greeting message
    """Return a greeting message to students in the IDS class."""
    return f"Hello, {name}, welcome to Data Engineering Systems (IDS 706)!"


def add(a: int, b: int) -> int:
    # function to add two numbers
    """Return the sum of two numbers."""
    return a + b


if __name__ == "__main__":
    print(say_hello("Prof. Yu"))
    print("2 + 3 =", add(2, 3))
