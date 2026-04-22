import subprocess
import sys
from pathlib import Path

def test_jit_hello():
    examples_dir = Path(__file__).parent.parent / "examples"
    hello_bf2 = examples_dir / "hello.bf2"
    
    # Run bf2 run hello.bf2
    res = subprocess.run(
        [sys.executable, "-m", "bf2", "run", str(hello_bf2)],
        capture_output=True,
        text=True,
        check=True
    )
    assert res.stdout == "Hello, World!\n"

def test_jit_fib():
    examples_dir = Path(__file__).parent.parent / "examples"
    fib_bf2 = examples_dir / "fibonacci-small.bf2"
    
    # Run bf2 run fibonacci-small.bf2
    res = subprocess.run(
        [sys.executable, "-m", "bf2", "run", str(fib_bf2)],
        capture_output=True,
        text=True,
        check=True
    )
    # Fib output usually contains 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
    assert "0" in res.stdout
    assert "34" in res.stdout
