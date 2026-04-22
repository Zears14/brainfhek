import pytest
import subprocess
import os
import sys

def run_bf2(source: str):
    with open("temp.bf2", "w") as f:
        f.write(source)
    try:
        res = subprocess.run([sys.executable, "-m", "bf2", "run", "temp.bf2"], capture_output=True, text=True, timeout=10)
        return res.returncode, res.stdout, res.stderr
    finally:
        if os.path.exists("temp.bf2"):
            os.remove("temp.bf2")

def test_basic_link():
    source = """
    seg a { f32[4] }
    seg b { f32[4] }
    seglink c { f32[4] } = a + b
    
    fn main() -> i32 {
        a[0] = 10.0
        b[0] = 5.0
        @c[0]
        if (== 15) {
            ret 0
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"

def test_dynamic_link():
    source = """
    seg a { i32[4] }
    seg b { i32[4] }
    seglink c { i32[4] } = a * b
    
    fn main() -> i32 {
        i32 i = 0
        {4} {
            a[i] = i + 1
            b[i] = 10
            i = i + 1
        }
        
        // c should be [10, 20, 30, 40]
        @c[0]
        if (== 10) {
            @c[1]
            if (== 20) {
                @c[2]
                if (== 30) {
                    @c[3]
                    if (== 40) {
                        ret 0
                    }
                }
            }
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"

def test_complex_expression():
    source = """
    seg a { f32[4] }
    seg b { f32[4] }
    seglink c { f32[4] } = a * 2.0 + b
    
    fn main() -> i32 {
        a[0] = 5.0
        b[0] = 1.0
        @c[0]
        if (== 11) {
            ret 0
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"

def test_multi_level_link():
    source = """
    seg a { i32[4] }
    seglink b { i32[4] } = a + 1
    seglink c { i32[4] } = b + 1
    
    fn main() -> i32 {
        a[0] = 10
        @c[0]
        if (== 12) {
            ret 0
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"

def test_circular_link():
    source = """
    seglink a { i32[4] } = b
    seglink b { i32[4] } = a
    fn main() -> i32 { ret 0 }
    """
    rc, out, err = run_bf2(source)
    assert rc != 0
    assert "circular dependency" in err

def test_self_link():
    source = """
    seglink a { i32[4] } = a + 1
    fn main() -> i32 { ret 0 }
    """
    rc, out, err = run_bf2(source)
    assert rc != 0
    assert "cannot depend on itself" in err

def test_unlink_stmt():
    source = """
    seg a { i32[1] }
    seglink b { i32[1] } = a
    fn main() -> i32 {
        a[0] = 10
        unlink b
        a[0] = 20
        @b[0]
        if (== 10) {
            ret 0
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"

def test_one_time_init():
    source = """
    seg a { i32[1] } = 5
    seg b { i32[1] } = a
    fn main() -> i32 {
        @b[0]
        if (== 5) {
            a[0] = 10
            @b[0]
            if (== 5) {
                ret 0
            }
        }
        ret 1
    }
    """
    rc, out, err = run_bf2(source)
    assert rc == 0, f"Error: {err}"
