import sys
sys.path.insert(0, ".")
from bf2.compiler.parser import parse_source

parse_source("fn main() -> i32 { [ { } ] ret 0 }")
parse_source("fn main() -> i32 { loop 5 { ret 0 } ret 0 }")
print("Parsed OK")
