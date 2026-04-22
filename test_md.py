import llvmlite.ir as ir
m = ir.Module()
f = ir.Function(m, ir.FunctionType(ir.VoidType(), []), name="test")
b = f.append_basic_block("entry")
builder = ir.IRBuilder(b)
br = builder.branch(b)
md = m.add_metadata([])
md.operands = [md]
br.set_metadata("llvm.loop", md)
print(m)
