mod structure;
mod validation;
use structure::{
    instruction::{Expr, Instr},
    r#type::FuncType,
    Func, WasmModule,
};
use validation::validate_module;

fn main() {
    let func = Func {
        typeidx: 0,
        locals: Vec::new(),
        body: Expr(vec![Instr::I32Const(5), Instr::I32Const(6), Instr::I32Add]),
    };
    let wasm_module = WasmModule {
        types: vec![FuncType(Vec::new(), Vec::new())],
        funcs: vec![func],
        tables: Vec::new(),
        mems: Vec::new(),
        globals: Vec::new(),
        elems: Vec::new(),
        datas: Vec::new(),
        start: Some(0),
        imports: Vec::new(),
        exports: Vec::new(),
    };
    println!("{}", validate_module(&wasm_module));
}
