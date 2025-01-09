pub mod instruction;
pub mod r#type;

use instruction::Expr;
use r#type::{FuncType, GlobalType, Limits, RefType, TableType, ValType};

pub type TypeIdx = u32;
pub type FuncIdx = u32;
pub type TableIdx = u32;
pub type MemIdx = u32;
pub type GlobalIdx = u32;
pub type ElemIdx = u32;
pub type DataIdx = u32;
pub type LocalIdx = u32;
pub type LabelIdx = u32;

pub struct WasmModule {
    pub types: Vec<FuncType>,
    pub funcs: Vec<Func>,
    pub tables: Vec<TableType>,
    pub mems: Vec<Limits>,
    pub globals: Vec<(GlobalType, Expr)>,
    pub elems: Vec<Elem>,
    pub datas: Vec<Data>,
    pub start: Option<FuncIdx>,
    pub imports: Vec<Import>,
    pub exports: Vec<Export>,
}

pub struct Func {
    pub typeidx: TypeIdx,
    pub locals: Vec<ValType>,
    pub body: Expr,
}

pub enum ElemMode {
    Passive,
    Active(TableIdx, Expr),
    Declaritive,
}

pub struct Elem {
    pub elem_type: RefType,
    pub init: Vec<Expr>,
    pub mode: ElemMode,
}

pub enum DataMode {
    Passive,
    Active(MemIdx, Expr),
}

pub struct Data {
    pub init: u8,
    pub mode: DataMode,
}

pub enum ImportDesc {
    Func(TypeIdx),
    Table(TableType),
    Mem(Limits),
    Global(GlobalType),
}

pub struct Import {
    pub module: String,
    pub name: String,
    pub desc: ImportDesc,
}

pub enum ExportDesc {
    Func(FuncIdx),
    Table(TableIdx),
    Mem(MemIdx),
    Global(GlobalIdx),
}

pub struct Export {
    pub name: String,
    pub desc: ExportDesc,
}
