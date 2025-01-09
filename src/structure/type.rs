#[derive(PartialEq, Clone, Copy)]
pub enum Mut {
    Const,
    Var,
}

#[derive(PartialEq, Clone, Copy)]
pub struct Limits(pub u32, pub Option<u32>);

#[derive(PartialEq, Clone, Copy)]
pub struct TableType(pub Limits, pub RefType);

#[derive(PartialEq)]
pub enum ExternType {
    FuncType(FuncType),
    TableType(TableType),
    MemType(Limits),
    GlobalType(GlobalType),
}

#[derive(PartialEq, Clone, Copy)]
pub struct GlobalType(pub Mut, pub ValType);

#[derive(Clone, Copy, PartialEq)]
pub enum NumType {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum RefType {
    FuncRef,
    ExternRef,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ValType {
    NumType(NumType),
    VecType,
    RefType(RefType),
}

#[derive(PartialEq, Clone)]
pub struct FuncType(pub Vec<ValType>, pub Vec<ValType>);

impl FuncType {
    fn init() -> Self {
        Self(Vec::new(), Vec::new())
    }
}
