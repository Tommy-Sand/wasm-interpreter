use crate::structure::{
    instruction::{BlockType, Expr, Instr},
    r#type::{ExternType, FuncType, GlobalType, Limits, Mut, NumType, RefType, TableType, ValType},
    Data, DataMode, Elem, ElemMode, Export, ExportDesc, Func, FuncIdx, GlobalIdx, Import,
    ImportDesc, MemIdx, TableIdx, TypeIdx, WasmModule,
};
use std::mem;

pub fn validate_module(module: &WasmModule) -> bool {
    let mut wasm_context = WasmContext::new(module);

    wasm_context.validate()
}

struct CtrlFrame {
    instr: Instr,
    start_types: Vec<ValType>,
    end_types: Vec<ValType>,
    height: usize,
    unreachable: bool,
}

impl CtrlFrame {
    fn label_types(&self) -> &[ValType] {
        if mem::discriminant(&self.instr) == mem::discriminant(&Instr::Loop(BlockType::TypeIdx(0)))
        {
            return &self.start_types;
        }
        &self.end_types
    }
}

pub struct WasmContext<'a> {
    module: &'a WasmModule,
    funcs: Vec<FuncType>,
    tables: Vec<TableType>,
    mems: Vec<Limits>,
    globals: Vec<GlobalType>,
    elems: Vec<RefType>,
    datas: Vec<bool>,
    locals: Vec<ValType>,
    labels: Vec<Vec<ValType>>,
    returns: Option<Vec<ValType>>,
    refs: Vec<FuncIdx>,
    type_stack: Vec<Option<ValType>>,
    ctrl_stack: Vec<CtrlFrame>,
}

impl WasmContext<'_> {
    pub fn new<'a>(module: &'a WasmModule) -> WasmContext<'a> {
        let mod_func_types: Vec<FuncType> = module
            .funcs
            .iter()
            .map(|func| module.types.get(func.typeidx as usize).unwrap().clone())
            .collect();
        let mut funcs: Vec<FuncType> = module
            .imports
            .iter()
            .filter_map(|import| match import.desc {
                ImportDesc::Func(type_idx) => {
                    Some(module.types.get(type_idx as usize).unwrap().clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        funcs.extend_from_slice(&mod_func_types);
        let funcs = funcs;

        let mut tables: Vec<TableType> = module
            .imports
            .iter()
            .filter_map(|import| match import.desc {
                ImportDesc::Table(table_type) => Some(table_type),
                _ => None,
            })
            .collect();
        tables.extend_from_slice(&module.tables);
        let tables = tables;

        let mut mems: Vec<Limits> = module
            .imports
            .iter()
            .filter_map(|import| match import.desc {
                ImportDesc::Mem(limits) => Some(limits),
                _ => None,
            })
            .collect();
        mems.extend_from_slice(&module.mems);
        let mems = mems;

        let mod_global_types: Vec<GlobalType> =
            module.globals.iter().map(|global| global.0).collect();
        let mut globals: Vec<GlobalType> = module
            .imports
            .iter()
            .filter_map(|import| match import.desc {
                ImportDesc::Global(global_type) => Some(global_type),
                _ => None,
            })
            .collect();
        globals.extend_from_slice(&mod_global_types);
        let globals = globals;

        let elems: Vec<RefType> = module.elems.iter().map(|elem| elem.elem_type).collect();
        let datas: Vec<bool> = vec![true; module.datas.len()];

        return WasmContext {
            module,
            funcs,
            tables,
            mems,
            globals,
            elems,
            datas,
            locals: Vec::new(),
            labels: Vec::new(),
            returns: None,
            refs: Vec::new(),
            type_stack: Vec::new(),
            ctrl_stack: Vec::new(),
        };
    }

    //Documentation:
    //
    pub fn validate(&mut self) -> bool {
        //Adjust context
        let all_globals = self.globals.clone();
        self.globals.truncate(self.module.globals.len());

        //Check functypes
        if !self.validate_func_types() {
            return false;
        }

        if !self.validate_tables()
            | !self.validate_mems()
            | !self.validate_globals()
            | !self.validate_elems()
            | !self.validate_datas()
        {
            return false;
        }

        self.globals = all_globals;

        for func in &self.module.funcs {
            if !self.validate_func(&func) {
                return false;
            }
        }

        if !self.validate_start() | !self.validate_imports() | !self.validate_exports() {
            return false;
        }
        true
    }

    fn validate_limits(&self, limits: Limits, range: u32) -> bool {
        if limits.0 > range {
            return false;
        }
        if let Some(max) = limits.1 {
            if max > range || max < limits.0 {
                return false;
            }
        }
        true
    }

    fn validate_block_type(&self, block_type: BlockType) -> bool {
        match block_type {
            BlockType::TypeIdx(t) => (t as usize) < self.module.types.len(),
            BlockType::ValType(_) => true,
        }
    }

    fn validate_func_types(&self) -> bool {
        for func_type in &self.module.types {
            if !self.validate_func_type(func_type) {
                return false;
            }
        }
        true
    }

    fn validate_func_type(&self, func_type: &FuncType) -> bool {
        true
    }

    fn validate_table_type(&self, table_type: &TableType) -> bool {
        self.validate_limits(table_type.0, u32::MAX)
    }

    fn validate_mem_type(&self, limit: Limits) -> bool {
        self.validate_limits(limit, 1 << 16)
    }

    fn validate_global_type(&self, global_type: &GlobalType) -> bool {
        true
    }

    fn validate_external_type(&self, extern_type: ExternType) -> bool {
        match extern_type {
            ExternType::FuncType(func_type) => self.validate_func_type(&func_type),
            ExternType::TableType(table_type) => self.validate_table_type(&table_type),
            ExternType::MemType(limits) => self.validate_mem_type(limits),
            ExternType::GlobalType(global_type) => self.validate_global_type(&global_type),
        }
    }

    fn subtype_limits(&self, subtyped_limits: Limits, base_limits: Limits) -> bool {
        if subtyped_limits.0 < base_limits.0 {
            return false;
        }

        if base_limits.1.is_some() {
            if subtyped_limits.1.is_none() {
                return false;
            }
            if subtyped_limits.1 > base_limits.1 {
                return false;
            }
        }
        return true;
    }

    fn subtype_func_type(&self, subtyped_func_type: FuncType, base_func_type: FuncType) -> bool {
        subtyped_func_type == base_func_type
    }

    fn subtype_table_type(
        &self,
        subtyped_table_type: TableType,
        base_func_type: TableType,
    ) -> bool {
        if !self.subtype_limits(subtyped_table_type.0, base_func_type.0) {
            return false;
        }
        subtyped_table_type.1 == base_func_type.1
    }

    //TODO should be removed, acts as a convenience for memory types
    fn subtype_mem_type(&self, subtyped_limits: Limits, base_limits: Limits) -> bool {
        self.subtype_limits(subtyped_limits, base_limits)
    }

    fn subtype_global_type(
        &self,
        subtyped_global_type: GlobalType,
        base_global_type: GlobalType,
    ) -> bool {
        subtyped_global_type == base_global_type
    }

    //Documentation:
    //  https://webassembly.github.io/spec/core/valid/modules.html#functions
    fn validate_func(&mut self, func: &Func) -> bool {
        let Some(func_type) = self.module.types.get(func.typeidx as usize) else {
            return false;
        };
        self.locals = func_type.0.clone();
        self.locals.extend_from_slice(&func.locals);

        self.labels.clear();
        self.labels.push(func_type.1.clone());

        self.returns = Some(func_type.1.clone());

        self.validate_expr(&func.body)
    }

    //Documentation:
    //  https://webassembly.github.io/spec/core/valid/instructions.html#expressions
    fn validate_expr(&mut self, expr: &Expr) -> bool {
        for instr in &expr.0 {
            if !self.validate_instr(&instr) {
                return false;
            }
        }
        true
    }

    //TODO 1 Should return a Result with resulttype
    //Documentation:
    //  https://webassembly.github.io/spec/core/valid/instructions.html#constant-expressions
    fn validate_const_expr(&mut self, expr: &Expr) -> bool {
        for instr in &expr.0 {
            match instr {
                Instr::I32Const(_)
                | Instr::I64Const(_)
                | Instr::F32Const(_)
                | Instr::F64Const(_)
                | Instr::RefNull(_)
                | Instr::RefFunc(_) => {}
                Instr::GlobalGet(x) => {
                    if let Some(global) = self.module.globals.get(*x as usize) {
                        if !(global.0 .0 == Mut::Const) {
                            return false;
                        }
                    }
                }
                _ => return false,
            }
            if !self.validate_instr(&instr) {
                return false;
            }
        }
        true
    }

    fn validate_tables(&self) -> bool {
        for table in &self.module.tables {
            if !self.validate_table_type(table) {
                return false;
            }
        }
        true
    }

    fn validate_mems(&self) -> bool {
        for mem in &self.module.mems {
            if !self.validate_mem_type(*mem) {
                return false;
            }
        }
        true
    }

    fn validate_globals(&mut self) -> bool {
        for global in &self.module.globals {
            if !self.validate_global_type(&global.0) {
                return false;
            }

            if !self.validate_const_expr(&global.1) {
                return false;
            }
        }
        true
    }

    fn validate_elems(&mut self) -> bool {
        for elem in &self.module.elems {
            if !self.validate_elem(elem) {
                return false;
            }
        }
        true
    }

    fn validate_elem(&mut self, elem: &Elem) -> bool {
        for i in &elem.init {
            //TODO 1 Should be able to check the result type after validating
            if !self.validate_const_expr(i) {
                return false;
            }
        }

        match &elem.mode {
            ElemMode::Active(x, offset) => {
                let Some(table) = self.tables.get(*x as usize) else {
                    return false;
                };

                //TODO 1 should be able to check the result type is [i32]
                if !self.validate_const_expr(&offset) {
                    return false;
                }
            }
            _ => {}
        }

        true
    }

    fn validate_datas(&mut self) -> bool {
        for data in &self.module.datas {
            if !self.validate_data(data) {
                return false;
            }
        }
        true
    }

    fn validate_data(&mut self, data: &Data) -> bool {
        match &data.mode {
            DataMode::Passive => {}
            DataMode::Active(b, expr) => {
                if self.mems.get(*b as usize).is_none() {
                    return false;
                }
                //TODO 1 should be able to check the result type is [i32]
                if !self.validate_const_expr(&expr) {
                    return false;
                }
            }
        }
        true
    }

    fn validate_start(&mut self) -> bool {
        let Some(idx) = self.module.start else {
            return false;
        };
        match self.funcs.get(idx as usize) {
            Some(FuncType(param, ret)) => {
                if !param.is_empty() || !ret.is_empty() {
                    return false;
                }
            }
            None => return false,
        }
        true
    }

    fn validate_exports(&self) -> bool {
        for export in &self.module.exports {
            match export.desc {
                ExportDesc::Func(idx) if self.funcs.get(idx as usize).is_some() => {}
                ExportDesc::Table(idx) if self.tables.get(idx as usize).is_some() => {}
                ExportDesc::Mem(idx) if self.mems.get(idx as usize).is_some() => {}
                ExportDesc::Global(idx) if self.globals.get(idx as usize).is_some() => {}
                _ => return false,
            }
        }
        true
    }

    fn validate_imports(&self) -> bool {
        for import in &self.module.imports {
            match import.desc {
                ImportDesc::Func(x) => return self.module.types.get(x as usize).is_some(),
                ImportDesc::Table(table_type) => return self.validate_table_type(&table_type),
                ImportDesc::Mem(limits) => return self.validate_mem_type(limits),
                ImportDesc::Global(global_type) => return self.validate_global_type(&global_type),
                _ => return false,
            }
        }
        true
    }

    fn func_type_from_block_type(&self, block_type: &BlockType) -> Option<FuncType> {
        match block_type {
            BlockType::TypeIdx(type_idx) => self.module.types.get(*type_idx as usize).cloned(),
            BlockType::ValType(Some(val_type)) => {
                Some(FuncType(Vec::new(), vec![val_type.clone()]))
            }
            BlockType::ValType(None) => Some(FuncType(Vec::new(), Vec::new())),
        }
    }

    fn push_val(&mut self, val_type: Option<ValType>) {
        self.type_stack.push(val_type)
    }

    fn pop_val(&mut self) -> Result<Option<ValType>, ()> {
        if let Some(top_ctrl) = self.ctrl_stack.last() {
            if self.type_stack.len() == top_ctrl.height && top_ctrl.unreachable {
                return Ok(None);
            }
            if self.type_stack.len() == top_ctrl.height {
                return Err(());
            }
        }
        let Some(val_type) = self.type_stack.pop() else {
            return Err(());
        };
        Ok(val_type)
    }

    fn pop_explicit_val(&mut self, expect: Option<&ValType>) -> Result<Option<ValType>, ()> {
        let Ok(actual) = self.pop_val() else {
            return Err(());
        };
        if actual.as_ref() != expect && actual.is_none() && expect.is_some() {
            return Err(());
        }
        Ok(actual)
    }

    fn push_vals(&mut self, types: Vec<ValType>) {
        for t in types {
            self.push_val(Some(t));
        }
    }

    fn pop_vals(&mut self, types: &[ValType]) -> Result<Vec<Option<ValType>>, ()> {
        let mut popped = Vec::new();
        for t in types.iter().rev() {
            let Ok(val) = self.pop_explicit_val(Some(t)) else {
                return Err(());
            };
            popped.push(val);
        }
        return Ok(popped);
    }

    fn push_ctrl(&mut self, instr: Instr, start_types: Vec<ValType>, end_types: Vec<ValType>) {
        self.ctrl_stack.push(CtrlFrame {
            instr,
            start_types,
            end_types,
            height: self.type_stack.len(),
            unreachable: false,
        });
    }

    fn pop_ctrl(&mut self) -> Option<CtrlFrame> {
        let frame = self.ctrl_stack.pop()?;
        let _ = self.pop_vals(&frame.end_types);
        if self.type_stack.len() != frame.height {
            return None;
        }
        Some(frame)
    }

    fn unreachable(&mut self) {
        let Some(top_frame) = self.ctrl_stack.last_mut() else {
            return;
        };
        self.type_stack.truncate(top_frame.height);
        top_frame.unreachable = true;
    }

    fn validate_instr(&mut self, instr: &Instr) -> bool {
        match instr {
            Instr::Nop => {}
            Instr::Unreachable => {
                self.unreachable();
                panic!("No idea how unreachable works");
            }
            Instr::Block(block_type) => {
                let Some(func_type) = self.func_type_from_block_type(&block_type) else {
                    return false;
                };

                self.labels.push(func_type.1.clone());

                let _ = self.pop_vals(&func_type.0);
                self.push_ctrl(Instr::Block(*block_type), func_type.0, func_type.1);
            }
            Instr::Loop(block_type) => {
                let Some(func_type) = self.func_type_from_block_type(&block_type) else {
                    return false;
                };

                self.labels.push(func_type.0.clone());
                self.pop_vals(&func_type.0);
                self.push_ctrl(Instr::Loop(*block_type), func_type.0, func_type.1);
            }
            Instr::If(block_type) => {
                let Some(func_type) = self.func_type_from_block_type(&block_type) else {
                    return false;
                };

                self.labels.push(func_type.0.clone());
                let _ = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)));
                let _ = self.pop_vals(&func_type.0);
                self.push_ctrl(Instr::If(*block_type), func_type.0, func_type.1);
            }
            Instr::Else => {
                //Pop ctrl
                let Some(frame) = self.pop_ctrl() else {
                    return false;
                };
                //Check if
                if mem::discriminant(&frame.instr)
                    != mem::discriminant(&Instr::If(BlockType::TypeIdx(0)))
                {
                    return false;
                }
                //push new ctrl
                self.push_ctrl(Instr::Else, frame.start_types, frame.end_types);
            }
            Instr::End => {
                let _ = self.labels.pop();
                let _ = self.pop_ctrl();
            }
            Instr::Br(idx) => {
                if ((self.ctrl_stack.len() as i64) - 1 - (*idx as i64)) < 0 {
                    return false;
                }
                let labels_len = self.labels.len() - 1;
                let Some(result_types) = self.labels.get(labels_len - *idx as usize) else {
                    return false;
                };

                let _ = self.pop_vals(&result_types.clone());
                self.unreachable();
            }
            Instr::BrIf(idx) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)));
                if ((self.ctrl_stack.len() as i64) - 1 - (*idx as i64)) < 0 {
                    return false;
                }
                let labels_len = self.labels.len() - 1;
                let Some(label) = self.labels.get(labels_len - *idx as usize) else {
                    return false;
                };
                let label = label.to_vec();

                let _ = self.pop_vals(&label);
                self.push_vals(label.to_vec());
            }
            Instr::BrTable(label_idxs, label_idx) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)));

                for label_idx in label_idxs {
                    let Some(label) = self.labels.get(*label_idx as usize) else {
                        return false;
                    };
                    let label = label.to_vec();

                    self.pop_vals(&label);
                    self.push_vals(label);
                }

                let Some(label) = self.labels.get(*label_idx as usize) else {
                    return false;
                };
                let label = label.to_vec();
                self.pop_vals(&label);
                self.unreachable();
            }
            Instr::Return => {
                let Some(returns) = self.returns.take() else {
                    return false;
                };
                let _ = self.pop_vals(&returns);
                self.pop_ctrl();
            }
            Instr::Call(func_idx) => {
                let Some(func_type) = self.funcs.get(*func_idx as usize) else {
                    return false;
                };
                let func_type = func_type.clone();

                let _ = self.pop_vals(&func_type.0);
                self.push_vals(func_type.1);
            }
            Instr::CallIndirect(table_idx, type_idx) => {
                let _ = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)));

                let Some(TableType(_, RefType::FuncRef)) = self.tables.get(*table_idx as usize)
                else {
                    return false;
                };
                let Some(func_type) = self.module.types.get(*type_idx as usize) else {
                    return false;
                };
                let func_type = func_type.clone();

                let _ = self.pop_vals(&func_type.0);
                self.push_vals(func_type.1);
            }
            Instr::I32Const(_) => {
                self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64Const(_) => {
                self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::F32Const(_) => {
                self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::F64Const(_) => {
                self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::I32Eqz => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32WrapI64 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32TruncF32S
            | Instr::I32TruncF32U
            | Instr::I32ReinterpretF32
            | Instr::I32TruncSatF32S
            | Instr::I32TruncSatF32U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32TruncF64S
            | Instr::I32TruncF64U
            | Instr::I32TruncSatF64S
            | Instr::I32TruncSatF64U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64ExtendI32S | Instr::I64ExtendI32U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::I64TruncF32S
            | Instr::I64TruncF32U
            | Instr::I64TruncSatF32S
            | Instr::I64TruncSatF32U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::I64TruncF64S
            | Instr::I64TruncF64U
            | Instr::I64TruncSatF64S
            | Instr::I64TruncSatF64U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::F32ConvertI32S | Instr::F32ConvertI32U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F32ConvertI64S | Instr::F32ConvertI64U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F32DemoteF64 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F64ConvertI32S | Instr::F64ConvertI32U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::F64ConvertI64S | Instr::F64ConvertI64U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::F64PromoteF32 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::I64ReinterpretF64 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::F32ReinterpretI32 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F64ReinterpretI64 => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::I32Extend8S | Instr::I32Extend16S => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64Extend8S | Instr::I64Extend16S | Instr::I64Extend32S => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::I32Eq
            | Instr::I32Ne
            | Instr::I32LtS
            | Instr::I32LtU
            | Instr::I32GtS
            | Instr::I32GtU
            | Instr::I32LeS
            | Instr::I32LeU
            | Instr::I32GeS
            | Instr::I32GeU => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32Clz | Instr::I32Ctz | Instr::I32PopCnt => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32Add
            | Instr::I32Sub
            | Instr::I32Mul
            | Instr::I32DivS
            | Instr::I32DivU
            | Instr::I32RemS
            | Instr::I32RemU
            | Instr::I32And
            | Instr::I32Or
            | Instr::I32Xor
            | Instr::I32Shl
            | Instr::I32ShrS
            | Instr::I32ShrU
            | Instr::I32Rotl
            | Instr::I32Rotr => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                if lhs.is_none() || rhs.is_none() {
                    return false;
                }
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64Eqz => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64Eq
            | Instr::I64Ne
            | Instr::I64LtS
            | Instr::I64LtU
            | Instr::I64GtS
            | Instr::I64GtU
            | Instr::I64LeS
            | Instr::I64LeU
            | Instr::I64GeS
            | Instr::I64GeU => {}
            Instr::I64Clz | Instr::I64Ctz | Instr::I64PopCnt => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64Add
            | Instr::I64Sub
            | Instr::I64Mul
            | Instr::I64DivS
            | Instr::I64DivU
            | Instr::I64RemS
            | Instr::I64RemU
            | Instr::I64And
            | Instr::I64Or
            | Instr::I64Xor
            | Instr::I64Shl
            | Instr::I64ShrS
            | Instr::I64ShrU
            | Instr::I64Rotl
            | Instr::I64Rotr => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };
                if lhs.is_none() || rhs.is_none() {
                    return false;
                }
                let _ = self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::F32Eq
            | Instr::F32Ne
            | Instr::F32Lt
            | Instr::F32Gt
            | Instr::F32Le
            | Instr::F32Ge => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::F32Abs
            | Instr::F32Neg
            | Instr::F32Ceil
            | Instr::F32Floor
            | Instr::F32Trunc
            | Instr::F32Nearest
            | Instr::F32Sqrt => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F32Add
            | Instr::F32Sub
            | Instr::F32Mul
            | Instr::F32Div
            | Instr::F32Min
            | Instr::F32Max
            | Instr::F32CopySign => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };
                if lhs.is_none() || rhs.is_none() {
                    return false;
                }
                let _ = self.push_val(Some(ValType::NumType(NumType::F32)));
            }
            Instr::F64Eq
            | Instr::F64Ne
            | Instr::F64Lt
            | Instr::F64Gt
            | Instr::F64Le
            | Instr::F64Ge => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::F64Abs
            | Instr::F64Neg
            | Instr::F64Ceil
            | Instr::F64Floor
            | Instr::F64Trunc
            | Instr::F64Nearest
            | Instr::F64Sqrt => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::F64Add
            | Instr::F64Sub
            | Instr::F64Mul
            | Instr::F64Div
            | Instr::F64Min
            | Instr::F64Max
            | Instr::F64CopySign => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };
                if lhs.is_none() || rhs.is_none() {
                    return false;
                }
                let _ = self.push_val(Some(ValType::NumType(NumType::F64)));
            }
            Instr::RefNull(val_type) => {
                let _ = self.push_val(Some(*val_type));
            }
            Instr::RefIsNull => {
                let Ok(lhs) = self.pop_val() else {
                    return false;
                };
                let _ = self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::RefFunc(func_idx) => {
                if self.funcs.len() < (*func_idx as usize)
                    && self.refs.iter().find(|&i| i == func_idx).is_some()
                {
                    return false;
                }

                let _ = self.push_val(Some(ValType::RefType(RefType::FuncRef)));
            }
            Instr::V128Const(_) => {
                self.push_val(Some(ValType::VecType));
            }
            Instr::V128Not => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::V128And | Instr::V128AndNot | Instr::V128Or | Instr::V128Xor => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::V128BitSelect => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::V128AnyTrue => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I8x16Swizzle => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I8x16Shuffle(lane_idxs) => {
                for idx in lane_idxs {
                    if *idx > 32 {
                        return false;
                    }
                }

                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I8x16Splat | Instr::I16x8Splat | Instr::I32x4Splat => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I64x2Splat => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::F32x4Splat => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::F64x2Splat => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I8x16ExtractLaneS(lane_idx) | Instr::I8x16ExtractLaneU(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I16x8ExtractLaneS(lane_idx) | Instr::I16x8ExtractLaneU(lane_idx) => {
                if *lane_idx > 8 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I32x4ExtractLane(lane_idx) | Instr::F32x4ExtractLane(lane_idx) => {
                if *lane_idx > 4 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::NumType(NumType::I32)));
            }
            Instr::I64x2ExtractLane(lane_idx) | Instr::F64x2ExtractLane(lane_idx) => {
                if *lane_idx > 2 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::NumType(NumType::I64)));
            }
            Instr::I8x16ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I16x8ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I32x4ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I64x2ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I64))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::F32x4ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::F64x2ReplaceLane(lane_idx) => {
                if *lane_idx > 16 {
                    return false;
                }
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::F64))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I8x16Abs
            | Instr::I8x16Neg
            | Instr::I8x16Popcnt
            | Instr::I16x8Abs
            | Instr::I16x8Neg
            | Instr::I32x4Abs
            | Instr::I32x4Neg
            | Instr::I64x2Abs
            | Instr::I64x2Neg
            | Instr::F32x4Abs
            | Instr::F32x4Neg
            | Instr::F32x4Sqrt
            | Instr::F64x2Abs
            | Instr::F64x2Neg
            | Instr::F64x2Sqrt => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I8x16Add
            | Instr::I8x16Sub
            | Instr::I8x16MinS
            | Instr::I8x16MinU
            | Instr::I8x16MaxS
            | Instr::I8x16MaxU
            | Instr::I8x16AvgrU
            | Instr::I16x8Add
            | Instr::I16x8Sub
            | Instr::I16x8Mul
            | Instr::I16x8MinS
            | Instr::I16x8MinU
            | Instr::I16x8MaxS
            | Instr::I16x8MaxU
            | Instr::I16x8AvgrU
            | Instr::I16x8Q15MulrSatS
            | Instr::I32x4Add
            | Instr::I32x4Sub
            | Instr::I32x4Mul
            | Instr::I32x4MinS
            | Instr::I32x4MinU
            | Instr::I32x4MaxS
            | Instr::I32x4MaxU
            | Instr::I64x2Add
            | Instr::I64x2Sub
            | Instr::I64x2Mul
            | Instr::F32x4Add
            | Instr::F32x4Sub
            | Instr::F32x4Mul
            | Instr::F32x4Div
            | Instr::F64x2Add
            | Instr::F64x2Sub
            | Instr::F64x2Muk
            | Instr::F64x2Div
            | Instr::I8x16Eq
            | Instr::I8x16Ne
            | Instr::I8x16LtS
            | Instr::I8x16LtU
            | Instr::I8x16GtS
            | Instr::I8x16GtU
            | Instr::I8x16LeS
            | Instr::I8x16LeU
            | Instr::I8x16GeS
            | Instr::I8x16GeU
            | Instr::I16x8Eq
            | Instr::I16x8Ne
            | Instr::I16x8LtS
            | Instr::I16x8LtU
            | Instr::I16x8GtS
            | Instr::I16x8GtU
            | Instr::I16x8LeS
            | Instr::I16x8LeU
            | Instr::I16x8GeS
            | Instr::I16x8GeU
            | Instr::I32x4Eq
            | Instr::I32x4Ne
            | Instr::I32x4LtS
            | Instr::I32x4LtU
            | Instr::I32x4GtS
            | Instr::I32x4GtU
            | Instr::I32x4LeS
            | Instr::I32x4LeU
            | Instr::I32x4GeS
            | Instr::I32x4GeU
            | Instr::F32x4Eq
            | Instr::F32x4Ne
            | Instr::F32x4Lt
            | Instr::F32x4Gt
            | Instr::F32x4Le
            | Instr::F32x4Ge
            | Instr::F64x2Eq
            | Instr::F64x2Ne
            | Instr::F64x2Lt
            | Instr::F64x2Gt
            | Instr::F64x2Le
            | Instr::F64x2Ge => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I8x16Shl
            | Instr::I8x16ShrS
            | Instr::I8x16ShrU
            | Instr::I16x8Shl
            | Instr::I16x8ShrS
            | Instr::I16x8ShrU
            | Instr::I32x4Shl
            | Instr::I32x4ShrS
            | Instr::I32x4ShrU
            | Instr::I64x2Shl
            | Instr::I64x2ShrS
            | Instr::I64x2ShrU => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I8x16AllTrue | Instr::I16x8AllTrue | Instr::I32x4AllTrue => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::NumType(NumType::I32)));
            }

            Instr::I16x8ExtendLowI8x16S
            | Instr::I16x8ExtendHighI8x16S
            | Instr::I16x8ExtendLowI8x16U
            | Instr::I16x8ExtendHighI8x16U
            | Instr::I32x4ExtendLowI16x8S
            | Instr::I32x4ExtendHighI16x8S
            | Instr::I32x4ExtendLowI16x8U
            | Instr::I32x4ExtendHighI16x8U
            | Instr::I64x2ExtendLowi32x4S
            | Instr::I64x2ExtendHighi32x4S
            | Instr::I64x2ExtendLowi32x4U
            | Instr::I64x2ExtendHighi32x4U
            | Instr::I32x4TruncSatF32x4S
            | Instr::I32x4TruncSatF32x4U
            | Instr::F32x4ConvertI32x4S
            | Instr::F32x4ConvertI32x4U
            | Instr::F32x4DemoteF64x2Zero
            | Instr::F64x2PromoteLowF32x4
            | Instr::I32x4TruncSatF64x2SZero
            | Instr::I32x4TruncSatF64x2UZero
            | Instr::F64x2ConvertLowI32x4S
            | Instr::F64x2ConvertLowI32x4U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I8x16NarrowI16x8U | Instr::I16x8NarrowI32x4S | Instr::I16x8NarrowI32x4U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I8x16Bitmask
            | Instr::I16x8Bitmask
            | Instr::I32x4Bitmask
            | Instr::I64x2Bitmask => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::NumType(NumType::I32)));
            }

            Instr::I32x4DotI16x8S => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }

            Instr::I32x4ExtMulLowI16x8S
            | Instr::I32x4ExtMulHighI16x8S
            | Instr::I32x4ExtMulLowI16x8U
            | Instr::I32x4ExtMulHighI16x8U
            | Instr::I64x2ExtMulLowI32x4S
            | Instr::I64x2ExtMulHighI32x4S
            | Instr::I64x2ExtMulLowI32x4U
            | Instr::I64x2ExtMutHighI32x4U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                let Ok(rhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::I16x8ExtAddPairwiseI8x16S
            | Instr::I16x8ExtAddPairwiseI8x16U
            | Instr::I32x4ExtAddPairwiseI16x8S
            | Instr::I32x4ExtAddPairwiseI16x8U => {
                let Ok(lhs) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };

                self.push_val(Some(ValType::VecType));
            }
            Instr::Drop => {
                self.pop_val();
            }
            Instr::Select(Some(val_type)) => {
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(val_type)) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(val_type)) else {
                    return false;
                };

                self.push_val(Some(*val_type));
            }
            Instr::Select(None) => {
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(t1) = self.pop_val() else {
                    return false;
                };
                let Ok(t2) = self.pop_val() else {
                    return false;
                };
                if t1 != t2 {
                    return false;
                }

                self.push_val(t1);
            }
            Instr::LocalGet(idx) => match self.locals.get(*idx as usize) {
                Some(local) => {
                    self.push_val(Some(*local));
                }
                None => {
                    return false;
                }
            },
            Instr::LocalSet(idx) => match self.locals.get(*idx as usize) {
                Some(local) => {
                    self.pop_explicit_val(Some(&local.clone())).unwrap();
                }
                None => {
                    return false;
                }
            },
            Instr::LocalTee(idx) => match self.locals.get(*idx as usize) {
                Some(local) => {
                    let local: ValType = local.clone();
                    self.pop_explicit_val(Some(&local)).unwrap();
                    self.push_val(Some(local));
                }
                None => {
                    return false;
                }
            },
            Instr::GlobalGet(idx) => match self.globals.get(*idx as usize) {
                Some(global) => {
                    self.push_val(Some(global.1));
                }
                None => {
                    return false;
                }
            },
            Instr::GlobalSet(idx) => match self.globals.get(*idx as usize) {
                Some(global) => {
                    let global: GlobalType = *global;
                    if global.0 != Mut::Var {
                        return false;
                    }
                    self.pop_explicit_val(Some(&global.1)).unwrap();
                }
                None => return false,
            },
            Instr::TableGet(idx) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                    .unwrap();
                match self.tables.get(*idx as usize) {
                    Some(table) => {
                        self.push_val(Some(ValType::RefType(table.1)));
                    }
                    None => return false,
                }
            }
            Instr::TableSet(idx) => match self.tables.get(*idx as usize) {
                Some(table) => {
                    self.pop_explicit_val(Some(&ValType::RefType(table.1)))
                        .unwrap();
                    self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                        .unwrap();
                }
                None => return false,
            },
            Instr::TableSize(idx) => match self.tables.get(*idx as usize) {
                Some(table) => {
                    self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                        .unwrap();
                }
                None => return false,
            },
            Instr::TableGrow(idx) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                    .unwrap();
                match self.tables.get(*idx as usize) {
                    Some(table) => {
                        self.pop_explicit_val(Some(&ValType::RefType(table.1)))
                            .unwrap();
                        self.push_val(Some(ValType::NumType(NumType::I32)));
                    }
                    None => return false,
                }
            }
            Instr::TableFill(idx) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                    .unwrap();
                match self.tables.get(*idx as usize) {
                    Some(table) => {
                        self.pop_explicit_val(Some(&ValType::RefType(table.1)))
                            .unwrap();
                        self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                            .unwrap();
                    }
                    None => return false,
                }
            }
            Instr::TableCopy(idx1, idx2) => {
                self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                    .unwrap();
                match (
                    self.tables.get(*idx1 as usize),
                    self.tables.get(*idx2 as usize),
                ) {
                    (Some(table1), Some(table2)) => {
                        if table1.1 != table2.1 {
                            return false;
                        }
                        self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                            .unwrap();
                        self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                            .unwrap();
                        self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                            .unwrap();
                    }
                    _ => return false,
                }
            }
            Instr::TableInit(idx1, idx2) => match (
                self.tables.get(*idx1 as usize),
                self.elems.get(*idx2 as usize),
            ) {
                (Some(table), Some(elem)) => {
                    if table.1 != *elem {
                        return false;
                    }
                    self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                        .unwrap();
                    self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                        .unwrap();
                    self.pop_explicit_val(Some(&ValType::NumType(NumType::I32)))
                        .unwrap();
                }
                _ => return false,
            },
            Instr::ElemDrop(idx) => match self.elems.get(*idx as usize) {
                Some(_) => {}
                None => return false,
            },
            Instr::I32Load(mem_arg)
            | Instr::I64Load(mem_arg)
            | Instr::F32Load(mem_arg)
            | Instr::F64Load(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::I32Load8S(mem_arg)
            | Instr::I32Load8U(mem_arg)
            | Instr::I32Load16S(mem_arg)
            | Instr::I32Load16U(mem_arg)
            | Instr::I64Load8S(mem_arg)
            | Instr::I64Load8U(mem_arg)
            | Instr::I64Load16S(mem_arg)
            | Instr::I64Load16U(mem_arg)
            | Instr::I64Load32S(mem_arg)
            | Instr::I64Load32U(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::I32Store(mem_arg)
            | Instr::I64Store(mem_arg)
            | Instr::F32Store(mem_arg)
            | Instr::F64Store(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::I32Store8(mem_arg)
            | Instr::I32Store16(mem_arg)
            | Instr::I64Store8(mem_arg)
            | Instr::I64Store16(mem_arg)
            | Instr::I64Store32(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::V128Load8x8S(mem_arg)
            | Instr::V128Load8x8U(mem_arg)
            | Instr::V128Load16x4S(mem_arg)
            | Instr::V128Load16x4U(mem_arg)
            | Instr::V128Load32x2S(mem_arg)
            | Instr::V128Load32x2U(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::V128Load8Splat(mem_arg)
            | Instr::V128Load16Splat(mem_arg)
            | Instr::V128Load32Splat(mem_arg)
            | Instr::V128Load64Splat(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::V128Load32Zero(mem_arg) | Instr::V128Load64Zero(mem_arg) => {
                panic!("UNIMPLEMENTED");
            }

            Instr::V128Load8Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 8) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (8 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                self.push_val(Some(ValType::VecType));
            }

            Instr::V128Load16Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 16) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (16 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                self.push_val(Some(ValType::VecType));
            }

            Instr::V128Load32Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 32) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (32 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                self.push_val(Some(ValType::VecType));
            }

            Instr::V128Load64Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 64) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (64 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
                self.push_val(Some(ValType::VecType));
            }

            Instr::V128Store8Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 8) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (8 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
            }

            Instr::V128Store16Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 16) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (16 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
            }

            Instr::V128Store32Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 32) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if (1 << (mem_arg.align) <= (32 / 8)) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
            }

            Instr::V128Store64Lane(mem_arg, lane_idx) => {
                if *lane_idx < (128 / 64) {
                    return false;
                }
                if self.mems.get(0).is_none() {
                    return false;
                }
                if 1 << (mem_arg.align) <= (64 / 8) {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::VecType)) else {
                    return false;
                };
            }

            Instr::MemorySize => {
                if self.mems.get(0).is_none() {
                    return false;
                }
                self.push_val(Some(ValType::NumType(NumType::I32)));
            }

            Instr::MemoryGrow => {
                if self.mems.get(0).is_none() {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                self.push_val(Some(ValType::NumType(NumType::I32)));
            }

            Instr::MemoryFill => {
                if self.mems.get(0).is_none() {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
            }
            Instr::MemoryCopy => {
                if self.mems.get(0).is_none() {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
            }
            Instr::MemoryInit(data_idx) => {
                if self.mems.get(0).is_none() {
                    return false;
                }
                if self.datas.get((*data_idx) as usize).is_some() {
                    return false;
                }
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
                let Ok(_) = self.pop_explicit_val(Some(&ValType::NumType(NumType::I32))) else {
                    return false;
                };
            }
            Instr::DataDrop(data_idx) => {
                if self.datas.get((*data_idx) as usize).is_some() {
                    return false;
                }
            }
            _ => (),
        };
        true
    }

    fn validate_call_indirect(&self, (x, y): (TableIdx, TypeIdx)) -> bool {
        let Some(TableType(limits, ref_type)) = self.module.tables.get(x as usize) else {
            return false;
        };

        if *ref_type != RefType::FuncRef {
            return false;
        }

        self.module.types.len() < (y as usize)
    }
}
