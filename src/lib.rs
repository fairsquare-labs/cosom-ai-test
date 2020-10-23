extern crate alloc;
pub mod contract;
pub mod error;
pub mod msg;
pub mod state;
pub mod engine;

#[cfg(target_arch = "wasm32")]
cosmwasm_std::create_entry_points!(contract);
