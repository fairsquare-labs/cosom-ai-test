mod math;
pub mod matrix;
pub mod activation;
pub mod nl;
pub mod nn;
pub mod sample;
pub mod cost;
mod utils;

mod prelude {
    pub use alloc::boxed::Box;
    pub use alloc::string::String;
    pub use alloc::vec::Vec;
    pub use alloc::borrow::ToOwned;
    pub use alloc::string::ToString;
    pub use core::option::Option;
}
