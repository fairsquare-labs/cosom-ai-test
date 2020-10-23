pub mod sigmoid;
pub mod identity;
pub mod hyperbolictangent;
pub mod softplus;
pub mod softmax;
pub mod rectifiedlinearunit;
pub mod leakyrectifiedlinearunit;

pub use self::sigmoid::Sigmoid;
pub use self::identity::Identity;
pub use self::hyperbolictangent::HyperbolicTangent;
pub use self::softplus::SoftPlus;
pub use self::softmax::SoftMax;
pub use self::rectifiedlinearunit::RectifiedLinearUnit;
pub use self::leakyrectifiedlinearunit::LeakyRectifiedLinearUnit;
use super::prelude::*;
use serde::{Serialize,Deserialize};
use schemars::JsonSchema;

#[derive(Serialize,Deserialize,JsonSchema,Clone,PartialEq)]
pub enum ActivationKind{
    Sigmoid,
    HyperbolicTangent,
    SoftPlus,
    SoftMax,
    RectifiedLinearUnit,
    LeakyRectifiedLinearUnit,
    Identity
}
#[derive(Serialize,Deserialize,JsonSchema,Clone,PartialEq)]
pub struct ActivationArgu{
    kind: ActivationKind,
    argu: Vec<f64>
}

impl ActivationArgu{
    pub fn new(kind:ActivationKind,argu: Vec<f64>) ->Self{
        ActivationArgu{
            kind,
            argu
        }      
    }
    pub fn get_activation(&self) -> Box<dyn Activation>{
        match self.kind{
            ActivationKind::Sigmoid => {
                Box::new(Sigmoid::new())
            }
            ActivationKind::HyperbolicTangent => {
                Box::new(HyperbolicTangent::new())
            }
            ActivationKind::SoftPlus => {
                Box::new(SoftPlus::new())
            }
            ActivationKind::SoftMax => {
                Box::new(SoftMax::new())
            }
            ActivationKind::RectifiedLinearUnit => {
                Box::new(RectifiedLinearUnit::new())
            }
            ActivationKind::LeakyRectifiedLinearUnit => {
                Box::new(LeakyRectifiedLinearUnit::new(self.argu[0]))
            }
            ActivationKind::Identity =>{
                Box::new(Identity::new())
            }
        }
    }
}

/// Activation trait
pub trait Activation {
    // the function itself
    fn calc(&self, x: Vec<f64>) -> Vec<f64>;
    // Derivative
    fn derivative(&self, x: Vec<f64>) -> Vec<f64>;
}

// serialize_trait_object!(Activation);

// impl<'de> Deserialize<'de> for Box<dyn Activation> {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         deserializer.de
//     }
// }