//! Runtime selection of the tensor compute device.

use candle_core::{Device, Result};

/// Selects a device from `MOPAPPA_DEVICE`.
///
/// Accepted values are `cpu`, `cuda`, and `auto`. `auto` is the default and
/// selects CUDA when the crate was compiled with the `cuda` feature.
pub fn selected_device() -> Result<Device> {
    match std::env::var("MOPAPPA_DEVICE")
        .unwrap_or_else(|_| "auto".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => Ok(Device::Cpu),
        "cuda" => cuda_device(),
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                cuda_device()
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Device::Cpu)
            }
        }
        value => {
            candle_core::bail!("unsupported MOPAPPA_DEVICE {value:?}; expected cpu, cuda, or auto")
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Result<Device> {
    Device::new_cuda(0)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> Result<Device> {
    candle_core::bail!("CUDA requested but mopappa was built without the cuda feature")
}

/// Short human-readable backend name for experiment output.
pub fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}
