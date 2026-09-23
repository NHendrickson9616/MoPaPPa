use mopappa::{
    engine::device::{device_name, selected_device},
    model::experiment::run_tiny_overfit_on,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = selected_device()?;
    println!("Device: {}", device_name(&device));
    let report = run_tiny_overfit_on(&device)?;
    println!("{report:#?}");

    let substantially_improved =
        report.final_loss < 0.1 && report.final_loss < report.initial_loss * 0.25;
    if !substantially_improved {
        return Err(format!(
            "tiny overfit did not improve substantially: {:.6} -> {:.6}",
            report.initial_loss, report.final_loss
        )
        .into());
    }
    if report.predictions != [0, 1] {
        return Err(format!(
            "tiny overfit predictions were {:?}, expected [0, 1]",
            report.predictions
        )
        .into());
    }
    Ok(())
}
