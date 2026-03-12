use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray1};
use realfft::RealFftPlanner;

#[pyfunction]
fn stft_spectrogram(
    py: Python,
    signal: PyReadonlyArray1<f32>,
    n_fft: usize,
    hop_length: usize,
) -> Py<PyArray2<f32>> {

    let y = signal.as_slice().unwrap();
    let len = y.len();

    let frames = (len - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;

    let mut spectrogram = vec![vec![0.0f32; freq_bins]; frames];

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let mut input = vec![0.0f32; n_fft];
    let mut spectrum = r2c.make_output_vec();

    for frame in 0..frames {

        let start = frame * hop_length;

        input.copy_from_slice(&y[start..start + n_fft]);

        r2c.process(&mut input, &mut spectrum).unwrap();

        for k in 0..freq_bins {
            spectrogram[frame][k] = spectrum[k].norm();
        }
    }

    PyArray2::from_vec2(py, &spectrogram).unwrap().into()
}

#[pyfunction]
fn generate_hashes(
    peaks: PyReadonlyArray2<u32>,
    fan_value: usize,
    delta_t_min: u32,
    delta_t_max: u32,
    freq_bin_size: u32,
) -> Vec<(u64, u32)> {

    let peaks = peaks.as_array();

    let mut peaks_vec: Vec<(u32,u32)> = peaks
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    peaks_vec.sort_by_key(|x| x.1);

    let total = peaks_vec.len();
    let mut hashes = Vec::with_capacity(total * fan_value);

    for i in 0..total {

        let (f1,t1) = peaks_vec[i];
        let mut valid_pairs = 0;

        for j in i+1..total {

            let (f2,t2) = peaks_vec[j];
            let delta = t2 - t1;

            if delta < delta_t_min {
                continue;
            }

            if delta > delta_t_max {
                break;
            }

            let f1c = f1 / freq_bin_size;
            let f2c = f2 / freq_bin_size;
            let dt = delta / 2;

            const FREQ_BITS:u64 = 9;
            const DELTA_BITS:u64 = 8;

            let hash =
                ((f1c as u64) << (FREQ_BITS+DELTA_BITS))
                | ((f2c as u64) << DELTA_BITS)
                | dt as u64;

            hashes.push((hash,t1));

            valid_pairs +=1;

            if valid_pairs >= fan_value {
                break;
            }
        }
    }

    hashes
}

#[pymodule]
fn rust_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(stft_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(generate_hashes, m)?)?;

    Ok(())
}
