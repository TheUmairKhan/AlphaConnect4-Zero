use std::{
    collections::VecDeque,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};

use burn::prelude::*;

pub struct StepMetrics {
    pub step: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub network_entropy: f32,
    pub target_entropy: f32,
    pub kl_network_to_target: f32,
    pub value_mae: f32,
}

impl StepMetrics {
    pub fn from_batch<B: Backend>(
        step: usize,
        policy_loss: &Tensor<B, 1>,
        value_loss: &Tensor<B, 1>,
        log_network_policy: Tensor<B, 2>,
        target_policy: Tensor<B, 2>,
        value_error: Tensor<B, 2>,
    ) -> Self {
        let network_policy = log_network_policy.clone().exp();
        let network_entropy =
            -(network_policy.clone() * log_network_policy.clone()).sum_dim(1).mean();
        let target_entropy = -(target_policy.clone()
            * (target_policy.clone() + 1e-8).log())
        .sum_dim(1)
        .mean();
        let smoothed_target = (target_policy + 1e-6).div_scalar(1.0 + 7e-6);
        let kl = (network_policy * (log_network_policy - smoothed_target.log()))
            .sum_dim(1)
            .mean();
        let value_mae = value_error.abs().mean();

        let values: Vec<f32> = Tensor::cat(
            vec![
                policy_loss.clone(),
                value_loss.clone(),
                network_entropy,
                target_entropy,
                kl,
                value_mae,
            ],
            0,
        )
        .into_data()
        .iter::<f32>()
        .collect();

        Self {
            step,
            policy_loss: values[0],
            value_loss: values[1],
            network_entropy: values[2],
            target_entropy: values[3],
            kl_network_to_target: values[4],
            value_mae: values[5],
        }
    }
}

pub struct MetricsLogger {
    csv: BufWriter<File>,
    html_path: std::path::PathBuf,
    recent: VecDeque<StepMetrics>,
}

impl MetricsLogger {
    pub fn new(directory: &Path) -> io::Result<Self> {
        let mut csv = BufWriter::new(File::create(directory.join("metrics.csv"))?);
        writeln!(csv, "step,policy_loss,value_loss,network_entropy,mcts_target_entropy,kl_network_to_mcts,value_mae")?;
        Ok(Self {
            csv,
            html_path: directory.join("metrics.html"),
            recent: VecDeque::new(),
        })
    }

    pub fn record(&mut self, metrics: StepMetrics) -> io::Result<()> {
        writeln!(
            self.csv,
            "{},{},{},{},{},{},{}",
            metrics.step,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.network_entropy,
            metrics.target_entropy,
            metrics.kl_network_to_target,
            metrics.value_mae,
        )?;
        self.csv.flush()?;
        self.recent.push_back(metrics);
        if self.recent.len() > 1000 {
            self.recent.pop_front();
        }
        fs::write(&self.html_path, self.dashboard())
    }

    fn dashboard(&self) -> String {
        let mut html = String::from(
            r#"<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Connect 4 training</title><style>body{font-family:system-ui,sans-serif;background:#101827;color:#e5eaf3;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.card{background:#1a2537;border-radius:10px;padding:16px}svg{width:100%;height:auto}h1{font-size:1.5rem}h2{font-size:1rem;margin:0 0 8px}.legend{font-size:.8rem;margin:8px 0;color:#b9c4d6}</style></head><body><h1>Connect 4 training</h1>"#,
        );
        if let Some(last) = self.recent.back() {
            html.push_str(&format!("<p>Step {} · Refreshes every 5 seconds · Showing up to 1,000 recent steps</p>", last.step));
        }
        html.push_str("<main>");
        html.push_str(&self.chart(
            "Loss",
            &[("Policy", "#60a5fa", |m| m.policy_loss), ("Value", "#fb923c", |m| m.value_loss)],
        ));
        html.push_str(&self.chart(
            "Policy entropy",
            &[("Network", "#60a5fa", |m| m.network_entropy), ("MCTS target", "#fb923c", |m| m.target_entropy)],
        ));
        html.push_str(&self.chart(
            "KL(network || smoothed MCTS)",
            &[("KL", "#a78bfa", |m| m.kl_network_to_target)],
        ));
        html.push_str(&self.chart(
            "Value prediction error (MAE)",
            &[("MAE", "#34d399", |m| m.value_mae)],
        ));
        html.push_str("</main></body></html>");
        html
    }

    fn chart(&self, title: &str, series: &[(&str, &str, fn(&StepMetrics) -> f32)]) -> String {
        let max = self.recent.iter().flat_map(|point| series.iter().map(move |(_, _, value)| value(point)))
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        let mut html = format!("<div class=\"card\"><h2>{title}</h2><svg viewBox=\"0 0 640 220\"><path d=\"M45 20 V190 H620\" fill=\"none\" stroke=\"#64748b\"/><text x=\"3\" y=\"25\" fill=\"#b9c4d6\" font-size=\"12\">{max:.2}</text><text x=\"20\" y=\"190\" fill=\"#b9c4d6\" font-size=\"12\">0</text>");
        if let (Some(first), Some(last)) = (self.recent.front(), self.recent.back()) {
            html.push_str(&format!("<text x=\"45\" y=\"210\" fill=\"#b9c4d6\" font-size=\"12\">step {}</text><text x=\"550\" y=\"210\" fill=\"#b9c4d6\" font-size=\"12\">{}</text>", first.step, last.step));
        }
        for (_, color, value) in series {
            html.push_str(&format!("<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\""));
            for (index, point) in self.recent.iter().enumerate() {
                let x = if self.recent.len() == 1 {
                    45.0
                } else {
                    45.0 + 575.0 * index as f32 / (self.recent.len() - 1) as f32
                };
                let y = 190.0 - 170.0 * (value(point) / max).clamp(0.0, 1.0);
                html.push_str(&format!("{x:.1},{y:.1} "));
            }
            html.push_str("\"/>");
            if let Some(last) = self.recent.back() {
                let x = if self.recent.len() == 1 { 45.0 } else { 620.0 };
                let y = 190.0 - 170.0 * (value(last) / max).clamp(0.0, 1.0);
                html.push_str(&format!("<circle cx=\"{x}\" cy=\"{y:.1}\" r=\"3\" fill=\"{color}\"/>"));
            }
        }
        html.push_str("</svg><div class=\"legend\">");
        for (name, color, value) in series {
            if let Some(last) = self.recent.back() {
                html.push_str(&format!("<span style=\"color:{color}\">● {name}: {:.3}</span> &nbsp;", value(last)));
            }
        }
        html.push_str("</div></div>");
        html
    }
}
